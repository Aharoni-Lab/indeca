import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import DashboardGrid from './components/DashboardGrid';
import Plot from 'react-plotly.js';

// WebSocket connection constants
const WS_URL = `ws://${window.location.hostname}:54321/dashboard/ws`;

// Dashboard component
function App() {
  // Get session ID from URL parameters
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get('session_id');
  const [clientId, setClientId] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [data, setData] = useState(null);
  const webSocket = useRef(null);
  const pollingIntervalRef = useRef(null);
  const dataSourceRef = useRef('none'); // 'websocket' or 'polling' or 'none'

  // State management
  const [traceData, setTraceData] = useState({});
  const [iterationData, setIterationData] = useState({});
  const [kernelData, setKernelData] = useState({});

  // WebSocket reference
  const ws = useRef(null);
  
  // Connect to WebSocket
  useEffect(() => {
    // Create WebSocket connection
    connectWebSocket();
    
    // Extract session_id from URL query params
    const params = new URLSearchParams(window.location.search);
    const urlSessionId = params.get('session_id');
    if (urlSessionId) {
      setSessionId(urlSessionId);
      setDataSourceRef.current = 'polling';
      console.log(`Found session_id in URL: ${urlSessionId}, switching to polling mode`);
    }
    
    // Cleanup on unmount
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);
  
  // Fetch dashboard data from REST API
  const fetchDashboardData = useCallback(async () => {
    if (!sessionId || dataSourceRef.current !== 'polling') return;
    
    try {
      const response = await fetch(`/dashboard/data/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        console.log('Received data from API:', data);
        
        // Update trace data
        if (data.traces) {
          setTraceData(data.traces);
        }
        
        // Update iteration data
        if (data.iterations) {
          setIterationData(data.iterations);
        }
        
        // Update kernel data
        if (data.kernels) {
          setKernelData(data.kernels);
        }
        
        // Update connection status
        setConnectionStatus('connected');
      } else {
        console.error('Error fetching dashboard data:', response.status);
        setConnectionStatus('disconnected');
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setConnectionStatus('disconnected');
    }
  }, [sessionId, dataSourceRef]);
  
  // Set up polling interval for REST API data
  useEffect(() => {
    if (sessionId && dataSourceRef.current === 'polling') {
      // Initial fetch
      fetchDashboardData();
      
      // Set up interval
      const interval = setInterval(fetchDashboardData, 1000);
      return () => clearInterval(interval);
    }
  }, [sessionId, dataSourceRef, fetchDashboardData]);
  
  // Connect WebSocket function
  const connectWebSocket = useCallback(() => {
    if (!sessionId) return;

    try {
      // Include the session_id in the URL
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/dashboard/ws?session_id=${sessionId}`;
      console.log(`Connecting to WebSocket: ${wsUrl}`);
      
      const socket = new WebSocket(wsUrl);
      webSocket.current = socket;
      
      socket.onopen = () => {
        console.log('WebSocket connected');
        setConnectionStatus('connected');
        dataSourceRef.current = 'websocket';
      };
      
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('WebSocket message received:', message);
          
          if (message.event === 'connected') {
            setClientId(message.client_id);
          } else if (message.event === 'data_update') {
            setData(message.data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      socket.onclose = (event) => {
        console.log('WebSocket disconnected:', event);
        setConnectionStatus('disconnected');
        
        // Try to reconnect after a delay
        setTimeout(() => {
          if (dataSourceRef.current === 'websocket') {
            connectWebSocket();
          }
        }, 3000);
      };
      
      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };
    } catch (error) {
      console.error('Error setting up WebSocket:', error);
      setConnectionStatus('error');
    }
  }, [sessionId]);
  
  // Function to poll for data from REST API
  const startPolling = useCallback(() => {
    if (!sessionId) return;
    
    // Stop any existing polling
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }
    
    console.log(`Starting polling for session: ${sessionId}`);
    setConnectionStatus('polling');
    dataSourceRef.current = 'polling';
    
    // Initial data fetch
    fetchDashboardData();
    
    // Set up interval for polling
    pollingIntervalRef.current = setInterval(fetchDashboardData, 1000); // Poll every second
    
    // Return cleanup function
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [sessionId]);

  // Effect to handle connection setup
  useEffect(() => {
    // If no session ID, can't connect
    if (!sessionId) {
      setConnectionStatus('no-session');
      return;
    }
    
    console.log(`Session ID: ${sessionId}`);
    
    // Try WebSocket first, fall back to polling
    try {
      connectWebSocket();
      
      // Set up polling as fallback after a delay if WebSocket doesn't connect
      const pollFallbackTimer = setTimeout(() => {
        if (connectionStatus !== 'connected' && dataSourceRef.current !== 'polling') {
          console.log('WebSocket connection failed, falling back to polling');
          startPolling();
        }
      }, 5000);
      
      return () => {
        clearTimeout(pollFallbackTimer);
        if (webSocket.current) {
          webSocket.current.close();
          webSocket.current = null;
        }
        
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      };
    } catch (error) {
      console.error('Error during connection setup:', error);
      startPolling();
    }
  }, [sessionId, connectWebSocket, startPolling, connectionStatus]);

  // If no session ID provided, try to get one from REST API
  useEffect(() => {
    if (!sessionId) {
      setConnectionStatus('no-session');
    }
  }, [sessionId]);

  // Render traces for each cell
  const renderCellTraces = () => {
    return Object.entries(traceData).map(([uid, data]) => {
      const cellId = Number(uid);
      
      // Skip if we don't have Y data
      if (!data.y) return null;
      
      // Prepare plot data
      const plotData = [
        {
          y: data.y,
          type: 'scatter',
          mode: 'lines',
          name: 'Y',
          line: { color: 'blue' }
        }
      ];
      
      // Add C data if available
      if (data.c) {
        plotData.push({
          y: data.c,
          type: 'scatter',
          mode: 'lines',
          name: 'C',
          line: { color: 'green' }
        });
      }
      
      // Add S data if available
      if (data.s) {
        plotData.push({
          y: data.s,
          type: 'scatter',
          mode: 'lines',
          name: 'S',
          line: { color: 'red' }
        });
      }
      
      // Get iteration data if available
      const iterData = iterationData[cellId] || {};
      const scale = iterData.scale || 'N/A';
      const err = iterData.err || 'N/A';
      const tau_d = iterData.tau_d || 'N/A';
      const tau_r = iterData.tau_r || 'N/A';
      const iter = iterData.iter || 'N/A';
      
      return (
        <div key={cellId} className="trace-container">
          <div className="cell-header">
            <div>Cell {cellId}</div>
            <div>
              Iteration: {iter} | Scale: {typeof scale === 'number' ? scale.toFixed(4) : scale} | 
              Error: {typeof err === 'number' ? err.toFixed(4) : err} | 
              Tau_d: {typeof tau_d === 'number' ? tau_d.toFixed(4) : tau_d} | 
              Tau_r: {typeof tau_r === 'number' ? tau_r.toFixed(4) : tau_r}
            </div>
          </div>
          
          <Plot
            data={plotData}
            layout={{
              autosize: true,
              height: 200,
              margin: { l: 50, r: 30, t: 10, b: 30 },
              showlegend: true,
              legend: { orientation: 'h', y: 1.1 },
              yaxis: { title: 'Value' },
              xaxis: { title: 'Frame' }
            }}
            useResizeHandler={true}
            style={{ width: '100%', height: '100%' }}
          />
          
          {/* Render kernel if available */}
          {kernelData[cellId] && (kernelData[cellId].h || kernelData[cellId].h_fit) && (
            <div style={{ marginTop: '10px' }}>
              <Plot
                data={[
                  ...(kernelData[cellId].h ? [{
                    y: kernelData[cellId].h,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Kernel',
                    line: { color: 'purple' }
                  }] : []),
                  ...(kernelData[cellId].h_fit ? [{
                    y: kernelData[cellId].h_fit,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Kernel Fit',
                    line: { color: 'orange' }
                  }] : [])
                ]}
                layout={{
                  autosize: true,
                  height: 150,
                  margin: { l: 50, r: 30, t: 10, b: 30 },
                  showlegend: true,
                  legend: { orientation: 'h', y: 1.1 },
                  yaxis: { title: 'Value' },
                  xaxis: { title: 'Frame' }
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          )}
        </div>
      );
    });
  };
  
  return (
    <div className="App">
      <header className="App-header">
        <h1>Minian-bin Dashboard</h1>
        <div className="connection-status">
          Status: <span className={`status-${connectionStatus}`}>{connectionStatus}</span>
          {clientId && <span> | Client ID: {clientId}</span>}
          {sessionId && <span> | Session ID: {sessionId}</span>}
        </div>
      </header>
      <main>
        {data ? (
          <DashboardGrid data={data} />
        ) : (
          <div className="loading-container">
            {connectionStatus === 'no-session' ? (
              <p>No session ID provided. Add ?session_id=YOUR_SESSION_ID to the URL.</p>
            ) : (
              <p>Loading dashboard data...</p>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App; 