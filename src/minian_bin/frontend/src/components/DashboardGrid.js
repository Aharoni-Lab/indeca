import React from 'react';
import Plot from 'react-plotly.js';
import { Container, Row, Col, Card } from 'react-bootstrap';
import '../styles/DashboardGrid.css';

const DashboardGrid = ({ data }) => {
  // If no data, show loading
  if (!data) {
    return <div className="loading">Loading dashboard data...</div>;
  }

  // Extract cell data from the data object
  const cellIds = Object.keys(data.traces || {});

  if (cellIds.length === 0) {
    return <div className="no-data">No cell data available yet. Processing may still be in progress...</div>;
  }

  return (
    <Container fluid className="dashboard-grid">
      <h2>Dashboard Data - {cellIds.length} Cells</h2>
      
      {cellIds.map((cellId) => {
        const cellTraces = data.traces[cellId] || {};
        const cellKernels = data.kernels?.[cellId] || {};
        const cellIterations = data.iterations?.[cellId] || {};
        
        // Skip cells that don't have complete data yet
        if (!cellTraces.y) return null;
        
        // Create trace data for Plotly
        const traces = [];
        
        // Y values (original)
        if (cellTraces.y) {
          traces.push({
            y: cellTraces.y,
            type: 'scatter',
            mode: 'lines',
            name: 'Raw',
            line: { color: '#333' }
          });
        }
        
        // C values (deconvolved)
        if (cellTraces.c) {
          traces.push({
            y: cellTraces.c,
            type: 'scatter',
            mode: 'lines',
            name: 'Calcium',
            line: { color: '#1f77b4' }
          });
        }
        
        // S values (spikes)
        if (cellTraces.s) {
          traces.push({
            y: cellTraces.s,
            type: 'scatter',
            mode: 'lines',
            name: 'Spikes',
            line: { color: '#d62728' }
          });
        }
        
        // Kernel traces for Plotly
        const kernelTraces = [];
        
        // Kernel h
        if (cellKernels.h) {
          kernelTraces.push({
            y: cellKernels.h,
            type: 'scatter',
            mode: 'lines',
            name: 'Kernel',
            line: { color: '#ff7f0e' }
          });
        }
        
        // Kernel h_fit
        if (cellKernels.h_fit) {
          kernelTraces.push({
            y: cellKernels.h_fit,
            type: 'scatter',
            mode: 'lines',
            name: 'Kernel Fit',
            line: { color: '#2ca02c' }
          });
        }
        
        return (
          <Card key={cellId} className="cell-card mb-4">
            <Card.Header>
              <h3>Cell {cellId}</h3>
              {cellIterations.iter !== undefined && (
                <div className="iteration-info">
                  Iteration: {cellIterations.iter}
                </div>
              )}
            </Card.Header>
            <Card.Body>
              <Row>
                <Col md={8}>
                  <div className="plot-container">
                    <Plot
                      data={traces}
                      layout={{
                        title: `Cell ${cellId} Traces`,
                        autosize: true,
                        margin: { l: 50, r: 20, t: 40, b: 40 },
                        xaxis: { title: 'Frame' },
                        yaxis: { title: 'Value' },
                        legend: { orientation: 'h', y: -0.2 }
                      }}
                      useResizeHandler={true}
                      style={{ width: '100%', height: '100%' }}
                      config={{ responsive: true }}
                    />
                  </div>
                </Col>
                <Col md={4}>
                  <div className="plot-container">
                    <Plot
                      data={kernelTraces}
                      layout={{
                        title: 'Kernel',
                        autosize: true,
                        margin: { l: 50, r: 20, t: 40, b: 40 },
                        xaxis: { title: 'Time' },
                        yaxis: { title: 'Value' },
                        legend: { orientation: 'h', y: -0.2 }
                      }}
                      useResizeHandler={true}
                      style={{ width: '100%', height: '100%' }}
                      config={{ responsive: true }}
                    />
                  </div>
                  
                  <div className="parameters-container mt-3">
                    <h4>Parameters</h4>
                    <table className="parameters-table">
                      <tbody>
                        {cellIterations.tau_d !== undefined && (
                          <tr>
                            <td>Tau D:</td>
                            <td>{cellIterations.tau_d.toFixed(4)}</td>
                          </tr>
                        )}
                        {cellIterations.tau_r !== undefined && (
                          <tr>
                            <td>Tau R:</td>
                            <td>{cellIterations.tau_r.toFixed(4)}</td>
                          </tr>
                        )}
                        {cellIterations.scale !== undefined && (
                          <tr>
                            <td>Scale:</td>
                            <td>{cellIterations.scale.toFixed(4)}</td>
                          </tr>
                        )}
                        {cellIterations.err !== undefined && (
                          <tr>
                            <td>Error:</td>
                            <td>{cellIterations.err.toFixed(4)}</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        );
      })}
    </Container>
  );
};

export default DashboardGrid; 