"""API routes for the dashboard."""

from fastapi import APIRouter, Depends, HTTPException, Path, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uuid

from minian_bin.api.models.dashboard import DashboardConnection
from minian_bin.api.websockets.dashboard import connection_manager
from minian_bin.logging_config import get_module_logger

# Import dashboard data from fixed adapter
try:
    from minian_bin.api.dashboard_adapter_fixed import get_session_data, get_all_sessions
    HAS_FIXED_ADAPTER = True
except ImportError:
    HAS_FIXED_ADAPTER = False

# Initialize logger for this module
logger = get_module_logger("api.routes.dashboard")

# Create router
router = APIRouter(prefix="/dashboard", tags=["dashboard"])


class SessionInfo(BaseModel):
    """Information about a session."""
    session_id: str
    client_count: int


@router.get("/sessions", response_model=list[SessionInfo])
async def get_sessions():
    """Get information about all active sessions."""
    sessions = []
    
    # Add sessions from WebSocket connections
    for session_id, clients in connection_manager.session_connections.items():
        sessions.append(SessionInfo(
            session_id=session_id,
            client_count=len(clients)
        ))
    
    # Add sessions from file storage
    if HAS_FIXED_ADAPTER:
        for session_id in get_all_sessions():
            # Only add if not already in the list
            if not any(s.session_id == session_id for s in sessions):
                sessions.append(SessionInfo(
                    session_id=session_id,
                    client_count=0  # No WebSocket clients for this session
                ))
    
    return sessions


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str = Path(...)):
    """Get information about a specific session."""
    client_count = 0
    
    # Check WebSocket connections
    if session_id in connection_manager.session_connections:
        client_count = len(connection_manager.session_connections[session_id])
    # Check file storage
    elif HAS_FIXED_ADAPTER:
        data = get_session_data(session_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return SessionInfo(
        session_id=session_id,
        client_count=client_count
    )


@router.post("/sessions", response_model=SessionInfo)
async def create_session():
    """Create a new session."""
    session_id = str(uuid.uuid4())
    connection_manager.session_connections[session_id] = set()
    logger.info(f"Created new session: {session_id}")
    
    return SessionInfo(
        session_id=session_id,
        client_count=0
    )


@router.get("/data/{session_id}")
async def get_dashboard_data(session_id: str = Path(...)):
    """Get dashboard data for a specific session.
    
    This endpoint retrieves data from the fixed dashboard adapter
    which is compatible with Dask distributed processing.
    """
    if not HAS_FIXED_ADAPTER:
        raise HTTPException(status_code=501, detail="Fixed dashboard adapter not available")
    
    data = get_session_data(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return data


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(None),
    session_id: str = Query(None)
):
    """WebSocket endpoint for dashboard connections."""
    try:
        client_id = await connection_manager.connect(
            websocket=websocket,
            client_id=client_id,
            session_id=session_id
        )
        
        # Send connection confirmation
        await connection_manager.send_personal_message(
            {"event": "connected", "client_id": client_id, "session_id": session_id},
            client_id
        )
        
        # Listen for messages from the client
        while True:
            try:
                # Wait for messages from the client
                data = await websocket.receive_json()
                logger.debug(f"Received message from client {client_id}: {data}")
                
                # Echo message back (can be expanded for more functionality)
                await connection_manager.send_personal_message(
                    {"event": "echo", "data": data},
                    client_id
                )
            except Exception as e:
                logger.error(f"Error processing message from client {client_id}: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    finally:
        await connection_manager.disconnect(client_id, session_id) 