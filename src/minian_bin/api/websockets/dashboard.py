"""WebSocket handler for dashboard connections."""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Set

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from minian_bin.logging_config import get_module_logger

# Initialize logger for this module
logger = get_module_logger("api.websockets.dashboard")


class ConnectionManager:
    """Manager for WebSocket connections."""

    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, Set[str]] = {}
        self.lock = asyncio.Lock()
        logger.info("WebSocket connection manager initialized")

    async def connect(
        self, websocket: WebSocket, client_id: str = None, session_id: str = None
    ) -> str:
        """Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            client_id: Optional client ID, will be generated if not provided
            session_id: Optional session ID to group connections

        Returns:
            The client ID
        """
        await websocket.accept()
        if client_id is None:
            client_id = str(uuid.uuid4())

        async with self.lock:
            self.active_connections[client_id] = websocket

            # Add to session if provided
            if session_id:
                if session_id not in self.session_connections:
                    self.session_connections[session_id] = set()
                self.session_connections[session_id].add(client_id)

        logger.info(f"Client {client_id} connected to WebSocket")
        if session_id:
            logger.info(f"Client {client_id} added to session {session_id}")
        return client_id

    async def disconnect(self, client_id: str, session_id: str = None):
        """Disconnect a client.

        Args:
            client_id: The client ID to disconnect
            session_id: Optional session ID to remove from
        """
        async with self.lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                logger.info(f"Client {client_id} disconnected from WebSocket")

            # Remove from session if provided
            if session_id and session_id in self.session_connections:
                if client_id in self.session_connections[session_id]:
                    self.session_connections[session_id].remove(client_id)
                    logger.info(f"Client {client_id} removed from session {session_id}")

                # Clean up empty sessions
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
                    logger.info(f"Session {session_id} removed (no clients)")

    async def send_personal_message(self, message: dict, client_id: str):
        """Send a message to a specific client.

        Args:
            message: The message to send
            client_id: The client ID to send to
        """
        if client_id in self.active_connections:
            try:
                # Convert numpy arrays to lists for JSON serialization
                json_compatible_message = self._prepare_message(message)
                await self.active_connections[client_id].send_json(
                    json_compatible_message
                )
                logger.debug(f"Message sent to client {client_id}")
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                await self.disconnect(client_id)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients.

        Args:
            message: The message to broadcast
        """
        disconnected_clients = []

        # Convert numpy arrays to lists for JSON serialization
        json_compatible_message = self._prepare_message(message)

        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(json_compatible_message)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)

    async def broadcast_to_session(self, message: dict, session_id: str):
        """Broadcast a message to all clients in a session.

        Args:
            message: The message to broadcast
            session_id: The session ID to broadcast to
        """
        if session_id not in self.session_connections:
            logger.warning(
                f"Attempted to broadcast to non-existent session {session_id}"
            )
            return

        # Convert numpy arrays to lists for JSON serialization
        json_compatible_message = self._prepare_message(message)

        clients = list(self.session_connections[session_id])
        for client_id in clients:
            await self.send_personal_message(json_compatible_message, client_id)

    def _prepare_message(self, message: dict) -> dict:
        """Prepare a message for JSON serialization.

        Args:
            message: The message to prepare

        Returns:
            JSON-compatible message
        """
        if isinstance(message, dict):
            result = {}
            for key, value in message.items():
                result[key] = self._prepare_message(value)
            return result
        elif isinstance(message, list):
            return [self._prepare_message(item) for item in message]
        elif isinstance(message, np.ndarray):
            return message.tolist()
        elif isinstance(message, (np.int32, np.int64)):
            return int(message)
        elif isinstance(message, (np.float32, np.float64)):
            return float(message)
        else:
            return message


# Global connection manager instance
connection_manager = ConnectionManager()
