"""Script to run the FastAPI server."""

import argparse
import logging
import os
import sys
import uvicorn

from minian_bin.logging_config import get_module_logger
from minian_bin.api.app import app

# Initialize logger
logger = get_module_logger("api.server")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the FastAPI server")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=54321, 
        help="Port to run the server on"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="info", 
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Run the FastAPI server."""
    args = parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    
    # Run the server
    logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
    uvicorn.run(
        "minian_bin.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main() 