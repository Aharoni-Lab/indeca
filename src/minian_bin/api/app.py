"""FastAPI main application entry point."""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from minian_bin.api.routes import dashboard
from minian_bin.logging_config import get_module_logger

# Initialize logger
logger = get_module_logger("api.app")

# Create FastAPI app
app = FastAPI(
    title="Minian-bin API",
    description="API for minian-bin analysis",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dashboard.router)

# Get the directory of the frontend build
static_dir = Path(os.path.join(os.path.dirname(__file__), "frontend", "build"))
logger.info(f"Looking for frontend files in: {static_dir}")

# Make sure the build directory exists
static_dir.mkdir(exist_ok=True, parents=True)

# Check if index.html exists
index_path = static_dir / "index.html"
if index_path.exists():
    logger.info(f"Found index.html at {index_path}")
    
    # Serve static files if they exist
    static_files_dir = static_dir / "static"
    if static_files_dir.exists():
        logger.info(f"Mounting static files from {static_files_dir}")
        app.mount("/static", StaticFiles(directory=str(static_files_dir)), name="static")
    
    # Root endpoint to serve index.html
    @app.get("/", include_in_schema=False)
    async def root():
        logger.info(f"Serving index.html from {index_path}")
        return FileResponse(index_path)
    
    # Catch-all route to handle client-side routing
    @app.get("/{catch_all:path}")
    async def catch_all(catch_all: str, request: Request):
        # Skip API routes
        if catch_all.startswith("dashboard/") or catch_all.startswith("docs") or catch_all.startswith("openapi.json"):
            raise HTTPException(status_code=404, detail="Not found")
        
        # Check if the file exists in the build directory
        file_path = static_dir / catch_all
        if file_path.exists() and file_path.is_file():
            logger.debug(f"Serving file: {file_path}")
            return FileResponse(file_path)
        
        # If not found, serve index.html for client-side routing
        logger.debug(f"Path {catch_all} not found, serving index.html")
        return FileResponse(index_path)
else:
    logger.warning(f"index.html not found at {index_path}, serving API docs instead")
    
    # Root endpoint redirects to API docs if no frontend is available
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

# Health check endpoint
@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    """Event handler for application startup."""
    logger.info("Starting up the API server")

@app.on_event("shutdown")
async def shutdown_event():
    """Event handler for application shutdown."""
    logger.info("Shutting down the API server") 