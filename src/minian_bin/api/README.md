# Minian-bin FastAPI Dashboard

This module provides a FastAPI-based backend and React frontend for visualizing the Minian-bin pipeline in real-time.

## Features

- Real-time visualization of spike inference process
- Decoupled frontend and backend architecture
- WebSocket-based real-time updates
- Compatible with Dask for distributed processing
- Session-based connections for multi-user support

## Installation

### Backend Dependencies

```bash
pip install fastapi uvicorn websockets pydantic
```

### Frontend Dependencies (for development)

```bash
cd src/minian_bin/frontend
npm install
```

## Running the Dashboard

### Option 1: Using the Benchmark Script

The easiest way to try the dashboard is to run the benchmark script with the FastAPI dashboard enabled:

```bash
python benchmarks/s00_benchmark_realds.py --fastapi-dashboard --open-browser --subset
```

Options:
- `--fastapi-dashboard`: Enable the FastAPI dashboard
- `--open-browser`: Automatically open the dashboard in a browser
- `--subset`: Use a small subset of data for quicker testing
- `--dashboard-port`: Specify the port for the dashboard (default: 54321)

### Option 2: Standalone Dashboard Server

You can run the dashboard server independently:

```bash
python -m minian_bin.api.server --port 54321
```

Then access the dashboard at `http://localhost:54321`

### Option 3: Programmatic Usage

```python
from minian_bin.pipeline import pipeline_bin

# Create a session ID to group connections
session_id = "my-session-id"

# Run pipeline with FastAPI dashboard
C_bin, S_bin, iter_df = pipeline_bin(
    Y,                           # Input data
    use_fastapi_dashboard=True,  # Enable FastAPI dashboard
    dashboard_session_id=session_id,  # Optional session ID
    # ... other parameters
)
```

## Development

### Running the Frontend in Development Mode

```bash
cd src/minian_bin/frontend
npm start
```

This will start the React development server (usually on port 3000) with hot-reloading.

### Building the Frontend

```bash
cd src/minian_bin/frontend
npm run build
```

The build output will be in the `build` directory, which will be served by the FastAPI server.

## Architecture

- `api/app.py`: FastAPI application
- `api/models/`: Pydantic data models
- `api/routes/`: API endpoints
- `api/websockets/`: WebSocket connection handlers
- `api/dashboard_adapter.py`: Adapter for the existing dashboard interface
- `frontend/`: React frontend application

## Extending the Dashboard

### Adding New Data Types

1. Add a new data type to `DataType` enum in `api/models/dashboard.py`
2. Create a new Pydantic model for your data
3. Update the frontend to handle the new data type

### Adding New Visualization Components

The React frontend is designed to be modular. To add a new visualization:

1. Create a new React component in the frontend
2. Update the WebSocket message handler to process the relevant data
3. Render the new component when appropriate

## Using with Dask

The FastAPI dashboard is fully compatible with Dask distributed processing. The `DashboardAdapter` 
class handles the communication between Dask workers and the WebSocket server.

When using Dask, updates will be automatically sent to the dashboard from any worker, 
maintaining real-time visualization even in distributed processing environments. 