# Minian-bin Dashboard Implementation Plan

## Project Overview

### Background
The current minian-bin dashboard functionality is tightly integrated with the neural signal processing pipeline. It uses Panel and Plotly for visualization but lacks a decoupled architecture that would allow for more flexible and extensible dashboard development. The current dashboard is primarily used to visualize the progress and results of the binary pursuit pipeline during execution.

### Goals
1. Decouple the dashboard functionality from the core processing pipeline
2. Create a modern web-based frontend for visualization
3. Design a flexible API that allows for custom dashboard development
4. Maintain all current visualization capabilities while improving the user experience
5. Enable real-time monitoring of pipeline execution

### Key Technical Requirements
1. **Backward Compatibility**: All functionality and visualizations in the current dashboard must be preserved
2. **Real-time Updates**: The new dashboard must support real-time progress monitoring
3. **Extensibility**: The architecture must support custom visualization components
4. **Performance**: The dashboard must handle large datasets efficiently
5. **Usability**: The UI must be intuitive and improve upon the current dashboard

## Architecture Design

### System Architecture Overview
```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  Minian-bin     │──────▶  FastAPI        │◀─────▶  React Frontend │
│  Pipeline       │      │  Backend        │      │                 │
│                 │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
                                 │
                                 │
                         ┌───────▼───────┐
                         │               │
                         │  Data Storage │
                         │               │
                         └───────────────┘
```

### Backend Architecture
The backend will be built using FastAPI and will consist of the following components:

1. **REST API Layer**:
   - Pipeline configuration and control endpoints
   - Data retrieval endpoints for processed results
   - Authentication and authorization (future)

2. **WebSocket Layer**:
   - Real-time updates during pipeline execution
   - Status and progress notifications

3. **Pipeline Adapter Layer**:
   - Interface between the API and the minian-bin pipeline
   - Data transformation and formatting

4. **Data Management Layer**:
   - Storage of pipeline configurations
   - Caching of results and intermediate data
   - Export/import functionality

### Frontend Architecture
The frontend will be built using React and will consist of the following components:

1. **Dashboard Framework**:
   - Layout management
   - View configuration
   - User preferences

2. **Visualization Components**:
   - Cell trace visualizations
   - Kernel visualizations
   - Error metric visualizations
   - Timeline and progress visualizations

3. **Control Components**:
   - Pipeline configuration forms
   - Execution controls
   - Filter and selection tools

4. **Data Management**:
   - State management (Redux or Context API)
   - Data fetching and caching (React Query)
   - WebSocket integration

## Migration Strategy

### Current Dashboard Assessment
The current dashboard implementation in `src/minian_bin/dashboard.py` is tightly coupled with the pipeline execution in the following ways:

1. **Direct Integration**: The `Dashboard` class is instantiated directly within the `pipeline_bin` function
2. **Real-time Updates**: Dashboard updates occur throughout the pipeline execution
3. **Visualization Dependencies**: Uses Panel and Plotly which are embedded in the pipeline code
4. **State Management**: Dashboard manages its own state within the pipeline process

### Migration Approach
We will use a phased migration approach to replace the current dashboard with the new decoupled architecture:

#### Phase 1: Data Collection Layer
1. **Create a DataCollector Class**:
   - Implement a new `DataCollector` class that mimics the data collection functionality of the current dashboard
   - This class will not include any visualization logic, only data collection
   - Store collected data in a standardized format for API consumption

2. **Modify Pipeline Code**:
   - Update the `pipeline_bin` function to accept either a `Dashboard` (legacy) or a `DataCollector` (new)
   - Add a feature flag parameter like `use_legacy_dashboard` (default: True)
   - Implement conditional logic to use either approach

3. **Add API Bridge**:
   - Create an API bridge that translates between the `DataCollector` and the API endpoints
   - Implement the necessary serialization/deserialization logic

#### Phase 2: Dashboard Deprecation
1. **Feature Flag Introduction**:
   - Add documentation about the new dashboard and feature flag
   - Mark the current dashboard as "deprecated" in docstrings and comments
   - Create migration guides for current users

2. **Dual-Support Period**:
   - Support both dashboard approaches for a defined period (e.g., 2 versions)
   - Log usage of the legacy dashboard to track adoption
   - Provide automated tools to help users migrate

3. **Test Suite Development**:
   - Create tests to ensure data equivalence between old and new dashboards
   - Verify that all existing functionality is preserved in the new dashboard
   - Test backward compatibility for different use cases

#### Phase 3: Complete Replacement
1. **Remove Legacy Dashboard**:
   - Change default value of `use_legacy_dashboard` to False
   - Schedule removal of legacy dashboard code in a future release
   - Update all documentation to reflect the new approach only

2. **Code Cleanup**:
   - Remove all Panel dependencies from the core package if not used elsewhere
   - Refactor pipeline code to use only the data collector approach
   - Clean up any interim compatibility code

3. **Final Migration Support**:
   - Create tools to convert any saved dashboard states to the new format
   - Provide migration scripts for users with custom dashboards

### Code Changes Required

1. **Update pipeline_bin Function**:
```python
@profile
def pipeline_bin(
    Y,
    use_legacy_dashboard=True,  # <-- New parameter
    data_collector=None,        # <-- New parameter
    up_factor=1,
    # ... existing parameters
):
    # ... existing code ...

    # Create visualization component based on configuration
    if use_legacy_dashboard:
        logger.debug("Using legacy dashboard (deprecated)")
        try:
            dashboard = Dashboard(Y=Y, kn_len=ar_kn_len)
            # ... existing dashboard code ...
        except Exception as e:
            logger.error(f"Failed to create dashboard: {str(e)}")
            raise
    else:
        logger.debug("Using data collector for API-based dashboard")
        if data_collector is None:
            data_collector = DataCollector(Y=Y, kn_len=ar_kn_len, max_iters=max_iters)
        
        # Use data_collector instead of dashboard
        # ...
```

2. **Create DataCollector Class**:
```python
class DataCollector:
    """
    Collects data during pipeline execution for API consumption.
    Replaces the Dashboard class with a non-visual data collection mechanism.
    """
    def __init__(self, Y=None, ncell=None, T=None, max_iters=20, kn_len=60):
        # Initialize data structures similar to Dashboard but without UI components
        if Y is None:
            assert ncell is not None and T is not None
            Y = np.ones((ncell, T))
        else:
            ncell, T = Y.shape
            
        self.Y = Y
        self.ncell = ncell
        self.T = T
        self.kn_len = kn_len
        self.max_iters = max_iters
        self.it_update = 0
        self.it_view = 0
        self.it_vars = {
            # Same data structures as Dashboard but without UI components
            "c": np.full((max_iters, ncell, T), np.nan),
            "s": np.full((max_iters, ncell, T), np.nan),
            "h": np.full((max_iters, ncell, kn_len), np.nan),
            "h_fit": np.full((max_iters, ncell, kn_len), np.nan),
            "scale": np.full((max_iters, ncell), np.nan),
            "tau_d": np.full((max_iters, ncell), np.nan),
            "tau_r": np.full((max_iters, ncell), np.nan),
            "err": np.full((max_iters, ncell), np.nan),
            "penal_err": np.array(
                [
                    [{"penal": [], "scale": [], "err": []} for _ in range(ncell)]
                    for _ in range(max_iters)
                ]
            ),
        }
    
    # Implement update methods similar to Dashboard but without UI updates
    def update(self, uid=None, **kwargs):
        # Similar implementation to Dashboard.update but without UI updates
        pass
    
    def set_iter(self, it):
        # Similar implementation to Dashboard.set_iter but without UI updates
        self.it_update = it
        
    # Add methods to export data for API
    def get_iteration_data(self, iteration=None):
        """Get data for a specific iteration or the current iteration."""
        it = iteration if iteration is not None else self.it_update
        return {
            "iteration": it,
            "cells": [self.get_cell_data(i, it) for i in range(self.ncell)],
            "metrics": self.get_metrics(it),
        }
    
    def get_cell_data(self, cell_id, iteration=None):
        """Get data for a specific cell and iteration."""
        it = iteration if iteration is not None else self.it_update
        return {
            "cell_id": cell_id,
            "y": self.Y[cell_id].tolist(),
            "s": self.it_vars["s"][it, cell_id].tolist(),
            "c": self.it_vars["c"][it, cell_id].tolist(),
            "h": self.it_vars["h"][it, cell_id].tolist(),
            "h_fit": self.it_vars["h_fit"][it, cell_id].tolist(),
            "metrics": {
                "tau_d": self.it_vars["tau_d"][it, cell_id].item(),
                "tau_r": self.it_vars["tau_r"][it, cell_id].item(),
                "scale": self.it_vars["scale"][it, cell_id].item(),
                "err": self.it_vars["err"][it, cell_id].item(),
            },
        }
    
    def get_metrics(self, iteration=None):
        """Get metrics for a specific iteration."""
        it = iteration if iteration is not None else self.it_update
        return {
            "tau_d": self.it_vars["tau_d"][it].tolist(),
            "tau_r": self.it_vars["tau_r"][it].tolist(),
            "scale": self.it_vars["scale"][it].tolist(),
            "err": self.it_vars["err"][it].tolist(),
        }
```

### Timeline for Dashboard Replacement

1. **Preparation Phase (2 weeks)**:
   - Analyze current dashboard code and identify all dependencies
   - Design the `DataCollector` class
   - Create plan for necessary changes to pipeline code

2. **Implementation Phase (4 weeks)**:
   - Implement `DataCollector` class
   - Modify pipeline code to support both approaches
   - Add feature flags and deprecation notices

3. **Dual Support Phase (2-3 months)**:
   - Support both dashboard approaches
   - Monitor usage and collect feedback
   - Make improvements based on user experience

4. **Transition Phase (1 month)**:
   - Change default to new dashboard approach
   - Finalize all documentation updates
   - Provide migration assistance to users

5. **Completion Phase (2 weeks)**:
   - Remove legacy dashboard code
   - Clean up any remaining compatibility code
   - Verify all functionality is preserved

## Implementation Plan

### Phase 1: Backend Development

#### 1.1 Core API Design
- **Task 1.1.1**: Define API endpoints and data models
  - Create OpenAPI specifications for all endpoints
  - Design Pydantic models for requests and responses
  - Document API versioning strategy

- **Task 1.1.2**: Implement authentication system (if required)
  - OAuth2 with Password flow for user authentication
  - JWT token management
  - Role-based permissions

- **Task 1.1.3**: Set up project structure
  - Initialize FastAPI application
  - Configure middleware (CORS, authentication, logging)
  - Set up dependency injection system

#### 1.2 Pipeline Adapter Development
- **Task 1.2.1**: Create DataCollector class
  - Replace current Dashboard class with a data collection mechanism
  - Implement callbacks for pipeline progress
  - Store intermediate results in a structured format

- **Task 1.2.2**: Modify pipeline_bin function
  - Add optional data_collector parameter
  - Update the pipeline to use the collector instead of dashboard
  - Ensure backward compatibility

- **Task 1.2.3**: Implement pipeline execution service
  - Create background task system for pipeline execution
  - Implement job queue for multiple pipeline runs
  - Add monitoring and error handling

#### 1.3 Data Management Implementation
- **Task 1.3.1**: Design data storage schema
  - Define storage format for pipeline results
  - Create database models (if using a DB)
  - Design caching strategy

- **Task 1.3.2**: Implement data access layer
  - Create services for data retrieval and storage
  - Implement query optimization for large datasets
  - Add data validation and transformation

- **Task 1.3.3**: Add export/import functionality
  - Implement data export in multiple formats (CSV, JSON, etc.)
  - Create import validation and processing
  - Add batch processing capabilities

#### 1.4 WebSocket Implementation
- **Task 1.4.1**: Set up WebSocket connections
  - Create connection manager for multiple clients
  - Implement authentication for WebSocket connections
  - Add connection lifecycle management

- **Task 1.4.2**: Design notification system
  - Create pipeline event system
  - Implement message formatting
  - Add client-specific filtering

- **Task 1.4.3**: Connect WebSockets to pipeline execution
  - Integrate with pipeline adapter
  - Implement progress reporting
  - Add error notifications

### Phase 2: Frontend Development

#### 2.1 Project Setup
- **Task 2.1.1**: Initialize React application
  - Set up project with Create React App or Vite
  - Configure build system
  - Add testing framework

- **Task 2.1.2**: Set up state management
  - Configure Redux or Context API
  - Create data models and reducers
  - Implement initial state and actions

- **Task 2.1.3**: Implement API client
  - Create API service layer
  - Set up React Query for data fetching
  - Implement error handling and retry logic

#### 2.2 Dashboard Framework
- **Task 2.2.1**: Create dashboard layout system
  - Implement responsive grid layout
  - Create panel and card components
  - Add drag-and-drop functionality for customization

- **Task 2.2.2**: Implement navigation
  - Create sidebar and navigation menu
  - Implement routing
  - Add breadcrumbs and context navigation

- **Task 2.2.3**: Add user preferences
  - Create theme switching
  - Implement layout saving
  - Add accessibility features

#### 2.3 Visualization Components
- **Task 2.3.1**: Implement cell trace visualizations
  - Create Plotly.js components for cell traces
  - Add zoom, pan, and selection tools
  - Implement color schemes and styling

- **Task 2.3.2**: Create kernel visualization components
  - Implement kernel plotting
  - Add comparison tools
  - Create parameter visualization

- **Task 2.3.3**: Implement error metric visualizations
  - Create heatmap visualization
  - Implement line charts for convergence
  - Add tooltips and annotations

- **Task 2.3.4**: Add timeline and progress visualizations
  - Create iteration slider
  - Implement progress bars
  - Add animation controls

#### 2.4 Control Components
- **Task 2.4.1**: Implement pipeline configuration forms
  - Create dynamic form builder
  - Add validation and error handling
  - Implement configuration presets

- **Task 2.4.2**: Create execution controls
  - Implement start/stop buttons
  - Add parameter adjustment during execution
  - Create execution history

- **Task 2.4.3**: Add filter and selection tools
  - Implement cell selection tools
  - Create filtering controls
  - Add search functionality

#### 2.5 WebSocket Integration
- **Task 2.5.1**: Implement WebSocket client
  - Create connection management
  - Handle reconnection and errors
  - Add message parsing

- **Task 2.5.2**: Create real-time updates
  - Implement data streaming to visualization components
  - Create notification system
  - Add progress updates

### Phase 3: Integration and Testing

#### 3.1 Backend-Frontend Integration
- **Task 3.1.1**: Connect frontend to backend API
  - Test all API endpoints
  - Verify data format compatibility
  - Implement error handling

- **Task 3.1.2**: Test WebSocket functionality
  - Verify real-time updates
  - Test reconnection logic
  - Measure performance under load

- **Task 3.1.3**: End-to-end testing
  - Create test cases for common workflows
  - Automate testing with Cypress or similar
  - Document test results

#### 3.2 Pipeline Integration Testing
- **Task 3.2.1**: Test pipeline execution
  - Verify results match original dashboard
  - Test with various parameter configurations
  - Measure performance impact

- **Task 3.2.2**: Test real-time visualization
  - Compare with original dashboard
  - Verify all metrics are displayed correctly
  - Test with large datasets

#### 3.3 User Experience Testing
- **Task 3.3.1**: Conduct usability testing
  - Gather feedback from current users
  - Test navigation and workflow
  - Identify UX improvements

- **Task 3.3.2**: Performance testing
  - Measure load times and responsiveness
  - Test with various device types
  - Optimize for performance bottlenecks

### Phase 4: Deployment and Documentation

#### 4.1 Deployment Setup
- **Task 4.1.1**: Create Docker configuration
  - Build Docker images for backend and frontend
  - Create Docker Compose configuration
  - Test containerized deployment

- **Task 4.1.2**: Set up CI/CD pipeline
  - Configure GitHub Actions or similar
  - Implement automated testing in pipeline
  - Add automated deployment

#### 4.2 Documentation
- **Task 4.2.1**: Create API documentation
  - Generate OpenAPI documentation
  - Create usage examples
  - Document authentication and security

- **Task 4.2.2**: Write frontend documentation
  - Document component usage
  - Create customization guide
  - Write developer onboarding guide

- **Task 4.2.3**: Create user documentation
  - Write user manual
  - Create tutorial videos
  - Document common workflows

## Development Setup and Environment

### Backend Development Environment

#### Required Dependencies
```bash
# Core dependencies
pip install fastapi uvicorn sqlalchemy pydantic
pip install python-jose pyjwt passlib
pip install websockets
pip install numpy pandas xarray

# Development dependencies
pip install pytest pytest-cov black isort mypy
```

#### Directory Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration settings
│   ├── dependencies.py      # Dependency injection
│   ├── logging_config.py    # Logging configuration
│   ├── api/                 # API endpoints
│   │   ├── __init__.py
│   │   ├── router.py        # Main API router
│   │   ├── pipelines.py     # Pipeline endpoints
│   │   └── data.py          # Data retrieval endpoints
│   ├── core/                # Core business logic
│   │   ├── __init__.py
│   │   ├── security.py      # Authentication and authorization
│   │   └── events.py        # Application event handlers
│   ├── db/                  # Database models and repositories
│   │   ├── __init__.py
│   │   ├── models.py        # SQLAlchemy models
│   │   └── repositories.py  # Data access layer
│   ├── schemas/             # Pydantic models for API
│   │   ├── __init__.py
│   │   ├── pipeline.py      # Pipeline schemas
│   │   └── data.py          # Data schemas
│   ├── services/            # Business services
│   │   ├── __init__.py
│   │   ├── pipeline.py      # Pipeline execution service
│   │   └── data.py          # Data processing service
│   └── utils/               # Utility functions
│       ├── __init__.py
│       └── helpers.py       # Helper functions
├── tests/                   # Test files
│   ├── __init__.py
│   ├── conftest.py          # Test configuration
│   ├── test_api/            # API tests
│   └── test_services/       # Service tests
└── alembic/                 # Database migrations
```

#### Running the Backend Locally
```bash
# Run the FastAPI server with hot reload
uvicorn app.main:app --reload --port 8000

# Run tests
pytest

# Generate API documentation
# Visit http://localhost:8000/docs after starting the server
```

### Frontend Development Environment

#### Required Dependencies
```bash
# Create React app with TypeScript
npx create-react-app dashboard-frontend --template typescript

# Install core dependencies
npm install react-router-dom @tanstack/react-query
npm install @reduxjs/toolkit react-redux
npm install plotly.js react-plotly.js
npm install @mui/material @emotion/react @emotion/styled

# Development dependencies
npm install -D prettier eslint-config-prettier cypress
```

#### Directory Structure
```
frontend/
├── public/                  # Static assets
├── src/
│   ├── App.tsx              # Main application component
│   ├── index.tsx            # Application entry point
│   ├── api/                 # API client
│   │   ├── index.ts         # API client setup
│   │   ├── pipeline.ts      # Pipeline API
│   │   └── websocket.ts     # WebSocket client
│   ├── components/          # Reusable components
│   │   ├── layout/          # Layout components
│   │   ├── visualizations/  # Visualization components
│   │   └── controls/        # UI control components
│   ├── pages/               # Page components
│   │   ├── Dashboard.tsx    # Main dashboard page
│   │   ├── Configuration.tsx # Configuration page
│   │   └── Results.tsx      # Results page
│   ├── store/               # Redux store
│   │   ├── index.ts         # Store configuration
│   │   └── slices/          # Redux slices
│   ├── hooks/               # Custom hooks
│   ├── utils/               # Utility functions
│   └── types/               # TypeScript type definitions
├── cypress/                 # End-to-end tests
│   └── integration/         # Test files
└── tests/                   # Unit and integration tests
    └── components/          # Component tests
```

#### Running the Frontend Locally
```bash
# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build
```

## Testing Strategy

### Unit Testing

#### Backend Unit Tests
- Test each service function in isolation
- Mock external dependencies
- Test edge cases and error handling
- Aim for >80% code coverage

#### Frontend Unit Tests
- Test React components in isolation
- Test Redux reducers and actions
- Test custom hooks
- Use React Testing Library for component tests

### Integration Testing

#### Backend Integration Tests
- Test API endpoints with test database
- Test WebSocket communication
- Test database operations
- Test authentication and authorization

#### Frontend Integration Tests
- Test component interaction
- Test data fetching with mocked API
- Test user workflows
- Test state management

### End-to-End Testing

#### Cypress Tests
- Test complete user workflows
- Test real API communication
- Test responsive design
- Test browser compatibility

### Performance Testing

- Test API response times
- Test WebSocket performance with multiple clients
- Test frontend rendering performance
- Test large dataset handling

## Specific Implementation Guidance for AI Agents

### For Implementing the DataCollector

1. Study the current Dashboard class implementation in `src/minian_bin/dashboard.py`.
2. Identify all data collection and storage functionality.
3. Create a new DataCollector class that maintains the same data structure but removes all UI-related code.
4. Ensure the API aligns with the current Dashboard API for seamless integration.
5. Add serialization methods for API consumption.

### For Modifying pipeline_bin

1. Analyze all Dashboard usage in the pipeline_bin function.
2. Identify all update points and data flows.
3. Add the feature flag and conditional logic.
4. Test with both approaches to ensure no functionality is lost.
5. Ensure error handling is robust for both approaches.

### For Creating the FastAPI Backend

1. Start with the core API endpoints for pipeline execution and data retrieval.
2. Implement WebSockets for real-time updates.
3. Create data models that match the output of the DataCollector.
4. Ensure proper error handling and validation.
5. Document all endpoints with OpenAPI comments.

### For Creating the React Frontend

1. Begin with the basic dashboard layout and navigation.
2. Implement the core visualization components.
3. Add real-time updates via WebSockets.
4. Implement user controls and configuration.
5. Add responsive design and accessibility features.

## Potential Challenges and Solutions

### Challenge 1: Maintaining Backward Compatibility
**Solution**: Implement feature flags and thorough testing to ensure the new approach works alongside the old one without breaking changes.

### Challenge 2: Real-time Performance
**Solution**: Optimize WebSocket communication, implement efficient data serialization, and use client-side caching.

### Challenge 3: Large Dataset Handling
**Solution**: Implement pagination, data compression, and lazy loading for large datasets.

### Challenge 4: User Adoption
**Solution**: Create comprehensive documentation, provide migration guides, and maintain feature parity.

### Challenge 5: Integration Complexity
**Solution**: Use a modular approach, implement clear interfaces, and develop incrementally with frequent testing.

## Success Criteria

The migration will be considered successful when:

1. **Functional Equivalence**: All visualization capabilities of the current dashboard are preserved.
2. **Decoupled Architecture**: The dashboard is completely decoupled from the pipeline code.
3. **Performance**: The new dashboard performs at least as well as the current one.
4. **User Adoption**: Users are able to transition to the new dashboard without issues.
5. **Extensibility**: New visualization components can be added without modifying the core pipeline.

## References and Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/getting-started.html)
- [Plotly.js Documentation](https://plotly.com/javascript/)
- [WebSockets in FastAPI](https://fastapi.tiangolo.com/advanced/websockets/)

### Related Codebases and Examples
- [Plotly Dash](https://dash.plotly.com/) - For visualization patterns
- [React-Plotly](https://github.com/plotly/react-plotly.js/) - For React integration with Plotly
- [FastAPI WebSocket Chat Example](https://github.com/tiangolo/fastapi/tree/master/docs_src/websockets/tutorial001.py) - For WebSocket implementation

### Learning Resources
- [Building Data Visualization Tools Course](https://www.coursera.org/learn/data-visualization-tools)
- [FastAPI Course](https://testdriven.io/courses/tdd-fastapi/)
- [React Hooks and Context API Tutorial](https://www.digitalocean.com/community/tutorials/react-usecontext)

## Next Steps

1. Review and refine this implementation plan
2. Set up development environment
3. Begin implementation of Phase 1
4. Schedule regular progress reviews and updates 