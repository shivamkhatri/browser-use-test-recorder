# Browser-Use Recorder

An interaction recorder and test automation tool built on top of browser-use.

## 🎯 Features

- **Web-based UI** for easy interaction recording
- **Real-time interaction recording** (clicks, typing, scrolls, etc.)
- **Live step display** with visual timeline
- **Module creation** for grouping steps
- **WebSocket communication** for real-time updates
- **JSON storage** for sessions and modules
- **Code export** (Python, Cypress)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Start Steel Browser
```bash
docker-compose up -d
```

### 3. Start the UI
```bash
uv run python start.py
```

### 4. Open Browser
Navigate to: http://localhost:8080

## 📁 Architecture

```
browser_use_recorder/
├── server.py      # FastAPI web server
└── ui/            # Web UI
    ├── static/    # JavaScript recorder
    └── templates/ # HTML interface
```

## 🎨 UI Features

- **Step Timeline**: Live updating list of recorded actions
- **Module Creation**: Group related steps into reusable modules  
- **Session Controls**: Start/stop/pause recording
- **Code Export**: Generate Python or Cypress test code
- **Chat Interface**: Interact with assistant

## 📊 Data Models

### RecordedStep
```python
{
    "id": "unique_id",
    "action_type": "click|type|scroll|...",
    "tag": "button", 
    "text": "Login",
    "xpath": "//button[@id='login']",
    "description": "Click on Login button"
}
```

### TestModule
```python
{
    "id": "module_id",
    "name": "Login Flow", 
    "description": "Standard login process",
    "steps": [...]
}
```
