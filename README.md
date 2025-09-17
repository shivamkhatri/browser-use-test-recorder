# FlowForge - A hybrid recorder to supercharge browser-use for Contextual AI Actions and QA.
---

### Contextual AI Actions

<img width="1024" height="490" alt="Image" src="https://github.com/user-attachments/assets/65ba44fd-0e0c-44fb-a109-1152643b7d6b" />

https://github.com/user-attachments/assets/b87a322e-b7c2-4d63-8248-fe1c2d22aad8

---

## What does it do?

AI copilots today are **stateless**—they only respond prompt-by-prompt, without understanding the larger workflow. This forces users to repeatedly re-explain context instead of letting AI learn from the full journey.

## Problem

* Current copilots can’t **observe user actions** across steps.
* They react only to a single screen, not the end-to-end process.
* As a result, they can’t recommend meaningful, high-level automations.

## Solution: Hybrid Flow Recorder

A **Flow Recorder** that captures both:

1. **Manual actions** by the user
2. **AI agent-driven actions**

This creates a unified sequential log, which a **Contextual AI engine** uses to recommend relevant next steps. Once approved, the browser-use agent executes them.

## Examples

* **Context-aware recommendation**: While copying tasks into a calendar, AI suggests: *“Create calendar events automatically?”*
* **Proactive completion**: While browsing apartments, AI offers: *“Compare top 3 apartments”* or *“Schedule a viewing.”*

## For QA & Automation

* Flows export as JSON → generate Cypress/Playwright/Selenium tests.
* Reuse flows as manual/agent actions in **browser-use**.
* Integrates into **workflow-use** and **QA-use** for a best-in-class no-code/low-code automation platform.


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
