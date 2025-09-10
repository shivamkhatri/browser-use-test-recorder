# Project: **Browser-use Assisted Flow Recorder**

## 🎥 Demo & Launch Tweet

[Tweet announcement / context](https://x.com/ShivamKhatri_/status/1965906282675843116)

<div align="center">
    <video width="800" controls title="Flow Recorder Demo">
        <source src="./demo.mp4" type="video/mp4" />
        Your viewer doesn't support embedded video. Open the clip here: <a href="./demo.mp4">demo.mp4</a>
    </video>
  
    <p>
        If the player doesn't render in your viewer, <a href="./demo.mp4">download / open the demo video</a>.
    </p>
</div>

Key moments in the clip:
1. Start recording & navigation
2. Manual + agent steps captured side-by-side
3. Grouping steps into a module
4. Exporting to Python & Cypress code

---

### Problem Statement

Browser-use is a promising framework for AI-driven automation, but based on my extensive usage for QA automation, I’ve observed several challenges:

* **Instruction following issues:** LLMs like GPT-4.1 often skip essential actions in multi-step tests.
* **Model tradeoffs:** Claude 4 Sonnet handles instruction following better, but it’s slow and expensive (\~\$2.50 for a 25-step test).
* **Wrong element interactions:** If browser-use acts on the wrong element, subsequent steps still continue instead of failing early.
* **Custom UI elements:** Many enterprise apps use in-house icons unfamiliar to LLMs. Agents misidentify them, especially when elements lack useful attributes.
* **No relative selectors:** Unlike Cypress/Selenium, identifying elements relative to nearby elements is hard with browser-use.
* **Debugging difficulty:** Without step-by-step execution during test creation, users write long tasks (20–30 steps) only to discover failures deep inside, wasting time in debugging.

---

### My Idea

I propose a **Flow Recorder** that combines **manual recording** (like workflow-use) with **browser-use agent interactions**, solving the above pain points.

* Users can **record manual interactions** and **agent-driven steps** side by side.
* The **steps export as JSON**, which can:

  * Feed into Cursor/VSCode for generating robust test automation code (Cypress, Selenium, Playwright).
  * Be reused as **initial/final/manual actions** for browser-use.

This bridges the gap between **human precision** and **AI-driven automation**.

---

### Why This Matters

* **Practicality:** QA engineers are already used to manual recorders (LambdaTest, KaneAI, etc.), validating this idea.
* **Flexibility:** Manual + Agent interaction recording reduces reliance on LLM correctness alone.
* **Debuggability:** Step-by-step creation and export make tests easier to validate and maintain.
* **Adoption:** Enterprises can gradually adopt browser-use by combining familiar workflows with AI assistance.

---

### Future Work

* Integrate visual assertions so execution halts if an unexpected element interaction occurs.
* Add relative element selection support.
* Optimize cost by minimizing redundant LLM calls.

---

👉 This way, the project demonstrates both **a real-world QA testing problem** and **a practical hybrid solution**.

---


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
