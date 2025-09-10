"""Web UI server for browser-use with Steel Browser integration."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from browser_use import Browser, Agent, ChatGoogle
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class UIServer:
    def __init__(self, host: str = "localhost", port: int = 8080, steel_ws_url: str = "ws://localhost:3000/"):
        self.host = host
        self.port = port
        self.steel_ws_url = steel_ws_url
        self.app = FastAPI(title="Browser-Use Web UI")
        self.browser: Browser | None = None
        # Persistent agent (created on first chat message)
        self.agent: Agent | None = None
        self._agent_lock = asyncio.Lock()
        self.websocket_connections: List[WebSocket] = []
        self.recording_active: bool = False
        self.chat_messages: List[Dict] = []
        self.recorded_steps: List[Dict] = []
        self.step_counter: int = 0
        self.last_event_time: float = 0
        self.last_event_signature: str = ""
        self.typing_buffer: Dict[str, Dict] = {}  # Track ongoing typing sessions
        self.script_injected: bool = False  # Track if interaction recorder script is already injected
        self.listeners_attached: bool = False # Track if CDP listeners are attached

        # Setup routes
        self._setup_routes()

        # Mount static files
        static_dir = Path(__file__).parent / "ui" / "static"
        static_dir.mkdir(exist_ok=True)
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
    def _setup_routes(self):
        templates_dir = Path(__file__).parent / "ui" / "templates"
        
        templates = Jinja2Templates(directory=str(templates_dir))
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return templates.TemplateResponse("index.html", {"request": request})
            
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.append(websocket)
            logger.info("WebSocket connection established")
            
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    await self._handle_websocket_message(data, websocket)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
        
        @self.app.post("/api/sessions/create")
        async def create_session():
            # If agent already created a browser, use that instead of creating new one
            if self.browser:
                try:
                    # Test if existing browser is still working
                    cdp_session = await self.browser.get_or_create_cdp_session()
                    if cdp_session:
                        # Browser is working, just ensure CDP listeners are set up
                        if not self.listeners_attached:
                            asyncio.create_task(self.setup_cdp_listeners())
                        
                        return JSONResponse(
                            content={
                                "status": "success",
                                "message": "Using existing browser session",
                                "browser_id": getattr(self.browser, 'id', 'shared'),
                                "current_url": "shared_session",
                            }
                        )
                except Exception as e:
                    logger.warning(f"Existing browser session unhealthy: {e}, creating new one")
                    # Clean up old browser
                    try:
                        await self.browser.stop()
                    except Exception:
                        pass
                    self.browser = None
                    self.agent = None  # Reset agent since browser changed
                    self.script_injected = False
                    self.listeners_attached = False

            # Create new browser session
            temp_user_data_dir = tempfile.mkdtemp()
            try:
                self.browser = Browser(
                    cdp_url="ws://localhost:3000",
                    user_data_dir=temp_user_data_dir,
                    headless=False,
                    keep_alive=True  # Keep alive for both recording and agent use
                )
                await self.browser.start()
                
                # Add event listeners for recording browser interactions
                asyncio.create_task(self.setup_cdp_listeners())

                return JSONResponse(
                    content={
                        "status": "success",
                        "message": "Browser session created successfully",
                        "browser_id": self.browser.id,
                        "current_url": "about:blank",
                    }
                )
            except Exception as e:
                logger.error(f"Failed to create browser session: {e}", exc_info=True)
                if os.path.exists(temp_user_data_dir):
                    shutil.rmtree(temp_user_data_dir)
                raise HTTPException(status_code=500, detail=str(e))


        @self.app.post("/api/recording/start")
        async def start_recording():
            if not self.browser:
                raise HTTPException(status_code=400, detail="Browser session not active.")
            # Reset previous steps so a fresh recording always starts clean
            self.recorded_steps = []
            self.step_counter = 0
            self.last_event_time = 0
            self.last_event_signature = ""
            self.typing_buffer = {}
            await self._broadcast({"type": "steps_cleared"})
            self.recording_active = True
            await self._broadcast({"type": "recording_started"})
            return JSONResponse(content={"status": "success", "message": "Recording started"})

        @self.app.post("/api/recording/stop")
        async def stop_recording():
            if not self.browser:
                raise HTTPException(status_code=400, detail="Browser session not active.")
            self.recording_active = False
            await self._broadcast({"type": "recording_stopped"})
            # Clear steps after stopping so they don't reappear on next start
            self.recorded_steps = []
            self.step_counter = 0
            self.last_event_time = 0
            self.last_event_signature = ""
            self.typing_buffer = {}
            await self._broadcast({"type": "steps_cleared"})
            return JSONResponse(content={"status": "success", "message": "Recording stopped"})

        @self.app.get("/api/steps")
        async def get_steps():
            return {"status": "success", "steps": self.recorded_steps}
        
        @self.app.post("/api/sessions/stop")
        async def stop_session():
            """Stop the current browser session."""
            try:
                self.recording_active = False
                if self.browser:
                    try:
                        await self.browser.stop()
                    except Exception:
                        pass
                    self.browser = None
                self.recorded_steps = []
                self.step_counter = 0
                
                await self._broadcast({"type": "session_stopped"})
                await self._broadcast({"type": "recording_stopped"})
                return {"status": "success"}
                
            except Exception as e:
                logger.error(f"Failed to stop session: {e}")
                return {"status": "error", "message": str(e)}

        @self.app.get("/api/chat/messages")
        async def get_chat_messages():
            return {"status": "success", "messages": self.chat_messages}

        @self.app.post("/api/chat/send")
        async def send_chat_message(request: Request):
            """Receive a user chat message, ensure an Agent exists, add task, run it, and stream back result."""
            try:
                data = await request.json()
                message_text: str | None = data.get("message")
                if not message_text:
                    return {"status": "error", "message": "Empty message"}

                # Record user message
                user_message = {"role": "user", "content": message_text, "timestamp": time.time()}
                self.chat_messages.append(user_message)
                await self._broadcast({"type": "chat_message", "message": user_message})

                # Process via agent (runs in same request to keep simple)
                assistant_content = await self._process_agent_message(message_text)
                assistant_message = {"role": "assistant", "content": assistant_content, "timestamp": time.time()}
                self.chat_messages.append(assistant_message)
                await self._broadcast({"type": "chat_message", "message": assistant_message})
                return {"status": "success"}
            except Exception as e:
                logger.error(f"Failed to process chat message: {e}", exc_info=True)
                error_message = {"role": "assistant", "content": f"Error: {e}", "timestamp": time.time()}
                self.chat_messages.append(error_message)
                await self._broadcast({"type": "chat_message", "message": error_message})
                return {"status": "error", "message": str(e)}

    async def _broadcast(self, message: dict):
        for connection in self.websocket_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to broadcast to a websocket client: {e}")

    async def _handle_websocket_message(self, data: dict, websocket: WebSocket):
        logger.info(f"Received websocket message: {data}")

    async def _ensure_browser_session(self):
        """Ensure browser session is healthy, recreate if necessary."""
        try:
            # Check if browser exists and is still connected
            if self.browser:
                # Try to get current URL to test if browser is responsive
                try:
                    cdp_session = await self.browser.get_or_create_cdp_session()
                    if cdp_session and cdp_session.cdp_client:
                        # Test if we can communicate with the browser
                        await cdp_session.cdp_client.send.Page.enable(session_id=cdp_session.session_id)
                        logger.debug("Browser session appears healthy")
                        return
                except Exception as e:
                    logger.warning(f"Browser session appears unhealthy: {e}, recreating...")
                    # Browser is unhealthy, clean up
                    try:
                        await self.browser.stop()
                    except Exception:
                        pass
                    self.browser = None
                    # Only reset agent if not actively recording manual steps
                    if not self.recording_active:
                        self.agent = None
                    self.script_injected = False
                    self.listeners_attached = False

            # Create new browser session
            if not self.browser:
                logger.info("Creating new browser session for agent tasks")
                temp_user_data_dir = tempfile.mkdtemp()
                self.browser = Browser(
                    cdp_url="ws://localhost:3000",
                    user_data_dir=temp_user_data_dir,
                    headless=False,
                    keep_alive=True  # Keep browser alive between tasks
                )
                await self.browser.start()
                
                # Set up CDP listeners if needed for manual recording
                if not self.listeners_attached:
                    asyncio.create_task(self.setup_cdp_listeners())
                
                # Only reset agent if not actively recording
                if not self.recording_active:
                    self.agent = None
                
        except Exception as e:
            logger.error(f"Failed to ensure browser session: {e}", exc_info=True)
            raise RuntimeError(f"Could not establish browser session: {e}")

    async def _process_agent_message(self, task_text: str) -> str:
        """Create or update the persistent Agent with a new task and run it.

        Returns a textual summary of the agent's outcome to display in chat.
        """
        # Ensure browser exists and is healthy
        await self._ensure_browser_session()

        async with self._agent_lock:  # Prevent concurrent runs
            # Automatically enable recording to capture agent actions
            was_recording = self.recording_active
            if not was_recording:
                self.recording_active = True
                await self._broadcast({"type": "recording_started", "reason": "agent_task"})
                logger.info("Auto-enabled recording to capture agent actions")
            
            try:
                if not self.agent:
                    # Load environment variables (idempotent)
                    load_dotenv()
                    api_key = os.getenv("GEMINI_API_KEY")
                    if not api_key:
                        raise RuntimeError("GEMINI_API_KEY not set in environment (.env)")
                    # Instantiate LLM + Agent with fresh browser session
                    llm = ChatGoogle(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), api_key=api_key)
                    self.agent = Agent(task=task_text, browser_session=self.browser, llm=llm)
                    logger.info(f"Created new agent with task: {task_text}")
                else:
                    # Add as a new task to existing agent, but verify browser is still connected
                    try:
                        # Test agent's browser connection before adding task
                        if hasattr(self.agent, 'browser') and self.agent.browser:
                            # Try a simple browser operation to test connectivity
                            await self.agent.browser.get_or_create_cdp_session()
                        self.agent.add_new_task(task_text)
                        logger.info(f"Added new task to existing agent: {task_text}")
                    except Exception as browser_error:
                        logger.warning(f"Agent's browser connection failed: {browser_error}, creating new agent")
                        # Browser connection failed, create new agent
                        llm = ChatGoogle(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), api_key=os.getenv("GEMINI_API_KEY"))
                        self.agent = Agent(task=task_text, browser_session=self.browser, llm=llm)

                # Run agent with step hooks for real-time updates
                logger.info(f"Running agent with task: {task_text}")
                
                # Add a marker step to indicate agent task start
                self.step_counter += 1
                task_start_step = {
                    "id": self.step_counter,
                    "timestamp": time.time(),
                    "type": "agent_task",
                    "action": "agent_task_start",
                    "description": f"🤖 Agent Task: {task_text}",
                    "task_text": task_text,
                    "is_agent_action": True
                }
                self.recorded_steps.append(task_start_step)
                await self._broadcast({"type": "step_recorded", "step": task_start_step})
                
                result = await self.agent.run(
                    max_steps=int(os.getenv("AGENT_MAX_STEPS", "8")),
                    on_step_start=self._on_agent_step_start,
                    on_step_end=self._on_agent_step_end
                )
                
                # Add a marker step to indicate agent task completion
                self.step_counter += 1
                
                # Get comprehensive results from agent history
                history = result if hasattr(result, 'final_result') else None
                final_result = history.final_result() if history else self._summarize_agent_result(result)
                errors = history.errors() if history else []
                total_steps = len(history.model_thoughts()) if history else 0
                
                task_end_step = {
                    "id": self.step_counter,
                    "timestamp": time.time(),
                    "type": "agent_result",
                    "action": "agent_task_end",
                    "description": f"✅ Agent Task Completed: {task_text} (Steps: {total_steps})",
                    "task_text": task_text,
                    "result_summary": str(final_result),
                    "total_steps": total_steps,
                    "errors": errors,
                    "is_agent_action": True
                }
                self.recorded_steps.append(task_end_step)
                await self._broadcast({"type": "step_recorded", "step": task_end_step})

                # Send final comprehensive result to chat
                final_message_parts = []
                final_message_parts.append(f"🎯 **Task Completed:** {task_text}")
                final_message_parts.append(f"📊 **Steps Executed:** {total_steps}")
                
                if final_result:
                    final_message_parts.append(f"✅ **Result:** {final_result}")
                
                if errors:
                    final_message_parts.append(f"⚠️ **Errors:** {len(errors)} error(s) occurred")
                
                final_summary_message = {
                    "role": "assistant",
                    "content": "\n".join(final_message_parts),
                    "timestamp": time.time(),
                    "is_final_result": True,
                    "task_text": task_text,
                    "total_steps": total_steps
                }
                await self._broadcast({"type": "chat_message", "message": final_summary_message})

                # Return summary for HTTP response
                return str(final_result) if final_result else "Task completed successfully"
            except Exception as e:
                logger.error("Agent run failed", exc_info=True)
                # Try to recover by resetting agent and browser
                self.agent = None
                try:
                    if self.browser:
                        await self.browser.stop()
                except Exception:
                    pass
                self.browser = None
                return f"Agent error: {e}. Browser session will be recreated for next task."
            finally:
                # Restore previous recording state if we auto-enabled it
                if not was_recording and self.recording_active:
                    logger.info("Agent task completed, keeping recording enabled to see results")
                    # Keep recording enabled so user can see what the agent did
                    # They can manually stop it if desired

    def _summarize_agent_result(self, result) -> str:
        """Best-effort extraction of a result string from various possible result object shapes."""
        try:
            # Common patterns – adjust defensively so UI always gets something.
            if result is None:
                return "Agent finished (no result)."
            # Pydantic model with dict()
            if hasattr(result, "final_result") and result.final_result:
                return str(result.final_result)
            if hasattr(result, "result") and result.result:
                return str(result.result)
            if hasattr(result, "model_dump"):
                dumped = result.model_dump()
                for k in ("final_result", "result", "output", "summary"):
                    if k in dumped and dumped[k]:
                        return str(dumped[k])
                return json.dumps(dumped)[:800]
            # Mapping / dict-like
            if isinstance(result, dict):
                for k in ("final_result", "result", "output", "summary"):
                    if k in result and result[k]:
                        return str(result[k])
                return json.dumps(result)[:800]
            return str(result)
        except Exception:
            return "Agent finished (unable to parse result)."

    async def _on_agent_step_start(self, agent):
        """Hook called at the start of each agent step."""
        # Skip sending start messages - not worth the noise
        pass

    async def _on_agent_step_end(self, agent):
        """Hook called at the end of each agent step."""
        try:
            thoughts = agent.history.model_thoughts()
            actions = agent.history.model_actions()
            
            if thoughts:
                latest_thought = thoughts[-1]
                step_number = len(thoughts)
                
                # Format the step update message
                content_parts = []
                
                # Add step number and goal
                if hasattr(latest_thought, 'next_goal') and latest_thought.next_goal:
                    content_parts.append(f"📍 Step {step_number}: {latest_thought.next_goal}")
                
                # Add evaluation if available
                if hasattr(latest_thought, 'evaluation') and latest_thought.evaluation:
                    content_parts.append(f"✅ {latest_thought.evaluation}")
                
                # Send step update to chat
                if content_parts:
                    step_message = {
                        "role": "assistant", 
                        "content": "\n".join(content_parts),
                        "timestamp": time.time(),
                        "is_step_update": True,
                        "step_type": "end",
                        "step_number": step_number
                    }
                    await self._broadcast({"type": "chat_message", "message": step_message})
                    
        except Exception as e:
            logger.error(f"Error in step end hook: {e}", exc_info=True)

    async def setup_cdp_listeners(self):
        """Setup CDP event listeners for manual user interactions."""
        if self.listeners_attached:
            logger.debug("CDP listeners already attached, skipping setup.")
            return
        
        try:
            await asyncio.sleep(2)  # Wait for browser and CDP to be ready
            if not self.browser or not self.browser.cdp_client:
                logger.error("Browser or CDP client not available for setting up listeners.")
                return

            cdp_session = await self.browser.get_or_create_cdp_session()
            if not cdp_session:
                logger.error("Failed to get CDP session.")
                return

            # Enable required CDP domains
            await cdp_session.cdp_client.send.Page.enable(session_id=cdp_session.session_id)
            await cdp_session.cdp_client.send.Runtime.enable(session_id=cdp_session.session_id)
            await cdp_session.cdp_client.send.DOM.enable(session_id=cdp_session.session_id)
            
            # Load and inject the interaction recorder script (only once per session)
            if not self.script_injected:
                script_path = Path(__file__).parent / "ui" / "static" / "interaction-recorder.js"
                try:
                    with open(script_path, 'r', encoding='utf-8') as f:
                        interaction_script = f.read()
                    await self.browser._cdp_add_init_script(interaction_script)
                    self.script_injected = True
                    logger.info("Interaction recorder script injected successfully.")
                except FileNotFoundError:
                    logger.error(f"Interaction recorder script not found at {script_path}")
                    return
            else:
                logger.debug("Interaction recorder script already injected, skipping.")

            # Listen for console messages from our script
            cdp_session.cdp_client.register.Runtime.consoleAPICalled(self.on_console_api_called)
            
            # Listen for navigation events
            cdp_session.cdp_client.register.Page.frameNavigated(self.on_frame_navigated)

            self.listeners_attached = True
            logger.info("CDP listeners for manual interaction have been set up.")

        except Exception as e:
            logger.error(f"Failed to setup CDP listeners: {e}", exc_info=True)

    def on_console_api_called(self, *cb_args, **cb_kwargs):
        """Handler for Runtime.consoleAPICalled CDP event.

        Some CDP registry implementations invoke handlers with (event_dict, session_id)
        while others may pass (method_name, event_dict) or additional context.
        We defensively accept *args and locate the event payload.
        """
        if not self.recording_active:
            return

        # Find the event dict in positional args
        event = None
        for a in cb_args:
            if isinstance(a, dict) and 'type' in a and 'args' in a:
                event = a
                break
        if event is None:
            # Check kwargs just in case
            for v in cb_kwargs.values():
                if isinstance(v, dict) and 'type' in v and 'args' in v:
                    event = v
                    break
        if event is None:
            logger.debug("consoleAPICalled handler received unexpected args: %s %s", cb_args, cb_kwargs)
            return

        try:
            log_type = event.get('type')
            args = event.get('args', [])
            if (
                log_type == 'log'
                and isinstance(args, list)
                and len(args) > 1
                and isinstance(args[0], dict)
                and args[0].get('value') == '__browser_use_interaction__'
            ):
                payload_str = args[1].get('value') if isinstance(args[1], dict) else None
                if payload_str:
                    interaction = json.loads(payload_str)
                    asyncio.create_task(self.record_cdp_interaction(interaction))
        except Exception as e:
            logger.error(f"Error processing console API call: {e}", exc_info=True)

    def on_frame_navigated(self, *cb_args, **cb_kwargs):
        """Handler for Page.frameNavigated CDP event (defensive signature)."""
        if not self.recording_active:
            return

        event = None
        for a in cb_args:
            if isinstance(a, dict) and 'frame' in a:
                event = a
                break
        if event is None:
            for v in cb_kwargs.values():
                if isinstance(v, dict) and 'frame' in v:
                    event = v
                    break
        if event is None:
            logger.debug("frameNavigated handler received unexpected args: %s %s", cb_args, cb_kwargs)
            return

        frame = event.get('frame')
        if frame and not frame.get('parentId'):
            url = frame.get('url')
            if url and url and url != 'about:blank':
                interaction = {'type': 'navigate', 'url': url}
                asyncio.create_task(self.record_cdp_interaction(interaction))

    async def _handle_typing_event(self, interaction: dict, current_time: float):
        """Handle typing events with debouncing to avoid spam."""
        tag = interaction.get('tag', '')
        value = interaction.get('value', '')
        event_type = interaction.get('event', 'input')
        xpath = interaction.get('xpath', '')
        
        # Create a unique key for this input element
        element_key = f"{tag}_{hash(xpath)}" if xpath else f"{tag}_{hash(str(interaction.get('position', {})))}"
        
        if event_type == 'input':
            # For input events, buffer the typing
            if element_key not in self.typing_buffer:
                self.typing_buffer[element_key] = {
                    'step_id': None,
                    'start_time': current_time,
                    'last_update': current_time,
                    'initial_value': value,
                    'current_value': value,
                    'tag': tag,
                    'xpath': xpath,
                    'interaction': interaction  # Store full interaction for context
                }
            else:
                # Update existing buffer
                self.typing_buffer[element_key]['last_update'] = current_time
                self.typing_buffer[element_key]['current_value'] = value
                self.typing_buffer[element_key]['interaction'] = interaction
            
            # Schedule a delayed commit (will be cancelled if more typing happens)
            asyncio.create_task(self._commit_typing_after_delay(element_key, 1.0))  # 1 second delay
            
        elif event_type == 'change':
            # Change event means typing is done, commit immediately
            if element_key in self.typing_buffer:
                await self._commit_typing_step(element_key, force=True)
            else:
                # Direct change without prior input events
                await self._create_typing_step(interaction, current_time)

    async def _commit_typing_after_delay(self, element_key: str, delay: float):
        """Commit a typing step after a delay, if no more typing happens."""
        await asyncio.sleep(delay)
        
        if element_key in self.typing_buffer:
            buffer_entry = self.typing_buffer[element_key]
            # Only commit if no recent updates (user stopped typing)
            if time.time() - buffer_entry['last_update'] >= delay * 0.9:  # 90% of delay
                await self._commit_typing_step(element_key)

    async def _commit_typing_step(self, element_key: str, force: bool = False):
        """Commit buffered typing to a step."""
        if element_key not in self.typing_buffer:
            return
            
        buffer_entry = self.typing_buffer[element_key]
        current_time = time.time()
        
        # Only commit if enough time has passed or forced
        if not force and current_time - buffer_entry['last_update'] < 0.9:
            return
            
        interaction = buffer_entry['interaction']
        value = buffer_entry['current_value']
        
        if buffer_entry['step_id']:
            # Update existing step
            await self._update_typing_step(buffer_entry['step_id'], value)
        else:
            # Create new step
            await self._create_typing_step(interaction, current_time)
        
        # Clean up buffer
        del self.typing_buffer[element_key]

    async def _create_typing_step(self, interaction: dict, timestamp: float):
        """Create a new typing step with enhanced context."""
        self.step_counter += 1
        
        # Extract enhanced element information
        selectors = interaction.get('selectors', [])
        primary_selector = selectors[0] if selectors else None
        
        step = {
            "id": self.step_counter,
            "timestamp": timestamp,
            "type": "type",
            "action": "type",
            "tag": interaction.get('tag'),
            "input_type": interaction.get('inputType'),  # Updated property name
            "value": interaction.get('value', ''),
            "xpath": interaction.get('xpath'),
            "selectors": selectors,
            "primary_selector": primary_selector,
            "element_id": interaction.get('id'),
            "element_name": interaction.get('name'),
            "element_class": interaction.get('className'),
            "placeholder": interaction.get('placeholder'),
            "aria_label": interaction.get('ariaLabel'),
            "description": f"Type '{interaction.get('value', '')}' into {interaction.get('tag', 'element')}"
        }
        
        # Add element identification context for Cypress generation
        if primary_selector:
            step["cypress_selector"] = primary_selector
        elif interaction.get('xpath'):
            step["cypress_selector"] = f"xpath='{interaction.get('xpath')}'"
        
        self.recorded_steps.append(step)
        await self._broadcast({"type": "step_recorded", "step": step})
        logger.info(f"Recorded CDP interaction: {step.get('description', 'Unknown typing action')}")

    async def _update_typing_step(self, step_id: int, new_value: str):
        """Update an existing typing step with new value."""
        for i, step in enumerate(self.recorded_steps):
            if step.get('id') == step_id:
                step['value'] = new_value
                step['description'] = f"Type '{new_value}' into {step.get('tag', 'element')}"
                await self._broadcast({"type": "step_updated", "index": i, "step": step})
                logger.info(f"Updated CDP interaction: {step.get('description', 'Unknown update')}")
                break

    async def record_cdp_interaction(self, interaction: dict):
        """Records an interaction captured from CDP events with enhanced context."""
        try:
            current_time = time.time()
            step_type = interaction.get("type")
            
            # Handle typing events with special debouncing logic
            if step_type == "type":
                await self._handle_typing_event(interaction, current_time)
                return
            
            # Create a signature for deduplication (non-typing events)
            xpath = interaction.get('xpath', '')
            position = interaction.get('position', {})
            
            if step_type == "click":
                signature = f"{step_type}:{interaction.get('tag')}:{xpath}:{position.get('x', 0)}:{position.get('y', 0)}"
            elif step_type == "key":
                signature = f"{step_type}:{interaction.get('tag')}:{xpath}:{interaction.get('key')}"
            elif step_type == "scroll":
                signature = f"{step_type}:{xpath}:{interaction.get('scrollTop', 0)}"
            elif step_type == "navigate":
                signature = f"{step_type}:{interaction.get('url')}"
            else:
                signature = f"{step_type}:{xpath}"
            
            # Different dedup timeouts for different event types
            dedup_timeout = 0.5  # Default 500ms
            if step_type == "key":
                dedup_timeout = 0.15  # Shorter for key events (150ms to match JS)
            elif step_type == "scroll":
                dedup_timeout = 0.3  # 300ms for scroll events
            
            # Deduplicate: ignore if same event within timeout
            if (signature == self.last_event_signature and 
                current_time - self.last_event_time < dedup_timeout):
                logger.debug(f"Ignoring duplicate event: {signature}")
                return
            
            self.last_event_signature = signature
            self.last_event_time = current_time
            self.step_counter += 1
            
            # Extract enhanced element information
            selectors = interaction.get('selectors', [])
            primary_selector = selectors[0] if selectors else None
            
            step = {
                "id": self.step_counter,
                "timestamp": current_time,
                "type": step_type,
                "action": step_type,
                "tag": interaction.get('tag'),
                "input_type": interaction.get('inputType'),  # Updated property name
                "xpath": xpath,
                "selectors": selectors,
                "primary_selector": primary_selector,
                "element_id": interaction.get('id'),
                "element_name": interaction.get('name'),
                "element_class": interaction.get('className'),
                "aria_label": interaction.get('ariaLabel'),
                "position": position
            }

            if step_type == "click":
                click_type = interaction.get('clickType', 'left')
                modifiers = interaction.get('modifiers', {})
                
                step.update({
                    "text": interaction.get("text"),
                    "click_type": click_type,
                    "modifiers": modifiers,
                    "description": f"Click {interaction.get('tag', 'element')}" + 
                                 (f" (right-click)" if click_type == 'right' else "")
                })
                
                # Add Cypress selector
                if primary_selector:
                    step["cypress_selector"] = primary_selector
                elif xpath:
                    step["cypress_selector"] = f"xpath='{xpath}'"
                    
            elif step_type == "key":
                key = interaction.get("key", "")
                modifiers = interaction.get('modifiers', {})
                step.update({
                    "key": key,
                    "modifiers": modifiers,
                    "description": f"Press {key} in {interaction.get('tag', 'element')}"
                })
                
            elif step_type == "scroll":
                step.update({
                    "scroll_top": interaction.get("scrollTop", 0),
                    "scroll_left": interaction.get("scrollLeft", 0),
                    "description": f"Scroll in {interaction.get('tag', 'element')}"
                })
                
            elif step_type == "navigate":
                step["url"] = interaction.get("url")
                step["description"] = f"Navigate to {interaction.get('url')}"
                
            elif step_type == "submit":
                step["description"] = f"Submit form {interaction.get('tag', 'FORM')}"
                
            elif step_type == "contextmenu":
                step["description"] = f"Right-click {interaction.get('tag', 'element')}"
            
            else:
                # Fallback description for any unhandled event types
                step["description"] = f"{step_type.title()} {interaction.get('tag', 'element')}"
            
            self.recorded_steps.append(step)
            await self._broadcast({"type": "step_recorded", "step": step})
            logger.info(f"Recorded CDP interaction: {step.get('description', 'Unknown action')}")

        except Exception as e:
            logger.error(f"Failed to record CDP interaction: {e}", exc_info=True)

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)

def start_ui(host: str = "localhost", port: int = 8080, steel_ws_url: str = "ws://localhost:3000/"):
    """Starts the UI server."""
    server = UIServer(host=host, port=port, steel_ws_url=steel_ws_url)
    server.run()
