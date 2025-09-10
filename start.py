#!/usr/bin/env python3
"""Simple launcher for Browser-Use Recorder."""

if __name__ == "__main__":
    from browser_use_recorder.server import start_ui
    
    print(" Starting Browser-Use Recorder...")
    print(" Connecting to Steel Browser at ws://localhost:3000/")
    print(" UI will be available at http://localhost:8080")
    print()
    print("Make sure Steel Browser is running:")
    print("  docker-compose up -d")
    print()
    print("Then open: http://localhost:8080")
    
    start_ui(
        host="localhost",
        port=8080,
        steel_ws_url="ws://localhost:3000/"
    )
