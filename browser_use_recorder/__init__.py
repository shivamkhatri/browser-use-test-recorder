"""Browser-Use Recorder

Web-based interaction recorder for browser automation built on browser-use.
"""

from .server import UIServer, start_ui

__version__ = "0.1.0"
__all__ = ["UIServer", "start_ui"]
