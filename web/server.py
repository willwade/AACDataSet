#!/usr/bin/env python3
"""
Simple HTTP server for the AAC Conversation Viewer.
This server serves static files and provides access to the batch_files directory.
"""

import http.server
import socketserver
import os
import argparse
from pathlib import Path


class AACViewerHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler that allows access to batch_files directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent.parent), **kwargs)

    def log_message(self, format, *args):
        """Override to provide more informative logging."""
        print(f"{self.address_string()} - {format % args}")


def main():
    parser = argparse.ArgumentParser(
        description="Start a simple HTTP server for the AAC Conversation Viewer"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)",
    )
    args = parser.parse_args()

    # Get the port
    port = args.port

    # Create the server
    handler = AACViewerHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)

    print(f"Starting server at http://localhost:{port}/web/")
    print("Press Ctrl+C to stop the server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
