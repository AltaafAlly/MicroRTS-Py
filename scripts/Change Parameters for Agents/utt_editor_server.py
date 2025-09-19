#!/usr/bin/env python3
"""
UTT Editor Server
Serves the UTT editor as a local web server so you can access it in any browser.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def start_server(port=8080):
    """Start a local web server for the UTT editor."""
    # Change to the scripts directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if the HTML file exists
    html_file = script_dir / "utt_editor.html"
    if not html_file.exists():
        print("❌ UTT editor HTML file not found!")
        return False
    
    # Create a custom handler that serves the HTML file
    class UTTEditorHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/' or self.path == '/utt_editor.html':
                self.path = '/utt_editor.html'
            return super().do_GET()
    
    try:
        # Start the server
        with socketserver.TCPServer(("", port), UTTEditorHandler) as httpd:
            print("🎮 MicroRTS UTT Editor Server")
            print("=" * 40)
            print(f"🌐 Server starting on port {port}...")
            print(f"📁 Serving from: {script_dir}")
            print(f"🔗 URL: http://localhost:{port}")
            print()
            print("💡 Tips:")
            print("   - Open the URL above in your web browser")
            print("   - Use 'Load UTT File' to load your UTT files")
            print("   - Press Ctrl+C to stop the server")
            print()
            
            # Try to open in browser
            try:
                webbrowser.open(f"http://localhost:{port}")
                print("🚀 Opening in your default browser...")
            except:
                print("⚠️  Could not open browser automatically")
                print(f"   Please manually open: http://localhost:{port}")
            
            print("\n" + "=" * 40)
            print("Server running... Press Ctrl+C to stop")
            print("=" * 40)
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        return True
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ Port {port} is already in use!")
            print(f"   Try a different port: python utt_editor_server.py {port + 1}")
            return False
        else:
            print(f"❌ Error starting server: {e}")
            return False

def main():
    """Main function."""
    port = 8080
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid port number!")
            print("Usage: python utt_editor_server.py [port]")
            sys.exit(1)
    
    success = start_server(port)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
