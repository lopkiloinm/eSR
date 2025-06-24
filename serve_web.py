#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple HTTP server for edge-SR web interface
Serves static files with proper CORS headers for ONNX model loading
"""
import http.server
import socketserver
import os
import mimetypes
import argparse
from pathlib import Path


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support for ONNX model loading"""
    
    def end_headers(self):
        """Add CORS headers to all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()

    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        """Custom GET handler to serve ONNX models from parent directory"""
        if self.path.startswith('/onnx_models/'):
            # Serve ONNX models from parent directory or current directory
            onnx_path = self.path[1:]  # Remove leading '/' -> 'onnx_models/filename.onnx'
            
            # Try multiple possible locations
            possible_paths = [
                Path('..') / onnx_path,  # ../onnx_models/filename.onnx (if in web dir)
                Path(onnx_path),         # onnx_models/filename.onnx (if in root dir)
                Path('..') / '..' / onnx_path  # ../../onnx_models/filename.onnx (just in case)
            ]
            
            full_path = None
            for path in possible_paths:
                if path.exists() and path.suffix == '.onnx':
                    full_path = path.resolve()
                    break
            
            if full_path:
                print(f"DEBUG: Serving ONNX model from: {full_path}")
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(full_path.stat().st_size))
                self.end_headers()
                
                with open(full_path, 'rb') as f:
                    self.wfile.write(f.read())
                return
            else:
                print(f"DEBUG: ONNX model not found, tried paths:")
                for path in possible_paths:
                    print(f"  - {path.resolve()} (exists: {path.exists()})")
                self.send_error(404, f"ONNX model not found: {onnx_path}")
                return
        
        # Default handling for other files
        super().do_GET()

    def guess_type(self, path):
        """Enhanced MIME type guessing for ONNX files"""
        base, ext = os.path.splitext(path)
        if ext == '.onnx':
            return 'application/octet-stream'
        return super().guess_type(path)

    def log_message(self, format, *args):
        """Custom logging format"""
        print(f"[{self.log_date_time_string()}] {format % args}")


def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


def serve_web_interface(port=None, directory=None):
    """Start the web server"""
    
    # Set working directory
    if directory:
        os.chdir(directory)
    else:
        # Default to web directory if it exists
        web_dir = Path('web')
        if web_dir.exists():
            os.chdir('web')
        
    # Find available port
    if port is None:
        try:
            port = find_available_port()
        except RuntimeError as e:
            print(f"Error: {e}")
            return False
    
    # Check for required files
    required_files = ['index.html', 'style.css', 'script.js']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"Warning: Missing files: {', '.join(missing_files)}")
        print("Make sure you're running this from the correct directory.")
    
    # Check for ONNX models
    onnx_dir = Path('../onnx_models')
    if not onnx_dir.exists():
        print("Warning: ONNX models directory not found.")
        print("Run 'python convert_to_onnx.py' first to convert PyTorch models.")
    else:
        onnx_files = list(onnx_dir.glob('*.onnx'))
        print(f"Found {len(onnx_files)} ONNX model(s)")
    
    try:
        # Start server
        with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
            print(f"\n{'='*50}")
            print(f"🚀 edge-SR Web Server Started")
            print(f"{'='*50}")
            print(f"📍 URL: http://localhost:{port}")
            print(f"📁 Serving: {os.getcwd()}")
            print(f"🔧 Press Ctrl+C to stop the server")
            print(f"{'='*50}\n")
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n{'='*50}")
        print("🛑 Server stopped by user")
        print(f"{'='*50}")
        return True
    except OSError as e:
        print(f"Error starting server: {e}")
        if "Address already in use" in str(e):
            print(f"Port {port} is already in use. Try a different port with --port")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Serve the edge-SR web interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python serve_web.py                    # Start server on auto-detected port
  python serve_web.py --port 8080        # Start server on specific port
  python serve_web.py --dir ./web        # Serve from specific directory
  
Note: Make sure to convert PyTorch models to ONNX format first:
  python convert_to_onnx.py --limit 5
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        help='Port to serve on (default: auto-detect starting from 8000)'
    )
    
    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='Directory to serve from (default: ./web if exists, else current)'
    )
    
    args = parser.parse_args()
    
    try:
        success = serve_web_interface(port=args.port, directory=args.dir)
        exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)


if __name__ == "__main__":
    main() 