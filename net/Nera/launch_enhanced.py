#!/usr/bin/env python3
"""
Enhanced Neural Nexus IDE Server Launcher v6.0
High-performance launch script with security and static analysis
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

async def main():
    """Main entry point with enhanced startup"""
    try:
        # Import the enhanced server
        from neural_nexus_server import NeuralNexusServer, logger

        # Print enhanced startup banner
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 18 + "Neural Nexus IDE Server v6.0" + " " * 11 + "║")
        print("║" + " " * 8 + "🚀 Enhanced with Security & Performance 🚀" + " " * 8 + "║")
        print("╠" + "═" * 58 + "╣")
        print("║  Web Interface: http://localhost:8765                   ║")
        print("║  WebSocket: ws://localhost:8765/ws                      ║")
        print("║  Health Check: http://localhost:8765/health             ║")
        print("║  API Docs: http://localhost:8765/docs                   ║")
        print("║                                                        ║")
        print("║  🆕 New Features v6.0:                                  ║")
        print("║  ✅ Ultra-fast JSON with orjson (6x faster)            ║")
        print("║  ✅ Enhanced event loop with uvloop (4x faster)        ║")
        print("║  ✅ Security scanning with Semgrep & Bandit            ║")
        print("║  ✅ Auto-formatting with Ruff                          ║")
        print("║  ✅ Type checking with Pyright                         ║")
        print("║  ✅ Rate limiting & security headers                   ║")
        print("║  ✅ Structured logging with loguru                     ║")
        print("║  ✅ Performance monitoring                             ║")
        print("║                                                        ║")
        print("║  🔒 Security Features:                                  ║")
        print("║  • CSP & COOP/COEP headers                             ║")
        print("║  • Rate limiting on API endpoints                     ║")
        print("║  • Comprehensive vulnerability scanning               ║")
        print("║  • Safe code execution environment                    ║")
        print("║                                                        ║")
        print("║  Open http://localhost:8765 in your browser!          ║")
        print("╚" + "═" * 58 + "╝")

        # Create server instance
        server = NeuralNexusServer(port=8765)

        # Check available features
        logger.info("🔧 Checking available features...")
        from neural_nexus_server import (
            HAS_UVLOOP, HAS_ORJSON, HAS_LOGURU, HAS_SLOWAPI,
            HAS_SEMGREP, HAS_BANDIT, HAS_RUFF, HAS_MSGSPEC
        )

        features = []
        if HAS_UVLOOP:
            features.append("uvloop (faster event loop)")
        if HAS_ORJSON:
            features.append("orjson (faster JSON)")
        if HAS_LOGURU:
            features.append("loguru (structured logging)")
        if HAS_SLOWAPI:
            features.append("slowapi (rate limiting)")
        if HAS_SEMGREP:
            features.append("semgrep (security scanning)")
        if HAS_BANDIT:
            features.append("bandit (vulnerability detection)")
        if HAS_RUFF:
            features.append("ruff (fast linting & formatting)")
        if HAS_MSGSPEC:
            features.append("msgspec (fast serialization)")

        if features:
            logger.info(f"✅ Enhanced features available: {', '.join(features)}")
        else:
            logger.warning("⚠️  Running with basic features only. Install enhanced packages for better performance.")

        # Start server with enhanced configuration
        import uvicorn

        config = uvicorn.Config(
            server.app,
            host="127.0.0.1",
            port=8765,
            log_level="info",
            access_log=True,
            use_colors=True,
            # Enhanced server configuration
            loop="uvloop" if HAS_UVLOOP else "asyncio",
            http="httptools",
            ws="websockets",
            reload=False,
            workers=1  # Single worker for WebSocket state consistency
        )

        server_instance = uvicorn.Server(config)

        # Graceful shutdown handling
        import signal

        def signal_handler(signum, frame):
            logger.info("🛑 Shutting down gracefully...")
            server_instance.should_exit = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the server
        logger.info("🚀 Starting Neural Nexus IDE Server v6.0...")
        await server_instance.serve()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing dependencies: pip install -r requirements_enhanced.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
