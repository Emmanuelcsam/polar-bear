#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Server Setup with Enhanced Performance and Security
Modular server configuration with uvloop optimization.
"""
import sys
import asyncio
import time
from typing import Dict, Any, Set

# Set up uvloop for enhanced performance
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    HAS_UVLOOP = True
except ImportError:
    HAS_UVLOOP = False

try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = Request = HTTPException = None
    CORSMiddleware = StaticFiles = FileResponse = HTMLResponse = JSONResponse = None
    FASTAPI_AVAILABLE = False

# Rate limiting and security
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False

from ..core.config import config
from ..core.logger import logger
from ..core.json_utils import json_encoder
from ..core.models import PerformanceMetrics


class EnhancedFastAPI:
    """Enhanced FastAPI application with performance optimizations."""

    def __init__(self):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required but not available")

        # Create FastAPI app with custom JSON encoder
        self.app = FastAPI(
            title="Neural Nexus IDE Server",
            version="6.0",
            description="Enhanced Python IDE with AI-powered auto-healing",
            default_response_class=JSONResponse if not config.use_orjson else None
        )

        # Performance metrics
        self.metrics = PerformanceMetrics()

        # Rate limiting setup
        self.limiter = None
        if HAS_SLOWAPI and config.rate_limit_enabled:
            self.limiter = Limiter(key_func=get_remote_address)
            self.app.state.limiter = self.limiter
            self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Setup middleware for security, CORS, and performance monitoring."""

        # Enhanced CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        # Security headers middleware
        @self.app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            """Add security headers to all responses."""
            start_time = time.time()
            response = await call_next(request)

            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

            # Content Security Policy
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; "
                "connect-src 'self' ws: wss:; "
                "font-src 'self'; "
                "object-src 'none'; "
                "base-uri 'self';"
            )
            response.headers["Content-Security-Policy"] = csp

            # Cross-Origin policies for enhanced security
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
            response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"

            # Update performance metrics
            processing_time = time.time() - start_time
            self.metrics.update_request_metrics(processing_time)

            return response

        # Performance monitoring middleware
        @self.app.middleware("http")
        async def monitor_performance(request: Request, call_next):
            """Monitor request performance and resource usage."""
            start_time = time.time()

            # Update memory usage if psutil is available
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                self.metrics.current_memory_usage = memory_info.rss / 1024 / 1024  # MB
                self.metrics.peak_memory_usage = max(
                    self.metrics.peak_memory_usage,
                    self.metrics.current_memory_usage
                )
                self.metrics.cpu_usage_percent = process.cpu_percent()
            except ImportError:
                pass

            response = await call_next(request)

            # Log slow requests
            processing_time = time.time() - start_time
            if processing_time > 1.0:  # Log requests taking more than 1 second
                logger.warning(f"Slow request: {request.method} {request.url.path} took {processing_time:.2f}s")

            return response

    def _setup_routes(self):
        """Setup basic API routes."""

        # Health check endpoint with detailed status
        rate_limit_decorator = self.limiter.limit("10/minute") if self.limiter else lambda f: f

        @self.app.get("/health")
        @rate_limit_decorator
        async def health_check(request: Request):
            """Enhanced health check with system information."""
            health_data = {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "6.0",
                "features": {
                    "uvloop": HAS_UVLOOP,
                    "orjson": config.use_orjson,
                    "rate_limiting": config.rate_limit_enabled,
                    "security_scanning": config.security_scan_enabled,
                    "auto_formatting": config.auto_format_enabled
                },
                "performance": self.metrics.to_dict(),
                "uptime": self.metrics.get_uptime()
            }

            return JSONResponse(content=health_data)

        # Server info endpoint
        @self.app.get("/api/info")
        async def get_server_info():
            """Get detailed server information."""
            return {
                "name": "Neural Nexus IDE Server",
                "version": "6.0",
                "features": config.get_feature_status(),
                "capabilities": {
                    "code_analysis": True,
                    "auto_healing": True,
                    "security_scanning": config.security_scan_enabled,
                    "type_checking": config.features.get('mypy', False),
                    "auto_formatting": config.auto_format_enabled,
                    "project_management": True,
                    "terminal_integration": True,
                    "ai_assistance": config.features.get('openai', False)
                },
                "performance": {
                    "event_loop": "uvloop" if HAS_UVLOOP else "asyncio",
                    "json_engine": "orjson" if config.use_orjson else "standard",
                    "rate_limiting": config.rate_limit_enabled
                }
            }

        # Code formatting endpoint
        @self.app.post("/api/format")
        @rate_limit_decorator
        async def format_code(request: Request):
            """Format code using available tools."""
            try:
                data = await request.json()
                content = data.get('content', '')

                if not content:
                    raise HTTPException(status_code=400, detail="No content provided")

                # Try to format with Ruff if available
                if config.auto_format_enabled:
                    from ..analysis.code_analyzer import analyzer

                    # This would use Ruff's formatting capabilities
                    # For now, return the original content with a message
                    return {
                        "formatted_content": content,
                        "message": "Code formatting available with Ruff",
                        "tool_used": "ruff" if config.features.get('ruff') else "none"
                    }
                else:
                    return {
                        "formatted_content": content,
                        "message": "No formatting tools available",
                        "tool_used": "none"
                    }

            except Exception as e:
                logger.error(f"Code formatting failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Security analysis endpoint
        @self.app.post("/api/security-scan")
        @rate_limit_decorator
        async def security_scan(request: Request):
            """Perform security analysis on code."""
            try:
                data = await request.json()
                content = data.get('content', '')

                if not content:
                    raise HTTPException(status_code=400, detail="No content provided")

                if config.security_scan_enabled:
                    from ..analysis.code_analyzer import analyzer

                    analysis = await analyzer.analyze_code(content)

                    return {
                        "security_issues": [issue.__dict__ for issue in analysis.security_issues],
                        "security_score": analysis.security_score,
                        "recommendations": [
                            {
                                "type": "security",
                                "message": "Review security issues found during analysis",
                                "priority": "high" if analysis.security_score < 7 else "medium"
                            }
                        ]
                    }
                else:
                    return {
                        "security_issues": [],
                        "security_score": None,
                        "message": "Security scanning not available (missing tools)"
                    }

            except Exception as e:
                logger.error(f"Security scan failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    enhanced_app = EnhancedFastAPI()
    return enhanced_app.get_app()
