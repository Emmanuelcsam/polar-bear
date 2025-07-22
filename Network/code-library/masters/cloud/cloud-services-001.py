# api_gateway.py
"""
API Gateway and Microservices Architecture for Fiber Inspection System
Features:
- GraphQL and REST API gateway
- Service mesh integration
- Rate limiting and authentication
- Request routing and load balancing
- Circuit breakers and retries
- API versioning
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
import strawberry
from strawberry.fastapi import GraphQLRouter
from typing import Optional, List, Dict, Any, AsyncGenerator
import httpx
import asyncio
import json
import jwt
import redis
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, generate_latest
import circuitbreaker
from tenacity import retry, stop_after_attempt, wait_exponential
import grpc
from google.protobuf import json_format
import aiofiles
import hashlib
from cachetools import TTLCache
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import consul
from jaeger_client import Config as JaegerConfig
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('api_gateway_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_gateway_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Counter('api_gateway_active_connections', 'Active WebSocket connections')

# Initialize FastAPI app
app = FastAPI(
    title="Fiber Inspection API Gateway",
    description="Unified API for fiber optic inspection system",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Service registry
consul_client = consul.Consul(host='consul', port=8500)

# Cache
cache = TTLCache(maxsize=1000, ttl=300)
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# Circuit breaker
circuit_breaker = circuitbreaker.CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=httpx.HTTPError
)

# Tracing
tracer_provider = TracerProvider()
jaeger_exporter = JaegerExporter(
    agent_host_name='jaeger',
    agent_port=6831,
)
span_processor = BatchSpanProcessor(jaeger_exporter)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)


@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    name: str
    host: str
    port: int
    version: str
    health_check: str
    protocol: str = 'http'


class ServiceRegistry:
    """Service discovery and registration"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceEndpoint]] = {}
        self._discover_services()
        
    def _discover_services(self):
        """Discover services from Consul"""
        try:
            # Get all services
            _, services = consul_client.health.service(index=None, passing=True)
            
            for service in services:
                name = service['Service']['Service']
                endpoint = ServiceEndpoint(
                    name=name,
                    host=service['Service']['Address'],
                    port=service['Service']['Port'],
                    version=service['Service']['Meta'].get('version', '1.0'),
                    health_check=f"/health"
                )
                
                if name not in self.services:
                    self.services[name] = []
                self.services[name].append(endpoint)
                
        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            # Fallback to default services
            self._load_default_services()
    
    def _load_default_services(self):
        """Load default service endpoints"""
        self.services = {
            'processor': [ServiceEndpoint(
                name='processor',
                host='processor-service',
                port=8765,
                version='1.0',
                health_check='/health'
            )],
            'feature-extractor': [ServiceEndpoint(
                name='feature-extractor',
                host='feature-service',
                port=8080,
                version='1.0',
                health_check='/health'
            )],
            'ml-engine': [ServiceEndpoint(
                name='ml-engine',
                host='ml-service',
                port=5000,
                version='1.0',
                health_check='/health'
            )]
        }
    
    def get_service(self, name: str, version: Optional[str] = None) -> Optional[ServiceEndpoint]:
        """Get service endpoint with load balancing"""
        if name not in self.services:
            return None
            
        endpoints = self.services[name]
        if version:
            endpoints = [e for e in endpoints if e.version == version]
            
        if not endpoints:
            return None
            
        # Simple round-robin load balancing
        # In production, use more sophisticated algorithms
        return endpoints[hash(datetime.now()) % len(endpoints)]


# Initialize service registry
service_registry = ServiceRegistry()


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: int = 100, per: int = 60):
        self.rate = rate
        self.per = per
        
    async def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        try:
            pipe = redis_client.pipeline()
            now = datetime.now()
            
            # Clear old entries
            clearBefore = now - timedelta(seconds=self.per)
            pipe.zremrangebyscore(key, 0, clearBefore.timestamp())
            
            # Count recent requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now.timestamp()): now.timestamp()})
            
            # Set expiry
            pipe.expire(key, self.per)
            
            results = pipe.execute()
            
            return results[1] < self.rate
            
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            return True  # Allow on error


rate_limiter = RateLimiter(rate=100, per=60)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Verify JWT token"""
    token = credentials.credentials
    
    try:
        # Decode token
        payload = jwt.decode(
            token,
            options={"verify_signature": False}  # In production, verify signature
        )
        
        # Check expiration
        if 'exp' in payload and payload['exp'] < datetime.now().timestamp():
            raise HTTPException(status_code=401, detail="Token expired")
            
        return payload
        
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# GraphQL Schema
@strawberry.type
class FiberImage:
    id: str
    url: str
    timestamp: datetime
    status: str
    quality_score: float
    anomaly_score: float
    defects: List[str]


@strawberry.type
class InspectionResult:
    id: str
    image_id: str
    is_anomalous: bool
    confidence: float
    severity: str
    defects: List[Dict[str, Any]]
    processing_time: float
    timestamp: datetime


@strawberry.type
class Stream:
    id: str
    name: str
    status: str
    fps: int
    resolution: str
    active: bool
    last_update: datetime


@strawberry.type
class Query:
    @strawberry.field
    async def inspection_result(self, id: str) -> Optional[InspectionResult]:
        """Get inspection result by ID"""
        with tracer.start_as_current_span("get_inspection_result"):
            # Check cache
            cache_key = f"result:{id}"
            cached = cache.get(cache_key)
            if cached:
                return cached
                
            # Fetch from service
            service = service_registry.get_service('processor')
            if not service:
                return None
                
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{service.protocol}://{service.host}:{service.port}/results/{id}")
                if response.status_code == 200:
                    data = response.json()
                    result = InspectionResult(**data)
                    cache[cache_key] = result
                    return result
                    
        return None
    
    @strawberry.field
    async def active_streams(self) -> List[Stream]:
        """Get all active streams"""
        service = service_registry.get_service('processor')
        if not service:
            return []
            
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{service.protocol}://{service.host}:{service.port}/streams")
            if response.status_code == 200:
                return [Stream(**s) for s in response.json()]
                
        return []
    
    @strawberry.field
    async def search_results(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        anomalous_only: bool = False,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[InspectionResult]:
        """Search inspection results"""
        params = {
            'limit': limit,
            'anomalous_only': anomalous_only
        }
        
        if start_date:
            params['start_date'] = start_date.isoformat()
        if end_date:
            params['end_date'] = end_date.isoformat()
        if severity:
            params['severity'] = severity
            
        service = service_registry.get_service('processor')
        if not service:
            return []
            
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{service.protocol}://{service.host}:{service.port}/results/search",
                params=params
            )
            if response.status_code == 200:
                return [InspectionResult(**r) for r in response.json()]
                
        return []


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def analyze_image(self, image_url: str) -> InspectionResult:
        """Analyze a single image"""
        with tracer.start_as_current_span("analyze_image"):
            # Extract features
            feature_service = service_registry.get_service('feature-extractor')
            if not feature_service:
                raise Exception("Feature extraction service unavailable")
                
            async with httpx.AsyncClient() as client:
                # Extract features
                feature_response = await client.post(
                    f"{feature_service.protocol}://{feature_service.host}:{feature_service.port}/extract_features",
                    json={'image_url': image_url}
                )
                
                if feature_response.status_code != 200:
                    raise Exception("Feature extraction failed")
                    
                features = feature_response.json()['features']
                
                # Get prediction
                ml_service = service_registry.get_service('ml-engine')
                if not ml_service:
                    raise Exception("ML service unavailable")
                    
                ml_response = await client.post(
                    f"{ml_service.protocol}://{ml_service.host}:{ml_service.port}/predict",
                    json={'features': features}
                )
                
                if ml_response.status_code != 200:
                    raise Exception("Prediction failed")
                    
                result = ml_response.json()
                return InspectionResult(**result)
    
    @strawberry.mutation
    async def start_stream(self, stream_id: str, source: str) -> Stream:
        """Start a new inspection stream"""
        service = service_registry.get_service('processor')
        if not service:
            raise Exception("Processor service unavailable")
            
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{service.protocol}://{service.host}:{service.port}/streams",
                json={'stream_id': stream_id, 'source': source}
            )
            
            if response.status_code == 200:
                return Stream(**response.json())
            else:
                raise Exception(f"Failed to start stream: {response.text}")
    
    @strawberry.mutation
    async def stop_stream(self, stream_id: str) -> bool:
        """Stop an inspection stream"""
        service = service_registry.get_service('processor')
        if not service:
            raise Exception("Processor service unavailable")
            
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{service.protocol}://{service.host}:{service.port}/streams/{stream_id}"
            )
            
            return response.status_code == 200


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def inspection_updates(self, stream_id: Optional[str] = None) -> AsyncGenerator[InspectionResult, None]:
        """Subscribe to real-time inspection updates"""
        # Connect to WebSocket
        service = service_registry.get_service('processor')
        if not service:
            return
            
        ws_url = f"ws://{service.host}:{service.port}/ws"
        if stream_id:
            ws_url += f"?stream_id={stream_id}"
            
        async with httpx.AsyncClient() as client:
            async with client.stream('GET', ws_url) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        yield InspectionResult(**data)


# Create GraphQL app
schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


# REST API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    services_health = {}
    
    for service_name, endpoints in service_registry.services.items():
        for endpoint in endpoints:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{endpoint.protocol}://{endpoint.host}:{endpoint.port}{endpoint.health_check}",
                        timeout=2.0
                    )
                    services_health[service_name] = response.status_code == 200
            except:
                services_health[service_name] = False
                
    all_healthy = all(services_health.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": services_health
    }


@app.post("/api/v2/analyze")
@circuit_breaker
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def analyze_image(
    request: Request,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    features: Optional[List[float]] = None,
    user: Dict = Depends(verify_token)
):
    """Analyze fiber optic image"""
    with tracer.start_as_current_span("analyze_image_rest") as span:
        # Rate limiting
        if not await rate_limiter.is_allowed(f"user:{user.get('sub', 'anonymous')}"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
        REQUEST_COUNT.labels(method='POST', endpoint='/api/v2/analyze', status='started').inc()
        
        try:
            # Input validation
            if not any([image_url, image_base64, features]):
                raise HTTPException(status_code=400, detail="No input provided")
                
            # Process based on input type
            if features:
                # Direct prediction
                ml_service = service_registry.get_service('ml-engine')
                if not ml_service:
                    raise HTTPException(status_code=503, detail="ML service unavailable")
                    
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{ml_service.protocol}://{ml_service.host}:{ml_service.port}/predict",
                        json={'features': features},
                        timeout=30.0
                    )
                    
                    if response.status_code != 200:
                        raise HTTPException(status_code=response.status_code, detail="Prediction failed")
                        
                    result = response.json()
                    
            else:
                # Feature extraction needed
                feature_service = service_registry.get_service('feature-extractor')
                if not feature_service:
                    raise HTTPException(status_code=503, detail="Feature service unavailable")
                    
                async with httpx.AsyncClient() as client:
                    # Extract features
                    feature_data = {}
                    if image_url:
                        feature_data['image_url'] = image_url
                    else:
                        feature_data['image_base64'] = image_base64
                        
                    feature_response = await client.post(
                        f"{feature_service.protocol}://{feature_service.host}:{feature_service.port}/extract_features",
                        json=feature_data,
                        timeout=30.0
                    )
                    
                    if feature_response.status_code != 200:
                        raise HTTPException(status_code=feature_response.status_code, detail="Feature extraction failed")
                        
                    features = feature_response.json()['features']
                    
                    # Get prediction
                    ml_service = service_registry.get_service('ml-engine')
                    if not ml_service:
                        raise HTTPException(status_code=503, detail="ML service unavailable")
                        
                    ml_response = await client.post(
                        f"{ml_service.protocol}://{ml_service.host}:{ml_service.port}/predict",
                        json={'features': features},
                        timeout=30.0
                    )
                    
                    if ml_response.status_code != 200:
                        raise HTTPException(status_code=ml_response.status_code, detail="Prediction failed")
                        
                    result = ml_response.json()
                    
            # Add metadata
            result['request_id'] = request.headers.get('X-Request-ID', str(hash(datetime.now())))
            result['user_id'] = user.get('sub')
            result['api_version'] = 'v2'
            
            REQUEST_COUNT.labels(method='POST', endpoint='/api/v2/analyze', status='success').inc()
            
            return result
            
        except Exception as e:
            REQUEST_COUNT.labels(method='POST', endpoint='/api/v2/analyze', status='error').inc()
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise


@app.get("/api/v2/streams")
async def get_streams(user: Dict = Depends(verify_token)):
    """Get all active streams"""
    service = service_registry.get_service('processor')
    if not service:
        raise HTTPException(status_code=503, detail="Processor service unavailable")
        
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{service.protocol}://{service.host}:{service.port}/streams"
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to get streams")


@app.post("/api/v2/streams/{stream_id}/start")
async def start_stream(
    stream_id: str,
    source: str,
    fps: int = 30,
    resolution: str = "1920x1080",
    user: Dict = Depends(verify_token)
):
    """Start a new inspection stream"""
    # Check permissions
    if 'streams:write' not in user.get('permissions', []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
        
    service = service_registry.get_service('processor')
    if not service:
        raise HTTPException(status_code=503, detail="Processor service unavailable")
        
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{service.protocol}://{service.host}:{service.port}/streams",
            json={
                'stream_id': stream_id,
                'source': source,
                'fps': fps,
                'resolution': resolution
            }
        )
        
        if response.status_code == 200:
            ACTIVE_CONNECTIONS.inc()
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to start stream")


@app.delete("/api/v2/streams/{stream_id}")
async def stop_stream(stream_id: str, user: Dict = Depends(verify_token)):
    """Stop an inspection stream"""
    # Check permissions
    if 'streams:write' not in user.get('permissions', []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
        
    service = service_registry.get_service('processor')
    if not service:
        raise HTTPException(status_code=503, detail="Processor service unavailable")
        
    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{service.protocol}://{service.host}:{service.port}/streams/{stream_id}"
        )
        
        if response.status_code == 200:
            ACTIVE_CONNECTIONS.dec()
            return {"message": "Stream stopped"}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to stop stream")


@app.get("/api/v2/results/{result_id}")
async def get_result(result_id: str, user: Dict = Depends(verify_token)):
    """Get inspection result by ID"""
    # Check cache
    cache_key = f"result:{result_id}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
        
    service = service_registry.get_service('processor')
    if not service:
        raise HTTPException(status_code=503, detail="Processor service unavailable")
        
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{service.protocol}://{service.host}:{service.port}/results/{result_id}"
        )
        
        if response.status_code == 200:
            result = response.json()
            # Cache for 5 minutes
            await redis_client.setex(cache_key, 300, json.dumps(result))
            return result
        elif response.status_code == 404:
            raise HTTPException(status_code=404, detail="Result not found")
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to get result")


@app.get("/api/v2/results")
async def search_results(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    anomalous_only: bool = False,
    severity: Optional[str] = None,
    stream_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    user: Dict = Depends(verify_token)
):
    """Search inspection results"""
    params = {
        'limit': min(limit, 1000),  # Cap at 1000
        'offset': offset,
        'anomalous_only': anomalous_only
    }
    
    if start_date:
        params['start_date'] = start_date.isoformat()
    if end_date:
        params['end_date'] = end_date.isoformat()
    if severity:
        params['severity'] = severity
    if stream_id:
        params['stream_id'] = stream_id
        
    service = service_registry.get_service('processor')
    if not service:
        raise HTTPException(status_code=503, detail="Processor service unavailable")
        
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{service.protocol}://{service.host}:{service.port}/results",
            params=params
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Search failed")


@app.post("/api/v2/batch")
async def batch_analyze(
    images: List[str],
    user: Dict = Depends(verify_token)
):
    """Batch analyze multiple images"""
    # Check rate limit for batch operations
    if not await rate_limiter.is_allowed(f"batch:{user.get('sub', 'anonymous')}"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded for batch operations")
        
    if len(images) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch")
        
    # Process images in parallel
    tasks = []
    for image_url in images:
        task = analyze_image(Request, image_url=image_url, user=user)
        tasks.append(task)
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Format results
    batch_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            batch_results.append({
                'image': images[i],
                'success': False,
                'error': str(result)
            })
        else:
            batch_results.append({
                'image': images[i],
                'success': True,
                'result': result
            })
            
    return {
        'total': len(images),
        'successful': sum(1 for r in batch_results if r['success']),
        'failed': sum(1 for r in batch_results if not r['success']),
        'results': batch_results
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    ACTIVE_CONNECTIONS.inc()
    
    try:
        # Get authentication token
        token = await websocket.receive_text()
        user = await verify_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
        
        # Subscribe to updates
        service = service_registry.get_service('processor')
        if not service:
            await websocket.send_json({"error": "Processor service unavailable"})
            return
            
        # Forward WebSocket connection to processor service
        async with httpx.AsyncClient() as client:
            ws_url = f"ws://{service.host}:{service.port}/ws"
            
            async with client.stream('GET', ws_url) as response:
                async for line in response.aiter_lines():
                    if line:
                        await websocket.send_text(line)
                        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ACTIVE_CONNECTIONS.dec()
        await websocket.close()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/api/v2/export/{format}")
async def export_data(
    format: str,
    start_date: datetime,
    end_date: datetime,
    user: Dict = Depends(verify_token)
):
    """Export inspection data in various formats"""
    if format not in ['csv', 'json', 'parquet']:
        raise HTTPException(status_code=400, detail="Invalid format")
        
    # Check permissions
    if 'data:export' not in user.get('permissions', []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
        
    # Generate export
    service = service_registry.get_service('processor')
    if not service:
        raise HTTPException(status_code=503, detail="Processor service unavailable")
        
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{service.protocol}://{service.host}:{service.port}/export",
            json={
                'format': format,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            timeout=300.0  # 5 minute timeout for large exports
        )
        
        if response.status_code == 200:
            export_url = response.json()['url']
            return {
                "url": export_url,
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
            }
        else:
            raise HTTPException(status_code=response.status_code, detail="Export failed")


# Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests"""
    request_id = request.headers.get('X-Request-ID', str(hash(datetime.now())))
    
    with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", str(request.url))
        span.set_attribute("request.id", request_id)
        
        response = await call_next(request)
        
        span.set_attribute("http.status_code", response.status_code)
        response.headers["X-Request-ID"] = request_id
        
        return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    duration = (datetime.now() - start_time).total_seconds()
    REQUEST_DURATION.labels(method=request.method, endpoint=request.url.path).observe(duration)
    
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s"
    )
    
    return response


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status='error'
    ).inc()
    
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "request_id": request.headers.get('X-Request-ID')
    }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status='error'
    ).inc()
    
    return {
        "error": "Internal server error",
        "status_code": 500,
        "request_id": request.headers.get('X-Request-ID')
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_gateway:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
