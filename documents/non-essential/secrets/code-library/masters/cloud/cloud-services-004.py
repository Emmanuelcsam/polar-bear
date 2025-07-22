"""Mock websockets module for testing."""

class WebSocketServerProtocol:
    async def send(self, data):
        pass
    
    async def recv(self):
        return b"test data"

async def serve(handler, host, port):
    """Mock websocket server."""
    class Server:
        async def wait_closed(self):
            pass
    return Server()

class ConnectionClosed(Exception):
    pass
