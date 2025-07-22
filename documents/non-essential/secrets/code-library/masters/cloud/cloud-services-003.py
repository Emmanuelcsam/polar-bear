"""Mock serial module for testing."""

class Serial:
    def __init__(self, port, baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_open = True
    
    def read(self, size=1):
        return b"$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
    
    def readline(self):
        return self.read()
    
    def write(self, data):
        return len(data)
    
    def close(self):
        self.is_open = False
