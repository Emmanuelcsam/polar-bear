"""Mock pynmea2 module for testing."""

class ParseError(Exception):
    pass

def parse(sentence):
    """Mock NMEA sentence parser."""
    class NMEASentence:
        def __init__(self):
            self.latitude = 48.117
            self.longitude = 11.517
            self.altitude = 545.4
            self.timestamp = "123519"
    
    return NMEASentence()
