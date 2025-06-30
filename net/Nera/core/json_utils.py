#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced JSON utilities with performance optimization
Supports orjson for 6x faster JSON operations with fallback to standard json.
"""
import json
from typing import Any, Dict, Union

try:
    from .config import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import config


class JSONEncoder:
    """Enhanced JSON encoder with performance optimization."""

    def __init__(self):
        self.use_orjson = config.features.get('orjson', False)
        if self.use_orjson:
            try:
                import orjson
                self.orjson = orjson
            except ImportError:
                self.use_orjson = False

    def dumps(self, obj: Any, **kwargs) -> str:
        """Serialize object to JSON string."""
        if self.use_orjson:
            # orjson returns bytes, need to decode to str
            return self.orjson.dumps(obj).decode('utf-8')
        else:
            # Use standard json with custom handling for sets and other types
            return json.dumps(obj, default=self._json_serializer, **kwargs)

    def loads(self, data: Union[str, bytes]) -> Any:
        """Deserialize JSON string to object."""
        if self.use_orjson:
            return self.orjson.loads(data)
        else:
            return json.loads(data)

    def _json_serializer(self, obj: Any) -> Any:
        """Custom serializer for non-standard types."""
        if isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Global JSON encoder instance
json_encoder = JSONEncoder()

# Convenience functions
def dumps(obj: Any, **kwargs) -> str:
    """Serialize object to JSON string."""
    return json_encoder.dumps(obj, **kwargs)

def loads(data: Union[str, bytes]) -> Any:
    """Deserialize JSON string to object."""
    return json_encoder.loads(data)
