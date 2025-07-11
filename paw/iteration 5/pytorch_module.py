#!/usr/bin/env python3
"""
Simple PyTorch demo: pass logged values through a Linear.
"""
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[Torch] PyTorch not available")

from data_store import load_events

def torch_learn():
    """Process logged values through a PyTorch Linear layer."""
    if not TORCH_AVAILABLE:
        print("[Torch] PyTorch not available, skipping")
        return None

    events = load_events()
    vals = [e.get("intensity", e.get("pixel")) for e in events]
    vals = [v for v in vals if v is not None]

    if not vals:
        print("[Torch] No data")
        return None

    try:
        x = torch.tensor(vals, dtype=torch.float32).unsqueeze(1)  # shape (N,1)
        layer = torch.nn.Linear(1, 1)
        out = layer(x)
        result = out[:5].detach().numpy() if len(out) >= 5 else out.detach().numpy()
        print("[Torch] First outputs:", result)
        return result
    except Exception as e:
        print(f"[Torch] Error: {e}")
        return None

if __name__ == "__main__":
    torch_learn()
