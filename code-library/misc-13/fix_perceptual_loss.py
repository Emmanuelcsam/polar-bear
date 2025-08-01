#!/usr/bin/env python3
import re

# Read the original file
with open('utilities/losses.py', 'r') as f:
    content = f.read()

# Find and replace the normalize_batch method
old_method = '''    def normalize_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a batch of images using ImageNet stats."""
        return (x - self.mean) / self.std'''

new_method = '''    def normalize_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a batch of images using ImageNet stats."""
        # Handle different channel sizes by converting to 3-channel
        if x.shape[1] == 1:  # Grayscale
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:  # Other channel sizes
            # Use adaptive pooling to convert to 3 channels
            if x.shape[1] > 3:
                x = x[:, :3, :, :]  # Take first 3 channels
            elif x.shape[1] < 3:
                # Pad with zeros to get 3 channels
                padding = torch.zeros(x.shape[0], 3 - x.shape[1], x.shape[2], x.shape[3], device=x.device)
                x = torch.cat([x, padding], dim=1)
        return (x - self.mean) / self.std'''

# Replace the method
content = content.replace(old_method, new_method)

# Write the modified content back
with open('utilities/losses.py', 'w') as f:
    f.write(content)

print("Fixed normalize_batch method in losses.py")
