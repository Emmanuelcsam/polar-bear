#!/usr/bin/env python3
import re

# Read the original file
with open('utilities/losses.py', 'r') as f:
    content = f.read()

# Find and replace the problematic line in the forward method
old_forward_section = '''                # Compute squared L2 distance
                diff = (feat_pred - feat_target) ** 2'''

new_forward_section = '''                # Compute squared L2 distance
                # Ensure features have same spatial dimensions
                if feat_pred.shape != feat_target.shape:
                    # Resize to match the smaller tensor
                    min_h = min(feat_pred.shape[2], feat_target.shape[2])
                    min_w = min(feat_pred.shape[3], feat_target.shape[3])
                    min_c = min(feat_pred.shape[1], feat_target.shape[1])
                    
                    if feat_pred.shape[2:] != (min_h, min_w):
                        feat_pred = F.interpolate(feat_pred, size=(min_h, min_w), mode='bilinear', align_corners=False)
                    if feat_target.shape[2:] != (min_h, min_w):
                        feat_target = F.interpolate(feat_target, size=(min_h, min_w), mode='bilinear', align_corners=False)
                    
                    # Handle channel dimension mismatch
                    if feat_pred.shape[1] != feat_target.shape[1]:
                        if feat_pred.shape[1] > min_c:
                            feat_pred = feat_pred[:, :min_c, :, :]
                        if feat_target.shape[1] > min_c:
                            feat_target = feat_target[:, :min_c, :, :]
                
                diff = (feat_pred - feat_target) ** 2'''

# Replace the section
content = content.replace(old_forward_section, new_forward_section)

# Add the import for F at the top if not present
if 'import torch.nn.functional as F' not in content:
    # Find the imports section and add F import
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('import torch.nn as nn'):
            lines.insert(i + 1, 'import torch.nn.functional as F')
            break
    content = '\n'.join(lines)

# Write the modified content back
with open('utilities/losses.py', 'w') as f:
    f.write(content)

print("Fixed perceptual loss forward method with robust tensor shape handling")
