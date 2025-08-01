#!/usr/bin/env python3
import re

# Read the original file
with open('utilities/losses.py', 'r') as f:
    content = f.read()

# Find and replace the reconstruction loss section
old_reconstruction_section = '''        # 7. Reconstruction loss
        if 'reconstruction' in predictions and 'image' in targets:
            losses['reconstruction'] = self.l1(
                predictions['reconstruction'],
                targets['image']
            )'''

new_reconstruction_section = '''        # 7. Reconstruction loss
        if 'reconstruction' in predictions and 'image' in targets:
            pred_recon = predictions['reconstruction']
            target_img = targets['image']
            
            # Handle size mismatch between reconstruction and target
            if pred_recon.shape != target_img.shape:
                # Resize to match the smaller tensor to avoid information loss
                min_h = min(pred_recon.shape[2], target_img.shape[2])
                min_w = min(pred_recon.shape[3], target_img.shape[3])
                
                if pred_recon.shape[2:] != (min_h, min_w):
                    pred_recon = F.interpolate(pred_recon, size=(min_h, min_w), mode='bilinear', align_corners=False)
                if target_img.shape[2:] != (min_h, min_w):
                    target_img = F.interpolate(target_img, size=(min_h, min_w), mode='bilinear', align_corners=False)
                    
                # Handle channel mismatch
                min_c = min(pred_recon.shape[1], target_img.shape[1])
                if pred_recon.shape[1] != target_img.shape[1]:
                    pred_recon = pred_recon[:, :min_c, :, :]
                    target_img = target_img[:, :min_c, :, :]
            
            losses['reconstruction'] = self.l1(pred_recon, target_img)'''

# Replace the section
content = content.replace(old_reconstruction_section, new_reconstruction_section)

# Write the modified content back
with open('utilities/losses.py', 'w') as f:
    f.write(content)

print("Fixed reconstruction loss with robust tensor shape handling")
