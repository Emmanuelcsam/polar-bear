import torch
import cv2
import numpy as np

while 1:
    img = torch.load('generated.pt').squeeze().permute(1,2,0)
    img = ((img + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
    cv2.imshow('Generated', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()