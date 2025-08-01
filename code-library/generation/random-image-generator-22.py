import numpy as np
import cv2
height, width = 256, 256
grayscale_pixels = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
cv2.imwrite('random_grayscale.png', grayscale_pixels)
print(f" {height}x{width} image")
color_pixels = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
cv2.imwrite('random_color.png', color_pixels)
print(f"{height}x{width} random 'random_color.png'")
