import numpy as np
import cv2

height, width = 512, 512

while True:
    color_pixels = np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8)

    cv2.imshow('Random Image Generator', color_pixels)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
