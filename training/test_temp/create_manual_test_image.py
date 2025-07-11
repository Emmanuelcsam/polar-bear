from PIL import Image
import os

# Create a green image for the dataset folder for manual testing
img_green = Image.new('RGB', (60, 30), color = 'green')
img_green.save('test_temp/dataset/manual_test_image.png')
