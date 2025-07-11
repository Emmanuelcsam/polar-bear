
from PIL import Image
import os

# Create directories if they don't exist
os.makedirs("test_temp/reference", exist_ok=True)
os.makedirs("test_temp/dataset", exist_ok=True)

# Create a red image for the reference folder
img_red = Image.new('RGB', (60, 30), color = 'red')
img_red.save('test_temp/reference/red_image.png')

# Create a blue image for the reference folder
img_blue = Image.new('RGB', (60, 30), color = 'blue')
img_blue.save('test_temp/reference/blue_image.png')

# Create a red image for the dataset folder
img_red_dataset = Image.new('RGB', (60, 30), color = 'red')
img_red_dataset.save('test_temp/dataset/test_image.png')
