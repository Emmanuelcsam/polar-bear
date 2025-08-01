
from PIL import Image

# Create a small black image
img = Image.new('RGB', (256, 256), 'black')
img.save('ref.jpg', 'jpeg')

print("Created dummy ref.jpg")
