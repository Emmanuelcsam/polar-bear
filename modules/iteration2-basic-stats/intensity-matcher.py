from PIL import Image; import random
img = list(Image.open("image.jpg").convert('L').getdata())
while 1:
    r=random.randint(0,255); print(f"{r} {img.count(r)/len(img)*100:.2f}%" if r in img else f"{r} 0%")
