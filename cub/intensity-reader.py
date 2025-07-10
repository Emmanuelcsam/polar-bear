from PIL import Image
img = Image.open(r"C:\Users\Saem1001\Documents\GitHub\polar-bear\training\dataset\1-91-clean.jpg").convert('L')
while 1:
    for p in img.getdata(): print(p)