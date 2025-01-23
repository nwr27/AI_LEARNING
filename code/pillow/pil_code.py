from PIL import Image

img = Image.open("img/cctv.png")
img = img.convert("L")
img = img.filter()
img.show()