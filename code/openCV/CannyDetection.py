import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r"img\lena.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
canny = cv2.Canny(img, 100, 200)

titles = ["image", "canny"]
images = [img, canny]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
