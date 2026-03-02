import cv2
import numpy as np
import matplotlib.pyplot as plt

gray_img = cv2.imread('aerial_view.tif', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('Original Image.jpg', gray_img)
img1 = gray_img.ravel()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(gray_img, cmap='gray')
plt.axis("off")
plt.title("Original Image")

plt.subplot(1,2,2)
plt.hist(img1, 256, [0,256], density=True)
plt.title("Histogram of Original Image")
plt.xlabel("Gray Level")
plt.ylabel("Normalized Frequency")

plt.tight_layout()
plt.show()
