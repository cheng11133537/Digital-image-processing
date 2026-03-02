import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('aerial_view.tif', cv2.IMREAD_GRAYSCALE)
flat = img.ravel()
hist = np.bincount(flat, minlength=256).astype(np.float64)
pdf  = hist / flat.size
cdf  = np.cumsum(pdf)
lut  = np.round(255 * cdf).astype(np.uint8)
img_eq = lut[img]
cv2.imwrite('Equalied Image.jpg', img_eq)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(img_eq, cmap='gray')
plt.axis("off")
plt.title("Equalized Image")

plt.subplot(1,2,2)
plt.hist(img_eq.ravel(), bins=256, range=(0,256), density=True, color='blue')
plt.title("Histogram of Equalized Image")
plt.xlabel("Gray Level")
plt.ylabel("Normalized Frequency")

plt.tight_layout()
plt.savefig('Equalized_with_Hist.png')
plt.show()
