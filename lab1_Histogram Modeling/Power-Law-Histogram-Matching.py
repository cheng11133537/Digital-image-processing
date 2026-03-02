import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('aerial_view.tif', cv2.IMREAD_GRAYSCALE)
src_flat = img.ravel()
src_hist = np.bincount(src_flat, minlength=256).astype(np.float64)
src_pdf  = src_hist / src_flat.size
src_cdf  = np.cumsum(src_pdf)
alpha = 0.4
z = np.arange(256, dtype=np.float64)
tgt_pdf = z**alpha
tgt_pdf /= tgt_pdf.sum()              
tgt_cdf = np.cumsum(tgt_pdf)
lut = np.searchsorted(tgt_cdf, src_cdf, side='left').astype(np.uint8)
img_mat = lut[img]
cv2.imwrite('mat.jpg', img_mat)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img_mat, cmap='gray')
plt.axis('off')
plt.title('Histogram Matched Image')

plt.subplot(1,2,2)
plt.hist(img_mat.ravel(), bins=256, range=(0,256), density=True)
plt.title('Histogram of Matched Image')
plt.xlabel('Gray Level')
plt.ylabel('Normalized Frequency')

plt.tight_layout()
plt.savefig('HistogramMatched_with_Hist.png')
plt.show()
