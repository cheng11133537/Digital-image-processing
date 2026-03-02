import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

img_path = Path(r"C:\Users\邱榮誠\Desktop\DIPHW_CH6-1\lenna-RGB.tif")
img = cv2.imread(str(img_path))
if img is None:
    from PIL import Image
    img = np.array(Image.open(img_path).convert("RGB"))
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float64)
Gx_kernel = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float64)
Gy_kernel = np.array([[ 1,  2,  1],
                      [ 0,  0,  0],
                      [-1, -2, -1]], dtype=np.float64)
Gx = np.zeros_like(img)
Gy = np.zeros_like(img)
for c in range(3):   # R,G,B
    Gx[..., c] = cv2.filter2D(img[..., c], -1, Gx_kernel)
    Gy[..., c] = cv2.filter2D(img[..., c], -1, Gy_kernel)
grad = np.sqrt(np.sum(Gx**2 + Gy**2, axis=2))
grad = grad / grad.max() * 255
grad = grad.astype(np.uint8)
result_path = img_path.with_name("RGB_vector_gradient_b.png")
cv2.imwrite(str(result_path), grad)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img.astype(np.uint8))
plt.title("lenna-RGB")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(grad, cmap='gray')
plt.title("final grdient image")
plt.axis("off")
plt.tight_layout()
plt.show()
