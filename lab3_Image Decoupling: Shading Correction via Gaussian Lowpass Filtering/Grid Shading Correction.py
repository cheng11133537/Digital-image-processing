import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_kernel(ker_size: int, sigma: float) -> np.ndarray:
    c = ker_size // 2
    y, x = np.ogrid[:ker_size, :ker_size]
    g = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    g /= g.sum()  
    return g

if __name__ == "__main__":
    img = cv2.imread("checkerboard1024-shaded.tif", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    ksize, sigma = 256.0, 64.0
    kernel = gaussian_kernel(ksize, sigma)
    shading = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT101)
    floor = np.percentile(shading, 5)
    shading_safe = np.maximum(shading, floor)
    gain = np.median(shading_safe)
    corrected = img * (gain / shading_safe)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    sh_norm = cv2.normalize(shading, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("original.png", img.astype(np.uint8))
    cv2.imwrite("shading.png", sh_norm)
    cv2.imwrite("corrected.png", corrected)
