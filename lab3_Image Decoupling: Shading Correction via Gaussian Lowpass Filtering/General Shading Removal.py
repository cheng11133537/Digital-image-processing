import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    assert ksize % 2 == 1 and ksize > 1
    c = ksize // 2
    y, x = np.ogrid[-c:c+1, -c:c+1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma * sigma))).astype(np.float32)
    g /= g.sum()
    return g

def remove_shading_lpf(img_gray: np.ndarray, ksize: int = 127, sigma: float = 31.0, floor_percentile: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    img_f = img_gray.astype(np.float32)
    kernel = gaussian_kernel(ksize, sigma)
    shading = cv2.filter2D(img_f, ddepth=-1, kernel=kernel,borderType=cv2.BORDER_REFLECT101)
    floor = float(np.percentile(shading, floor_percentile))
    shading_safe = np.maximum(shading, floor)
    gain = float(np.median(shading_safe))
    corrected = img_f * (gain / shading_safe)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return shading, corrected

if __name__ == "__main__":
    img = cv2.imread("N1.bmp", cv2.IMREAD_GRAYSCALE)
    KSIZE = 121
    SIGMA = 20.0
    shading, corrected = remove_shading_lpf(img, ksize=KSIZE, sigma=SIGMA, floor_percentile=5.0)
    cv2.imwrite("original.png", img)
    shading_vis = cv2.normalize(shading, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("shading.png", shading_vis)
    cv2.imwrite("corrected.png", corrected)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap="gray");       plt.title("Original");        plt.axis("on")
    plt.subplot(1, 3, 2); plt.imshow(shading, cmap="gray");   plt.title("Shading (LPF)");   plt.axis("on")
    plt.subplot(1, 3, 3); plt.imshow(corrected, cmap="gray"); plt.title("Corrected");       plt.axis("on")
    plt.tight_layout(); plt.show()
