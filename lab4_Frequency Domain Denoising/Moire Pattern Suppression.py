import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def read_gray(path: str) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0

def fft_shift(img01: np.ndarray) -> np.ndarray:
    dft = cv.dft(img01, flags=cv.DFT_COMPLEX_OUTPUT)
    return np.fft.fftshift(dft, axes=(0, 1))

def ifft_shift_to_img(dft_shift: np.ndarray) -> np.ndarray:
    ishift = np.fft.ifftshift(dft_shift, axes=(0, 1))
    img_back = cv.idft(ishift, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
    img_back = np.clip(img_back, 0, 1)
    return img_back

def log_spectrum(dft_shift: np.ndarray) -> np.ndarray:
    mag = cv.magnitude(dft_shift[...,0], dft_shift[...,1])
    mag = np.log1p(mag)
    mag_norm = cv.normalize(mag, None, 0, 1, cv.NORM_MINMAX)
    return mag_norm

def detect_peaks(dft_shift: np.ndarray, K=8, center_exclusion=0.05, cross_w=6, border_ratio=0.02):
    H, W, _ = dft_shift.shape
    cy, cx = H // 2, W // 2
    yv, xv = np.ogrid[:H, :W]
    mag = cv.magnitude(dft_shift[...,0], dft_shift[...,1])
    R = np.sqrt((yv - cy)**2 + (xv - cx)**2)
    mask = np.ones((H, W), dtype=bool)
    mask &= R > (min(H, W) * center_exclusion)
    b = int(min(H, W) * border_ratio)
    mask[:b,:] = mask[-b:,:] = mask[:,:b] = mask[:,-b:] = False
    mask &= (np.abs(xv - cx) > cross_w) & (np.abs(yv - cy) > cross_w)
    mag_mask = mag.copy()
    mag_mask[~mask] = 0.0
    dist = R / (R.max() + 1e-6)
    score = mag_mask * (1.0 + dist)
    idx = np.argpartition(score.ravel(), -K)[-K:]
    coords = np.column_stack(np.unravel_index(idx, score.shape))
    selected = []
    min_sep = 12.0
    for (py, px) in coords:
        if all((py - sy)**2 + (px - sx)**2 >= min_sep**2 for (sy,sx) in selected):
            selected.append((int(py), int(px)))
    return selected[:max(1, K//2)]

def gaussian_notch_mask(shape, peaks, sigma=10.0):
    H, W = shape
    cy, cx = H // 2, W // 2
    yv, xv = np.ogrid[:H, :W]
    mask = np.ones((H, W), dtype=np.float32)

    for (y0, x0) in peaks:
        d2 = (yv - y0)**2 + (xv - x0)**2
        notch = 1.0 - np.exp(-d2 / (2.0 * sigma**2))
        mask *= notch
        y1 = (2*cy - y0) % H
        x1 = (2*cx - x0) % W
        d2s = (yv - y1)**2 + (xv - x1)**2
        notch_s = 1.0 - np.exp(-d2s / (2.0 * sigma**2))
        mask *= notch_s
    return mask

def apply_mask_complex(dft_shift: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = dft_shift.copy()
    out[...,0] *= mask
    out[...,1] *= mask
    return out

def main():
    in_path = "car-moire-pattern.tif" if len(sys.argv) < 2 else sys.argv[1]
    stem, _ = os.path.splitext(in_path)
    img = read_gray(in_path)
    dft_s = fft_shift(img)
    spec_before = log_spectrum(dft_s)
    plt.imsave(f"{stem}_spectrum.png", spec_before, cmap='gray')
    peaks = detect_peaks(dft_s, K=8, center_exclusion=0.05)
    print("Detected peaks (y,x):", peaks)
    sigma = 12.0
    mask = gaussian_notch_mask(img.shape, peaks, sigma=sigma)
    dft_f = apply_mask_complex(dft_s, mask)
    spec_after = log_spectrum(dft_f)
    plt.imsave(f"{stem}_spectrum_filtered.png", spec_after, cmap='gray')
    plt.imsave(f"{stem}_mask.png", mask, cmap='gray')
    img_out = ifft_shift_to_img(dft_f)
    plt.imsave(f"{stem}_denoised.png", img_out, cmap='gray')
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1); plt.title("Original"); 
    plt.imshow(img, cmap='gray'); plt.axis('off')
    plt.subplot(1,2,2); plt.title("Denoised"); 
    plt.imshow(img_out, cmap='gray'); plt.axis('off')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
