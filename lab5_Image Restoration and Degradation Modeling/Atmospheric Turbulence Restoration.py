import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = r"C:\Users\邱榮誠\Desktop\DIP HW5_1\book-cover-blurred.tif"
OUT_DIR = os.path.dirname(IMG_PATH) or "."
K_TUNED = 0.13
K_BORDER_EST_GUARD = 1e-12
EPS = 1e-6
KERNEL_K = 0.0025
RADIAL_RADIUS = 45
BORDER_WIDTH_FOR_K = 5

def load_gray(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def save_float_as_uint8(arr, path):
    a = np.clip(arr, 0.0, 1.0)
    im = Image.fromarray((a * 255.0).astype(np.uint8))
    im.save(path)

def build_turbulence_H(M, N, k=0.0025):
    u = np.arange(-M//2, M//2)
    v = np.arange(-N//2, N//2)
    U, V = np.meshgrid(v, u)
    r2 = (U.astype(np.float64)**2 + V.astype(np.float64)**2)
    H = np.exp(-k * (r2 ** (5.0/6.0)))
    return np.fft.ifftshift(H)

def fft2(img):
    return np.fft.fft2(img)

def ifft2(spec):
    return np.fft.ifft2(spec)

def radial_mask(M, N, radius):
    u = np.arange(-M//2, M//2)
    v = np.arange(-N//2, N//2)
    U, V = np.meshgrid(v, u)
    R = np.sqrt(U**2 + V**2)
    mask = (R <= radius).astype(np.float32)
    return np.fft.ifftshift(mask)

def normalize01(x):
    x = np.array(x, dtype=np.float64)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

if __name__ == "__main__":
    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"Input image not found: {IMG_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)
    img = load_gray(IMG_PATH)
    M, N = img.shape
    print(f"Loaded image {IMG_PATH} shape={img.shape}")
    H = build_turbulence_H(M, N, k=KERNEL_K)
    G = fft2(img)
    H_safe = H.copy()
    H_safe[np.abs(H_safe) < EPS] = EPS
    F_hat_full = G / H_safe
    f_hat_full = np.real(ifft2(F_hat_full))
    mask = radial_mask(M, N, RADIAL_RADIUS)
    F_hat_limited = (G / H_safe) * mask
    f_hat_limited = np.real(ifft2(F_hat_limited))
    bw = BORDER_WIDTH_FOR_K
    top = img[:bw, :].ravel()
    bottom = img[-bw:, :].ravel()
    left = img[:, :bw].ravel()
    right = img[:, -bw:].ravel()
    border_pixels = np.concatenate([top, bottom, left, right])
    sigma_n2 = np.var(border_pixels) + 0.0
    sigma_img2 = np.var(img)
    sigma_f2 = max(sigma_img2 - sigma_n2, 1e-9)
    K_est = sigma_n2 / (sigma_f2 + K_BORDER_EST_GUARD)
    H_conj = np.conjugate(H)
    den_est = (np.abs(H)**2) + K_est
    W_est = H_conj / den_est
    F_hat_wiener_est = W_est * G
    f_hat_wiener_est = np.real(ifft2(F_hat_wiener_est))
    den_tuned = (np.abs(H)**2) + K_TUNED
    W_tuned = H_conj / den_tuned
    F_hat_wiener_tuned = W_tuned * G
    f_hat_wiener_tuned = np.real(ifft2(F_hat_wiener_tuned))
    orig_vis = normalize01(img)
    full_vis = normalize01(f_hat_full)
    limited_vis = normalize01(f_hat_limited)
    wiener_est_vis = normalize01(f_hat_wiener_est)
    wiener_tuned_vis = normalize01(f_hat_wiener_tuned)
    save_float_as_uint8(orig_vis, os.path.join(OUT_DIR, "Fig5.25_original.png"))
    save_float_as_uint8(full_vis, os.path.join(OUT_DIR, "Fig5.25_inverse_full.png"))
    save_float_as_uint8(limited_vis, os.path.join(OUT_DIR, "Fig5.25_inverse_radial.png"))
    save_float_as_uint8(wiener_est_vis, os.path.join(OUT_DIR, "Fig5.25_wiener_K_est.png"))
    save_float_as_uint8(wiener_tuned_vis, os.path.join(OUT_DIR, "Fig5.25_wiener_K_tuned.png"))
    labels = [
        "Original (blurred)",
        "Inverse - Full",
        f"Inverse - Radial r={RADIAL_RADIUS}",
        f"Wiener K_est={K_est:.3e}",
        f"Wiener K={K_TUNED}"
    ]
    arrays = [orig_vis, full_vis, limited_vis, wiener_est_vis, wiener_tuned_vis]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    show_arrays = [orig_vis, limited_vis, wiener_tuned_vis]
    show_labels = ["Original (blurred)",
                   f"Inverse - Radial r={RADIAL_RADIUS}",
                   f"Wiener K={K_TUNED}"]
    for ax, arr, title in zip(axes, show_arrays, show_labels):
        ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.suptitle("Restoration Result (Only Final images)", fontsize=12)
    plt.tight_layout()
    plt.show()
    imgs = [Image.fromarray((a * 255).astype(np.uint8)).convert("L") for a in arrays]
    spacing = 10
    w_total = sum(im.width for im in imgs) + spacing * (len(imgs) - 1)
    h_total = imgs[0].height + 24
    canvas = Image.new("L", (w_total, h_total), color=255)
    x = 0
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, im in enumerate(imgs):
        canvas.paste(im, (x, 0))
        draw = ImageDraw.Draw(canvas)
        text = labels[i]
        text_w, text_h = draw.textsize(text, font=font)
        tx = x + (im.width - text_w)//2
        ty = im.height + 6
        draw.text((tx, ty), text, fill=0, font=font)
        x += im.width + spacing
    out_comp = os.path.join(OUT_DIR, "Fig525_results_comparison.png")
    canvas.save(out_comp)
    print("Saved outputs to:", OUT_DIR)
    print(" - Fig5.25_original.png")
    print(" - Fig5.25_inverse_radial.png")
    print(" - Fig5.25_wiener_K_tuned.png")
    print(" - Fig525_results_comparison.png")
    print(f"Estimated K from border variance: {K_est:.6e}")
