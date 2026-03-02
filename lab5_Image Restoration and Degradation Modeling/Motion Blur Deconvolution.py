import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift

IMG_PATH = "book-cover-blurred.tif"

def center_transform(img):
    M, N = img.shape
    u = np.arange(M)
    v = np.arange(N)
    X, Y = np.meshgrid(v, u)  
    D = X + Y
    return img * ((-1) ** D)

def filter_H_no_shift(M, N, a, b, T):
    u = np.arange(M) - M / 2.0
    v = np.arange(N) - N / 2.0
    V_mesh, U_mesh = np.meshgrid(v, u)  
    D = (V_mesh * b + U_mesh * a) * np.pi
    epsilon = 1e-12
    H = np.where(np.abs(D) < epsilon,
                 T,
                 T / D * np.sin(D))
    return H

def wiener_filter1(k, F_fft, H):
    H2 = np.abs(H) ** 2
    H_stabilized = np.where(np.abs(H) > 1e-6, H, 1e-6)
    W = (1.0 / H_stabilized) * (H2 / (H2 + k))
    return W

def read_image_and_convert_to_gray_float(filename):
    try:
        img = plt.imread(filename)
    except FileNotFoundError:
        print(f"Error: cannot open file: {filename}")
        return None
    except Exception as e:
        print(f"Error while reading image: {e}")
        return None
    if img.ndim == 3:
        img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        img_gray = img
    img_float = img_gray.astype(np.float64)
    if img_float.max() > 1.0:
        img_float /= 255.0
    return img_float

def main():
    fig_original = read_image_and_convert_to_gray_float(IMG_PATH)
    if fig_original is None:
        print("Failed to load book-cover-blurred image. Please check the path.")
        return
    H_book, W_book = fig_original.shape
    print("Image size:", H_book, "x", W_book)
    F_book = fft2(center_transform(fig_original))
    a_motion = 0.1   
    b_motion = 0.1    
    T_motion = 1.0
    threshold_stab = 1e-6
    k_wiener_motion = 0.001  
    h_no_shift = filter_H_no_shift(H_book, W_book, a_motion, b_motion, T_motion)
    H_safe = np.where(np.abs(h_no_shift) > threshold_stab,
                      h_no_shift,
                      threshold_stab)
    inverse_restored_fft = F_book / H_safe
    inverse_restored_raw = np.clip( np.real(ifftshift(ifft2(inverse_restored_fft))), 0, 1 )
    wiener_restored_book_fft = wiener_filter1(k_wiener_motion, F_book, h_no_shift)
    wiener_restored_book_raw = np.clip(
        np.real(ifftshift(ifft2(wiener_restored_book_fft * F_book))), 0, 1)
    SHIFT_Y_2 = H_book // 2
    SHIFT_X_2 = W_book // 2
    inverse_restored = np.roll(np.roll(inverse_restored_raw, SHIFT_Y_2, axis=0),SHIFT_X_2, axis=1)
    wiener_restored_book = np.roll(np.roll(wiener_restored_book_raw, SHIFT_Y_2, axis=0),SHIFT_X_2, axis=1)
    plt.imsave("inverse_restored.png", inverse_restored, cmap="gray")
    plt.imsave("wiener_restored.png", wiener_restored_book, cmap="gray")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(fig_original, cmap='gray')
    plt.title('Book Cover - Blurred Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(inverse_restored, cmap='gray')
    plt.title('Inverse Filtering (shift corrected)')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(wiener_restored_book, cmap='gray')
    plt.title(f'Wiener Filtering (K={k_wiener_motion}, shift corrected)')
    plt.axis('off')
    plt.suptitle('Restoration of book-cover-blurred.tif (Motion Blur)')
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()
    print("Book cover restoration finished. Saved:")
    print("  inverse_restored.png")
    print("  wiener_restored.png")
    print("  comparison.png")

if __name__ == "__main__":
    main()
