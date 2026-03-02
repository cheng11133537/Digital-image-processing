import cv2
import numpy as np
import math
from pathlib import Path
from skimage.measure import shannon_entropy  

INPUT_NAME = "hidden_object.jpg"     
OUTPUT_NAME = "hsm_best.jpg"

def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    p = Path(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite_unicode(path, img, ext=".jpg"):
    ok, buf = cv2.imencode(ext, img)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))

def var(pdf, mu):
    return sum(((i - mu) ** 2) * pdf[i] for i in range(len(pdf)))

def mean(pdf):
    return sum(i * pdf[i] for i in range(len(pdf)))

def hist(img, dim, k0, k1, k2, k3, e):
    assert dim % 2 != 0
    height, width = img.shape[:2]
    img_pdf = np.zeros(256, dtype=np.float64)
    img_flat = img.ravel()
    for v in img_flat:
        img_pdf[int(v)] += 1.0
    img_pdf /= img_flat.size
    img_mean = mean(img_pdf)
    print(f"Global mean = {img_mean:.3f}")
    img_sd = math.sqrt(var(img_pdf, img_mean))
    print(f"Global std  = {img_sd:.3f}")
    local_area = dim * dim
    r = dim // 2
    out = img.copy()
    local_mean_img = np.full(img.shape, np.nan, dtype=np.float64)
    local_std_img  = np.full(img.shape, np.nan, dtype=np.float64)
    step = 50
    max_print = 20
    printed = 0
    for i in range(r, height - r):
        for j in range(r, width - r):
            local_pdf = np.zeros(256, dtype=np.float64)
            for q in range(i - r, i + r + 1):
                for p in range(j - r, j + r + 1):
                    local_pdf[int(img[q, p])] += 1.0 / local_area
            lm = mean(local_pdf)
            ls = math.sqrt(var(local_pdf, lm))
            local_mean_img[i, j] = lm
            local_std_img[i, j]  = ls
            if printed < max_print and ((i - r) % step == 0) and ((j - r) % step == 0):
                printed += 1
    core_mean = local_mean_img[r:height - r, r:width - r]
    core_std  = local_std_img [r:height - r, r:width - r]
    core_img  = img           [r:height - r, r:width - r]
    mask_core = (
        (core_mean >= img_mean * k0) & (core_mean <= img_mean * k1) &
        (core_std  >= img_sd   * k2) & (core_std  <= img_sd   * k3)
    )
    print("Overall range of the original image:", int(img.min()), "to", int(img.max()))
    if np.any(mask_core):
        sel = core_img[mask_core]
        print("Enhanced area range:", int(sel.min()), "to", int(sel.max()))
    out_core = out[r:height - r, r:width - r]
    out_core[mask_core] = np.clip(out_core[mask_core] * e, 0, 255)
    return out.astype(np.uint8)

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    in_path = here / INPUT_NAME
    img1 = imread_unicode(in_path, cv2.IMREAD_GRAYSCALE)
    k0 = 0.0
    k1 = 0.10
    k2 = 0.0
    k3 = 0.06
    e  = 22.0
    out = hist(img1, dim=3, k0=k0, k1=k1, k2=k2, k3=k3, e=e)
    out_path = here / OUTPUT_NAME
    imwrite_unicode(out_path, out, ext=".jpg")
    print(f"\nOutput：{out_path}")
