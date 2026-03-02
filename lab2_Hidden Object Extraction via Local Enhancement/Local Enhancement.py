import cv2
import numpy as np

def apply_gamma(img, gamma):
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def method_c(img, gamma=0.2, clip=8.0, tiles=1):
    img_g = apply_gamma(img, gamma)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return clahe.apply(img_g)

def post_ab(img, alpha=1.4, beta=30):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def main():
    input_file = "hidden_object.jpg"   
    output_file = "LE_result.png"      
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    base = method_c(img, gamma=0.2, clip=8.0, tiles=1)
    result = post_ab(base, alpha=1.4, beta=30)
    cv2.imwrite(output_file, result)
    cv2.imshow("Enhanced", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
