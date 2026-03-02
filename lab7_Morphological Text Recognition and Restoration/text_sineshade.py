import cv2
import numpy as np
import os
import sys
from google.cloud import vision
current_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(current_dir, "key.json")
if os.path.exists(key_path):
    print(f"[INFO] Successfully loaded Service Account Key: {key_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
else:
    print(f"[ERROR] Key file not found at: {key_path}")
    print("[ERROR] Please make sure 'key.json' is in the same folder.")
    sys.exit()

def remove_periodic_noise(image_path):
    print(f"[STEP 1] Processing image: {image_path} ...")
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"[ERROR] Could not read image file: {image_path}")
        sys.exit()
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r_vertical = 15   
    r_horizontal = 10 
    mask[crow-r_vertical:crow+r_vertical, ccol-r_horizontal:ccol-1] = 0
    mask[crow-r_vertical:crow+r_vertical, ccol+1:ccol+r_horizontal] = 0
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_clean = np.uint8(img_back)
    _, img_binary = cv2.threshold(img_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output_filename = "cleaned_image_result.png"
    cv2.imwrite(output_filename, img_binary)
    print(f"[INFO] Noise removal complete. Saved as: {output_filename}")
    return output_filename

def recognize_text_google(image_path):
    print(f"[STEP 2] Uploading to Google Cloud Vision API ...")
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f'API Error: {response.error.message}')
    full_text = response.full_text_annotation.text
    return full_text
if __name__ == "__main__":
    input_image = "text-sineshade.tif"
    if not os.path.exists(input_image):
        print(f"[ERROR] Input file '{input_image}' not found.")
        sys.exit()
    try:
        clean_image_path = remove_periodic_noise(input_image)
        text_result = recognize_text_google(clean_image_path)
        print("\n==============================================")
        print("          FINAL OCR RESULT                    ")
        print("==============================================")
        print(text_result)
        print("==============================================")
        
    except Exception as e:
        print(f"\n[EXCEPTION] An error occurred: {e}")
