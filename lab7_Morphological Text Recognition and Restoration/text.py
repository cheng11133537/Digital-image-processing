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
    sys.exit()

def clear_border_text(image_path):
    print(f"[STEP 1] Processing image borders: {image_path} ...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit()
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    H, W = binary_img.shape
    cleaned_img = binary_img.copy()
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        touches_border = (x <= 1) or (y <= 1) or (x + w >= W - 1) or (y + h >= H - 1)
        if touches_border:
            cleaned_img[labels == i] = 0
    output_filename = "cleaned_text.tif"
    cv2.imwrite(output_filename, cleaned_img)
    print(f"[INFO] Image cleaning complete. Saved as: {output_filename}")
    return output_filename

def recognize_text_google(image_path):
    print(f"[STEP 2] Uploading to Google Cloud Vision API for recognition...")
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["en"])
    response = client.document_text_detection(image=image, image_context=image_context)
    if response.error.message:
        raise Exception(f'API Error: {response.error.message}')
    return response.full_text_annotation.text

if __name__ == "__main__":
    input_image = 'text.tif' 
    if not os.path.exists(input_image):
        print(f"[WARNING] '{input_image}' not found.")
        if os.path.exists("2025-12-13 112047.jpg"):
             input_image = "2025-12-13 112047.jpg"
             print(f"[INFO] Using alternative file: {input_image}")
        else:
            print("[ERROR] No image file found. Please verify the file exists in the directory.")
            sys.exit()
    try:
        cleaned_image_path = clear_border_text(input_image)
        ocr_result = recognize_text_google(cleaned_image_path)
        print("\n==============================================")
        print("          FINAL OCR RESULT                    ")
        print("==============================================")
        print(ocr_result)
        print("==============================================")
    except Exception as e:
        print(f"\n[EXCEPTION] An error occurred: {e}")
