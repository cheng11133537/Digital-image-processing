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

def correct_illumination(image_path):
    print(f"[STEP 1] Processing image: {image_path} ...")
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"[ERROR] Could not read image file: {image_path}")
        sys.exit()
    background_illumination = cv2.GaussianBlur(img, (101, 101), 0)
    result_float = cv2.divide(img, background_illumination, scale=255)
    result_uint8 = np.uint8(result_float)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(result_uint8)
    _, img_binary = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output_filename = "cleaned_spotshade_v2.png"
    cv2.imwrite(output_filename, img_binary)
    print(f"[INFO] Illumination correction complete. Saved as: {output_filename}")
    return output_filename

def recognize_text_google(image_path):
    print(f"[STEP 2] Uploading to Google Cloud Vision API ...")
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
    input_image = "text-spotshade.tif" 
    if not os.path.exists(input_image):
        if os.path.exists("text-spotshade.jpg"):
            input_image = "text-spotshade.jpg"
        elif os.path.exists("text-spotshade.png"):
            input_image = "text-spotshade.png"
        else:
            print(f"[ERROR] Input file '{input_image}' not found.")
            sys.exit()
    try:
        clean_image_path = correct_illumination(input_image)
        text_result = recognize_text_google(clean_image_path)
        print("\n==============================================")
        print("          FINAL OCR RESULT          ")
        print("==============================================")
        print(text_result)
        print("==============================================")
    except Exception as e:
        print(f"\n[EXCEPTION] An error occurred: {e}")
