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

def join_broken_text(image_path):
    print(f"[STEP 1] Processing image: {image_path} ...")
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"[ERROR] Could not read image file. Please check the filename: {image_path}")
        sys.exit()
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    joined_img = cv2.dilate(binary_img, kernel, iterations=1)
    output_filename = "joined_text_final.png"
    cv2.imwrite(output_filename, joined_img)
    print(f"[INFO] Image processing complete. Saved as: {output_filename}")
    return output_filename

def recognize_text_google(image_path):

    print(f"[STEP 2] Uploading to Google Cloud Vision API for recognition...")
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f'API Error: {response.error.message}')
    return response.full_text_annotation.text

if __name__ == "__main__":
    input_image = "text-broken.tif" 
    if not os.path.exists(input_image):
        print(f"[WARNING] '{input_image}' not found. Searching for common filenames...")
        if os.path.exists("image_2.png"):
             input_image = "image_2.png"
        elif os.path.exists("joined_text_result.png"):
             input_image = "joined_text_result.png"
        else:
            print("[ERROR] No image file found. Please verify the file exists and update the 'input_image' variable in the code.")
            sys.exit()
    try:
        processed_image_path = join_broken_text(input_image)
        ocr_result = recognize_text_google(processed_image_path)
        print("\n==============================================")
        print("          FINAL OCR RESULT                    ")
        print("==============================================")
        print(ocr_result)
        print("==============================================")
    except Exception as e:
        print(f"\n[EXCEPTION] An error occurred: {e}")
