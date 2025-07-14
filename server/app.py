import os
import uuid
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont
from flask_cors import CORS
import preProcessing as pp
import wordExtract as we
import templateOperation as tempOp


app = Flask(__name__)
CORS(app , supports_credentials=True)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route("/upload", methods=["POST"])
def upload():
    uploaded_image_file = request.files["image"]
    word = request.form.get("word")

    if not uploaded_image_file or not word:
        return jsonify({"error": "Missing image or word"}), 400

    img_data_buffer = uploaded_image_file.read() 
    img_array = np.frombuffer(img_data_buffer, np.uint8) 

    original_filename = uploaded_image_file.filename
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = str(uuid.uuid4()) + file_extension
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    originalImage = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if originalImage is None:
        return jsonify({"error": "Could not decode color image."}), 400

    greyImage = cv2.imdecode(buf=img_array , flags=cv2.IMREAD_GRAYSCALE )
    if greyImage is None:
        return jsonify({"error": "Could not decode grayscale image."}), 400
    
    word_is_nepali_for_overall_processing = tempOp.is_nepali(word)
    print(f"[DEBUG APP] The search word '{word}' indicates Nepali content: {word_is_nepali_for_overall_processing}") 

    # --- Step 1: Image after B/W conversion ---
    image = pp.backgroundBlackForegroundWhite(image=greyImage)
    if image is None:
        return jsonify({"error": "Background/Foreground normalization failed."}), 500
    print("✓ B/W conversion completed")

    averageHeight_Pixel, tallestText_coords = pp.averageHeightOfLetters(image=image)
    
    if averageHeight_Pixel is None or tallestText_coords is None:
        print("Warning: No valid text lines detected after averageHeightOfLetters. Terminating process.")
        return jsonify({"error": "Could not detect valid text lines (average height or coordinates missing). Image might be too degraded or blank."}), 400

    if not isinstance(tallestText_coords, tuple) or len(tallestText_coords) != 2 or tallestText_coords[1] <= tallestText_coords[0]:
        print("Warning: Invalid or empty tallest text line coordinates. Cannot proceed with gap calculation.")
        return jsonify({"error": "Detected text but coordinates for tallest line are invalid."}), 400

    # Extract the tallest line image using the coordinates
    tallestLineImage = image[tallestText_coords[0]:tallestText_coords[1], :]
    
    # --- Step 2: Tallest Line Image ---
    if tallestLineImage is not None and tallestLineImage.size > 0:
        print("✓ Tallest line extracted successfully")
    else:
        print("Warning: Sliced tallestLineImage is empty. Cannot proceed with gap calculation.")
        return jsonify({"error": "Extracted text line is empty. Image might be too degraded."}), 400

    averageGap_Pixel = pp.averageGapOfLetter(image=tallestLineImage)

    if averageGap_Pixel is None:
        print("Warning: Could not calculate average gap. Image might be too degraded for gap analysis.")
        return jsonify({"error": "Could not calculate average gap for text. Image might be too degraded."}), 400

    iterationNumber = pp.requiredNumberOfIterations(
        averageGap=averageGap_Pixel,
        is_nepali_text=word_is_nepali_for_overall_processing 
    )
    print(f"✓ Processing parameters: avg_height={averageHeight_Pixel}, gap={averageGap_Pixel}, iterations={iterationNumber}")

    # --- Step 3: Image after Smudging ---
    smudgedImage = pp.prepareImageForWordExtraction(image=image, iteration=iterationNumber)
    if smudgedImage is None:
        return jsonify({"error": "Image smudging failed, likely due to corrupted or empty image after preprocessing."}), 500
    print("✓ Image enhancement completed")

    # --- Step 4: All Extracted Words (Before Matching) ---
    wordsProperty = we.wordExtract(image=smudgedImage , 
                                   averageHeight=averageHeight_Pixel,
                                   smudgedIteration=iterationNumber)
    
    if not wordsProperty:
        print("Warning: No words extracted after word extraction stage.")
        return jsonify({"message": "No words found in the document to match against. Image might be too degraded."}), 200
        
    print(f"✓ Word extraction completed: {len(wordsProperty)} words found") 
    extractedWordsImage = originalImage.copy() 
    if len(wordsProperty) > 0:
        extractedWordsImage = tempOp.putRectangles(image=extractedWordsImage, wordProperty=wordsProperty) 

    print("✓ Creating template...")
    template = tempOp.createTemplate(word=word ,fontSize=averageHeight_Pixel)
    if template is None:
        return jsonify({"error": "Failed to create template for the word. Check font paths or word content."}), 500

    print("✓ Template created, starting template matching...")
    foundWords = tempOp.templateMatching(image=image , template=template , wordsProperty=wordsProperty, matching_mode="TM_CCOEFF_NORMED")

    print(f"✓ Template matching completed: {len(foundWords)} matches found") 

    # --- Step 5: Final Detected Words ---
    finalImage = originalImage.copy() 
    if len(foundWords) > 0:
        finalImage = tempOp.putRectangles(image=finalImage, wordProperty=foundWords)
    else:
        cv2.putText(img=finalImage , text="No match found",
                    org=(20,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=(0,255,0),
                    thickness=2)

    cleaned_found_words = []
    for box_pair in foundWords:
        x1, y1 = int(box_pair[0][0]), int(box_pair[0][1])
        x2, y2 = int(box_pair[1][0]), int(box_pair[1][1])
        cleaned_found_words.append(((x1, y1), (x2, y2)))

    # Encode all images to base64
    bw_base64 = encode_image_to_base64(image)
    smudged_base64 = encode_image_to_base64(smudgedImage)
    extracted_base64 = encode_image_to_base64(extractedWordsImage)
    final_base64 = encode_image_to_base64(finalImage)

    print("✓ Processing completed successfully")

    return jsonify({
        "message": "Image processed successfully",
        "fileName": unique_filename, 
        "foundWords": cleaned_found_words,
        "image_base64": final_base64,
        "bw_image_base64": bw_base64,
        "smudged_image_base64": smudged_base64,
        "extracted_image_base64": extracted_base64
    }), 200


@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)