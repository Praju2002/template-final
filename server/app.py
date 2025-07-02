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

# ==== API Endpoint ====
@app.route("/upload", methods=["POST"])
def upload():
    # Store the FileStorage object
    uploaded_image_file = request.files["image"]
    word = request.form.get("word")

    if not uploaded_image_file or not word:
        return jsonify({"error": "Missing image or word"}), 400

    # --- START OF FIX ---
    # Read the image data INTO A BUFFER ONCE
    img_data_buffer = uploaded_image_file.read() 
    img_array = np.frombuffer(img_data_buffer, np.uint8) # Use the buffer here
    # --- END OF FIX ---

    # Generate a unique filename and save the image using the buffer
    original_filename = uploaded_image_file.filename
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = str(uuid.uuid4()) + file_extension
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Save the image using the buffer (instead of image_file.save())
    # You might need to use cv2.imwrite or similar if you want to save from img_array
    # For now, let's just make sure cv2.imdecode works from the buffer
    
    # Optional: Save the image if you really need a copy on disk
    # with open(image_path, 'wb') as f:
    #     f.write(img_data_buffer)

    # Decode image from the buffer (which is now correctly populated)
    originalImage = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if originalImage is None:
        return jsonify({"error": "Could not decode color image."}), 400

    # Use the same img_array for grayscale decoding
    greyImage = cv2.imdecode(buf=img_array , flags=cv2.IMREAD_GRAYSCALE )
    if greyImage is None:
        return jsonify({"error": "Could not decode grayscale image."}), 400
    word_is_nepali_for_overall_processing = tempOp.is_nepali(word)
    print(f"[DEBUG APP] The search word '{word}' indicates Nepali content: {word_is_nepali_for_overall_processing}") # <--- DEBUG PRINT

    # --- View 1: Image after B/W conversion ---
    image = pp.backgroundBlackForegroundWhite(image=greyImage)
    if image is None:
        return jsonify({"error": "Background/Foreground normalization failed."}), 500
    cv2.imshow("1. Image after B/W conversion", image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


    averageHeight_Pixel, tallestText_coords = pp.averageHeightOfLetters(image=image)
    
    # --- Robustness Checks for line detection results ---
    if averageHeight_Pixel is None or tallestText_coords is None:
        print("Warning: No valid text lines detected after averageHeightOfLetters. Terminating process.")
        return jsonify({"error": "Could not detect valid text lines (average height or coordinates missing). Image might be too degraded or blank."}), 400

    if not isinstance(tallestText_coords, tuple) or len(tallestText_coords) != 2 or tallestText_coords[1] <= tallestText_coords[0]:
        print("Warning: Invalid or empty tallest text line coordinates. Cannot proceed with gap calculation.")
        return jsonify({"error": "Detected text but coordinates for tallest line are invalid."}), 400

    # Extract the tallest line image using the coordinates
    tallestLineImage = image[tallestText_coords[0]:tallestText_coords[1], :]
    
    # --- View 2: Tallest Line Image ---
    if tallestLineImage is not None and tallestLineImage.size > 0:
        cv2.imshow("2. Tallest Line Image (for gap calculation)", tallestLineImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Warning: Sliced tallestLineImage is empty. Cannot proceed with gap calculation.")
        return jsonify({"error": "Extracted text line is empty. Image might be too degraded."}), 400
    

    averageGap_Pixel = pp.averageGapOfLetter(image=tallestLineImage)

    if averageGap_Pixel is None:
        print("Warning: Could not calculate average gap. Image might be too degraded for gap analysis.")
        return jsonify({"error": "Could not calculate average gap for text. Image might be too degraded."}), 400


    # iterationNumber = pp.requiredNumberOfIterations(averageGap=averageGap_Pixel)
    iterationNumber = pp.requiredNumberOfIterations(
        averageGap=averageGap_Pixel,
        is_nepali_text=word_is_nepali_for_overall_processing # <--- ADD THIS ARGUMENT
    )
    print("avg ht, gap and iteration", averageHeight_Pixel, averageGap_Pixel, iterationNumber)


    # --- View 3: Image after Smudging ---
    smudgedImage = pp.prepareImageForWordExtraction(image=image, iteration=iterationNumber)
    if smudgedImage is None:
        return jsonify({"error": "Image smudging failed, likely due to corrupted or empty image after preprocessing."}), 500
    cv2.imshow("3. Image after Smudging", smudgedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # --- View 4: All Extracted Words (Before Matching) ---
    wordsProperty = we.wordExtract(image=smudgedImage , 
                                   averageHeight=averageHeight_Pixel,
                                   smudgedIteration=iterationNumber)
    
    if not wordsProperty: # Check if wordsProperty is empty
        print("Warning: No words extracted after word extraction stage.")
        return jsonify({"message": "No words found in the document to match against. Image might be too degraded."}), 200
        
    print(f"Number of words extracted by wordExtract: {len(wordsProperty)}") 
    extractedWordsImage = originalImage.copy() # Use original color image for drawing
    if len(wordsProperty) > 0:
        extractedWordsImage = tempOp.putRectangles(image=extractedWordsImage, wordProperty=wordsProperty) 
    
    cv2.imshow("4. All Extracted Words (Before Matching)", extractedWordsImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    print("creating template")
    template = tempOp.createTemplate(word=word ,fontSize=averageHeight_Pixel)
    if template is None:
        return jsonify({"error": "Failed to create template for the word. Check font paths or word content."}), 500

    print("template has been created \n template matching images")

    # Pass matching_mode to the templateMatching function
    foundWords = tempOp.templateMatching(image=image , template=template , wordsProperty=wordsProperty, matching_mode="auto")

    print(foundWords) # This line prints the NumPy types

    # --- View 5: Final Detected Words ---
    finalImage = originalImage.copy() # Use original color image for drawing
    if len(foundWords) > 0:
        finalImage = tempOp.putRectangles(image=finalImage, wordProperty=foundWords)
    else:
        cv2.putText(img=finalImage , text="No match found",
                    org=(20,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=(0,255,0),
                    thickness=2)
    print(foundWords) # This line prints the NumPy types again
    cv2.imshow("5. Final Detected Words", finalImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert final image to base64 for API response
    _, buffer = cv2.imencode('.png', finalImage)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    # --- START OF PREVIOUS CHANGES (np.int32 to int conversion) ---
    cleaned_found_words = []
    for box_pair in foundWords:
        # Each box_pair is expected to be ((x1, y1), (x2, y2))
        x1, y1 = int(box_pair[0][0]), int(box_pair[0][1])
        x2, y2 = int(box_pair[1][0]), int(box_pair[1][1])
        cleaned_found_words.append(((x1, y1), (x2, y2)))
    # --- END OF PREVIOUS CHANGES ---

    # Use the cleaned_found_words list and the newly defined unique_filename
    return jsonify({
        "message": "Image processed successfully",
        "fileName": unique_filename, 
        "foundWords": cleaned_found_words,
        "image_base64": base64_img
    }), 200


@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)