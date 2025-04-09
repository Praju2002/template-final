import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ==== Preprocessing ====
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 10)

# ==== Generate Template ====
def generate_word_image(word, font_path="C:/Windows/Fonts/arial.ttf", font_size=36):
    font = ImageFont.truetype(font_path, font_size)
    dummy = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), word, font=font)
    size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    
    img = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), word, fill=0, font=font)
    
    word_np = np.array(img)
    _, binarized = cv2.threshold(word_np, 127, 255, cv2.THRESH_BINARY_INV)
    return binarized

# ==== Template Matching ====
def match_word_in_image(image, word_template, threshold=0.5):
    h_img, w_img = image.shape
    h_temp, w_temp = word_template.shape
    best_match = None
    best_val = -1
    best_scale = 1.0

    for scale in np.linspace(0.6, 1.4, 10):
        resized = cv2.resize(word_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if resized.shape[0] > h_img or resized.shape[1] > w_img:
            continue

        result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_match = (max_loc, resized.shape[::-1])
            best_scale = scale

    return best_match, best_val

# ==== API Endpoint ====
@app.route("/upload", methods=["POST"])
def upload():
    image_file = request.files["image"]
    word = request.form.get("word")

    if not image_file or not word:
        return jsonify({"error": "Missing image or word"}), 400

    img_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.png")
    image_file.save(img_path)
    
    img = cv2.imread(img_path)
    pre_img = preprocess(img)
    word_img = generate_word_image(word)
    
    match, score = match_word_in_image(pre_img, word_img)

    if match and score > 0.5:
        top_left = match[0]
        w, h = match[1]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(img, f"{word} ({score:.2f})", (top_left[0], top_left[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(img, "No match found", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    result_name = f"result_{uuid.uuid4().hex}.png"
    result_path = os.path.join(RESULT_FOLDER, result_name)
    cv2.imwrite(result_path, img)

    return jsonify({"processed_image_url": f"/results/{result_name}"})


@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
