from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Load image
    img = cv2.imread(filepath)

    # Placeholder for template matching (you can replace with your logic)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_filename = f"processed_{filename}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    cv2.imwrite(processed_path, gray)  # Save processed image

    return jsonify({
        'message': 'Image processed successfully',
        'processed_image_url': f"http://127.0.0.1:5000/processed/{processed_filename}"
    })

@app.route('/processed/<filename>')
def get_processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
