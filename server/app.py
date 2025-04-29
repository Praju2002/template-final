import os
import uuid
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ==== Optimized Preprocessing for Printed Text ====
def preprocess_document(img):
    """
    Optimized preprocessing for printed text documents
    Returns both grayscale and binary versions
    """
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Try different preprocessing approaches and select the best one for text detection
    
    # Approach 1: Basic adaptive threshold
    binary1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 5
    )
    
    # Approach 2: More aggressive with blur first
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary2 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, 15, 7
    )
    
    # Approach 3: Global Otsu thresholding
    _, binary3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Save all versions for debugging
    cv2.imwrite('binary1.png', binary1)
    cv2.imwrite('binary2.png', binary2)
    cv2.imwrite('binary3.png', binary3)
    
    # Use binary1 as the default, but you could implement logic to choose the best
    binary = binary1
    
    return gray, binary1

# ==== Template Generation ====
def generate_word_template(word, font_path="C:/Windows/Fonts/arial.ttf", font_size=36):
    """
    Generate a template for the target word
    Returns both the template image and the bounding box
    """
    try:
        # Try to load the specified font
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, IOError):
        try:
            # Try system fonts that might be available across platforms
            system_fonts = [
                "Arial.ttf", "arial.ttf",
                "Times.ttf", "times.ttf",
                "TimesNewRoman.ttf", "Courier.ttf",
                "DejaVuSans.ttf", "LiberationSans-Regular.ttf"
            ]
            
            font_found = False
            for sys_font in system_fonts:
                try:
                    # Check if font exists in various common directories
                    common_dirs = [
                        "C:/Windows/Fonts/", 
                        "/usr/share/fonts/truetype/", 
                        "/System/Library/Fonts/",
                        "/Library/Fonts/"
                    ]
                    
                    for directory in common_dirs:
                        potential_path = os.path.join(directory, sys_font)
                        if os.path.exists(potential_path):
                            font = ImageFont.truetype(potential_path, font_size)
                            font_found = True
                            print(f"Using font: {potential_path}")
                            break
                    
                    if font_found:
                        break
                except:
                    continue
            
            if not font_found:
                # Fall back to default font if no specified fonts are available
                print(f"No system fonts found, using default")
                font = ImageFont.load_default()
        except:
            # Ultimate fallback
            print(f"Font loading failed, using default")
            font = ImageFont.load_default()
    
    # Create a dummy image to calculate text dimensions
    dummy = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), word, font=font)
    
    # Calculate size with padding
    width = max(bbox[2] - bbox[0] + 20, 50)  # Add padding, minimum width
    height = max(bbox[3] - bbox[1] + 20, 50)  # Add padding, minimum height
    
    # Create actual image with proper size
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    # Draw text centered
    draw.text((10, 10), word, fill=0, font=font)
    
    # Convert to numpy array
    word_np = np.array(img)
    
    # Binarize
    _, binary = cv2.threshold(word_np, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Save template for debugging if needed
    # cv2.imwrite(f"template_{word}.png", binary)
    
    return binary

# ==== Multi-scale Template Matching ====
def multi_scale_template_matching(image, template, scale_range=(0.6, 1.1), scale_steps=10, threshold=0.4):
    """
    Perform template matching at multiple scales to account for font size variations
    Uses a wider scale range and lower threshold for better detection
    """
    h_img, w_img = image.shape
    h_temp, w_temp = template.shape
    
    # Debug visualizations - uncomment to save intermediate results
    # cv2.imwrite('debug_document.png', image)
    # cv2.imwrite('debug_template.png', template)
    
    matches = []
    
    # Try different scales
    for scale in np.linspace(scale_range[0], scale_range[1], scale_steps):
        # Resize template
        width = int(template.shape[1] * scale)
        height = int(template.shape[0] * scale)
        
        if width <= 0 or height <= 0 or width >= w_img or height >= h_img:
            continue
            
        resized_template = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        
        # Try multiple template matching methods
        methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
        
        for method in methods:
            # Match template
            result = cv2.matchTemplate(image, resized_template, method)
            
            # Find locations above threshold
            locations = np.where(result >= threshold)
            
            # Add matches to list
            for pt in zip(*locations[::-1]):  # Switch columns and rows
                matches.append({
                    'top_left': pt,
                    'bottom_right': (pt[0] + width, pt[1] + height),
                    'confidence': float(result[pt[1], pt[0]]),
                    'scale': float(scale),
                    'method': method
                })
    
    # Apply non-maximum suppression to remove overlapping matches
    return non_max_suppression(matches)

def non_max_suppression(matches, overlap_threshold=0.3):
    """
    Remove overlapping matches, keeping only the best one
    """
    if not matches:
        return []
    
    # Sort by confidence
    matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    for match in matches:
        should_keep = True
        
        for kept_match in keep:
            # Calculate overlap
            x1 = max(match['top_left'][0], kept_match['top_left'][0])
            y1 = max(match['top_left'][1], kept_match['top_left'][1])
            x2 = min(match['bottom_right'][0], kept_match['bottom_right'][0])
            y2 = min(match['bottom_right'][1], kept_match['bottom_right'][1])
            
            overlap_width = max(0, x2 - x1)
            overlap_height = max(0, y2 - y1)
            overlap_area = overlap_width * overlap_height
            
            match_area = (match['bottom_right'][0] - match['top_left'][0]) * (match['bottom_right'][1] - match['top_left'][1])
            
            if match_area > 0 and overlap_area / match_area > overlap_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep.append(match)
    
    return keep

# ==== Draw Results ====
def draw_matches(image, matches, word):
    """
    Draw bounding boxes and labels around matches
    """
    result = image.copy()
    
    # Convert to color if grayscale
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    # Check if we have any matches
    if not matches:
        # Draw a message indicating no matches found
        cv2.putText(
            result,
            f"No matches found for '{word}'",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),  # Red color
            2,
            cv2.LINE_AA
        )
        return result
    
    # Draw matches
    for match in matches:
        # Draw rectangle
        cv2.rectangle(
            result, 
            match['top_left'], 
            match['bottom_right'], 
            (0, 255, 0),  # Green color
            2
        )
        
        # Draw label with confidence
        label = f"{word} ({match['confidence']:.2f})"
        cv2.putText(
            result,
            label,
            (match['top_left'][0], match['top_left'][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),  # Green color
            1,
            cv2.LINE_AA
        )
    
    return result

# ==== API Endpoint ====
@app.route("/upload", methods=["POST"])
@app.route("/upload", methods=["POST"])
def upload():
    try:
        # Get image and word from request
        if 'image' not in request.files:
            return jsonify({"error": "No image file in request"}), 400

        image_file = request.files["image"]
        if image_file.filename == '':
            return jsonify({"error": "Empty image filename"}), 400

        word = request.form.get("word")
        if not word:
            return jsonify({"error": "Missing word parameter"}), 400

        print(f"Processing image: {image_file.filename}, searching for word: '{word}'")

        # Read image with additional error handling
        img_array = np.frombuffer(image_file.read(), np.uint8)
        if len(img_array) == 0:
            return jsonify({"error": "Empty image file"}), 400

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        print(f"Image loaded successfully. Shape: {img.shape}")

        # Preprocess image
        gray, binary = preprocess_document(img)
        print(f"Preprocessing completed. Binary shape: {binary.shape}")

        # Generate template
        template = generate_word_template(word)
        print(f"Template generated. Shape: {template.shape}")

        # Find matches
        matches = multi_scale_template_matching(binary, template)
        print(f"Found {len(matches)} matches")

        # Convert matches to JSON-serializable format
        serializable_matches = []
        for match in matches:
            serializable_matches.append({
                'top_left': [int(x) for x in match['top_left']],
                'bottom_right': [int(x) for x in match['bottom_right']],
                'confidence': float(match['confidence']),
                'scale': float(match['scale'])
            })

        # Print match details (moved outside the serializable_matches loop)
        print("Matches found:")
        for match in matches:
            print(f"  Top Left: {match['top_left']}, Confidence: {match['confidence']}, Scale: {match['scale']}")

        # Draw matches on the original image
        result = draw_matches(img, matches, word)

        # Save results
        result_filename = f"{uuid.uuid4()}.png"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, result)
        print(f"Result saved to {result_path}")

        # Convert final image to base64
        _, buffer = cv2.imencode('.png', result)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "image_base64": base64_img,
            "matches_count": len(matches),
            "matches": serializable_matches
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in upload endpoint: {str(e)}")
        print(error_details)
        return jsonify({
            "error": str(e),
            "details": error_details
        }), 500

        
        # Draw matches on the original image
        result = draw_matches(img, matches, word)
        
        # Save results
        result_filename = f"{uuid.uuid4()}.png"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, result)
        print(f"Result saved to {result_path}")
        
        # Convert final image to base64
        _, buffer = cv2.imencode('.png', result)
        base64_img = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "image_base64": base64_img,
            "matches_count": len(matches),
            "matches": serializable_matches
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in upload endpoint: {str(e)}")
        print(error_details)
        return jsonify({
            "error": str(e),
            "details": error_details
        }), 500

# ==== Debugging/Testing Routes ====
@app.route("/test_preprocess", methods=["POST"])
def test_preprocess():
    """For debugging preprocessing steps"""
    try:
        image_file = request.files["image"]
        
        if not image_file:
            return jsonify({"error": "Missing image"}), 400
        
        # Read image
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Preprocess image
        gray, binary = preprocess_document(img)
        
        # Convert to base64
        _, gray_buffer = cv2.imencode('.png', gray)
        gray_base64 = base64.b64encode(gray_buffer).decode('utf-8')
        
        _, binary_buffer = cv2.imencode('.png', binary)
        binary_base64 = base64.b64encode(binary_buffer).decode('utf-8')
        
        return jsonify({
            "gray_image": gray_base64,
            "binary_image": binary_base64
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test_template", methods=["POST"])
def test_template():
    """For debugging template generation"""
    try:
        word = request.form.get("word")
        
        if not word:
            return jsonify({"error": "Missing word"}), 400
        
        # Generate template
        template = generate_word_template(word)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', template)
        template_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "template_image": template_base64,
            "width": template.shape[1],
            "height": template.shape[0]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)