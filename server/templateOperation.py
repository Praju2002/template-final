# File: templateOperation.py

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont  
import preProcessing as pp
import wordExtract as we

# --- SIFT Feature Extraction ---
def extract_sift_features(img: np.ndarray):
    """Extracts SIFT keypoints and descriptors from an image."""
    # Ensure image is grayscale and 8-bit unsigned for SIFT
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    img_gray = img_gray.astype(np.uint8) # Explicitly cast to uint8

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    return kp, des

# --- SIFT Feature Matching ---
def match_sift_features(des_template, des_candidate):
    """
    Matches SIFT descriptors using BFMatcher and Lowe's Ratio Test.
    des_template: Descriptors from template (query)
    des_candidate: Descriptors from candidate image (train)
    """
    if des_template is None or des_candidate is None:
        return [] # Return empty list if no descriptors

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des_template, des_candidate, k=2)

    good_matches = []
    # Apply Lowe's Ratio Test
    for pair in matches:
        # Ensure there are two matches (m and n) to unpack
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance: # Lower value for stricter matching (e.g., 0.7)
                good_matches.append(m)
    return good_matches

# --- Template Creation (from your existing code) ---
def createTemplate(word: str, fontSize: int):
    """
    Create a template image for the given word with specified font size.
    """
    def is_nepali(text):
        return any('\u0900' <= ch <= '\u097F' for ch in text)

    font_path = "./fonts/nepali.TTF" if is_nepali(word) else "./fonts/arial.TTF"
    print(f"[DEBUG] Font path being used: {font_path}")
    try:
        # Scale font size, and ensure a minimum size for SIFT to work effectively
        fontSize = max(20, np.round(fontSize * 1.5).astype(int)) 
        font = ImageFont.truetype(font_path, fontSize)
    except Exception as e:
        print(f"[ERROR] Failed to load font: {e}")
        return None

    try:
        dummy = Image.new("L", (1, 1), color=255)
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), word, font=font)

        # Adding padding based on font size for better template creation
        width = (bbox[2] - bbox[0] + 2 * fontSize)
        height = (bbox[3] - bbox[1] + fontSize)

        img = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(img)
        draw.text((width // 2, height // 2), word, fill="white", font=font, anchor="mm")
        
        wordImage = np.array(img)

        _, wordImage = cv2.threshold(wordImage, 127, 255, cv2.THRESH_BINARY)
    except Exception as e:
        print(f"[ERROR] Template image creation failed: {e}")
        return None

    if wordImage is None or wordImage.size == 0:
        print(f"[ERROR] Created word image is empty for word: {word}")
        return None

    averageHeight_Pixel, tallestText = pp.averageHeightOfLetters(image=wordImage)
    
    if averageHeight_Pixel is None or tallestText is None: # Added check for None from preProcessing
        print(f"[ERROR] No lines detected in generated template for word: {word}")
        return None

    imageShape = wordImage.shape

    # Ensure tallestText indices are valid
    tallestText_top = max(0, min(tallestText[0], imageShape[0] - 1))
    tallestText_bottom = max(0, min(tallestText[1], imageShape[0])) # bottom is exclusive

    tallestLineImage = wordImage[tallestText_top:tallestText_bottom, 0:imageShape[1]]
    
    # Handle cases where tallestLineImage might be empty
    if tallestLineImage.size == 0:
        print("[WARNING] Tallest line image is empty in createTemplate. Cannot calculate average gap. Using default.")
        averageGap_Pixel = 1 # Default to a small gap to avoid division by zero or errors
    else:
        averageGap_Pixel = pp.averageGapOfLetter(image=tallestLineImage)
        if averageGap_Pixel is None: # Handle if gap calculation also failed
            print("[WARNING] averageGapOfLetter returned None in createTemplate. Using default.")
            averageGap_Pixel = 1

    iterationNumber = pp.requiredNumberOfIterations(averageGap=averageGap_Pixel)

    print(averageHeight_Pixel, averageGap_Pixel, iterationNumber)

    smudgedImage = pp.prepareImageForWordExtraction(image=wordImage, iteration=iterationNumber)

    wordsProperty = we.wordExtract(image=smudgedImage,
                                   averageHeight=averageHeight_Pixel,
                                   smudgedIteration=iterationNumber)
    
    # Ensure wordsProperty is not empty before popping
    if not wordsProperty:
        print(f"[ERROR] No word properties extracted from template for word: {word}")
        return None

    # pop() returns the last item, assumes there is at least one
    wordsProperty_single = wordsProperty.pop()

    left, top, right, bottom = wordPropertyTOdirectionConvertor(wordsProperty_single)

    # Ensure bounding box coordinates are valid
    img_height, img_width = wordImage.shape[:2]
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)

    if left >= right or top >= bottom:
        print(f"[ERROR] Invalid bounding box for extracted word from template: ({left}, {top}, {right}, {bottom})")
        return None

    wordImage = wordImage[top:bottom, left:right]

    # --- NEW CHECK: Ensure the final template image is not empty after all processing ---
    if wordImage.size == 0:
        print(f"[ERROR] Final template image is empty after word extraction and slicing for word: {word}")
        return None
    # --- END NEW CHECK ---

    return wordImage

# --- Helper for putting rectangles (from your existing code) ---
def wordPropertyTOdirectionConvertor(wordsProperty: tuple):
    """
    Convert wordsProperty from ((left, top), (right, bottom)) to (left, top, right, bottom).
    """
    return (wordsProperty[0][0], wordsProperty[0][1], wordsProperty[1][0], wordsProperty[1][1])

def putRectangles(image: np.ndarray, wordProperty: list):
    """
    Draw rectangles on the image for each word property.
    wordProperty is a list of ((left, top), (right, bottom)) tuples.
    """
    # Ensure image is mutable if it comes from base64 (might be read-only)
    if not image.flags['WRITEABLE']:
        image = image.copy()
        
    for leftTop_RightBottom in wordProperty:
        cv2.rectangle(image, pt1=leftTop_RightBottom[0], pt2=leftTop_RightBottom[1], color=(0, 255, 0), thickness=2) # Green rectangle, thickness 2
    return image


def dynamic_threshold(area: int) -> float:
    """
    Returns a template matching threshold based on bounding box area.
    Smaller area => lower threshold (more strict),
    Larger area => higher threshold (more lenient).
    """
    # Avoid zero or negative areas
    area = max(area, 1)

    # Define min/max thresholds
    min_thresh = 0.4
    max_thresh = 0.85

    # Use log-scaled range and normalize
    # Adjust constants if the curve needs to shift
    scale = (np.log10(area) - 1) / 2.0  # log10(10) -> 0; log10(1000) -> ~1.5

    # Clamp scale between 0 and 1
    scale = max(0.0, min(1.0, scale))

    return min_thresh + (max_thresh - min_thresh) * scale

def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to zero-mean and unit-variance.
    """
    img = img.astype(np.float32)
    mean = img.mean()
    std = img.std() if img.std() > 1e-5 else 1.0
    return (img - mean) / std

def _tm_sqdiff_normed_matching(image: np.ndarray, template: np.ndarray, wordsProperty: list[tuple]):
    """
    Performs template matching using TM_SQDIFF_NORMED.
    """
    foundWords = []
    if template is None or template.size == 0:
        print("[WARNING] Skipping TM_SQDIFF_NORMED matching: template is empty.")
        return []

    for wp in wordsProperty:
        left, top, right, bottom = wordPropertyTOdirectionConvertor(wp)
        width = right - left
        height = bottom - top
        area = width * height

        # Skip if bounding box is invalid
        if width <= 0 or height <= 0:
            continue

        try:
            scaledTemplate = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"[ERROR] Failed to resize template for TM_SQDIFF_NORMED: {e}")
            continue

        h_img, w_img = image.shape[:2]
        top_exp = max(0, top - 1)
        bottom_exp = min(h_img, bottom + 1)
        left_exp = max(0, left - 1)
        right_exp = min(w_img, right + 1)

        sectionOfImage = image[top_exp:bottom_exp, left_exp:right_exp]

        if sectionOfImage.shape[0] >= scaledTemplate.shape[0] and sectionOfImage.shape[1] >= scaledTemplate.shape[1]:
            similarityScore = cv2.matchTemplate(sectionOfImage, scaledTemplate, cv2.TM_SQDIFF_NORMED)
            min_val, _, _, _ = cv2.minMaxLoc(similarityScore)

            threshold = dynamic_threshold(area)
            # print(f"[DEBUG] TM_SQDIFF_NORMED: Area: {area}, Threshold: {threshold:.2f}, Score: {min_val:.2f}")

            if min_val < threshold:
                foundWords.append(wp)
        # else:
            # print(f"[SKIP] TM_SQDIFF_NORMED: Word property {wp} - section {sectionOfImage.shape}, template {scaledTemplate.shape}")

    return foundWords

# --- Main Template Matching Function with SIFT and Homography ---
def templateMatching(image: np.ndarray, template: np.ndarray, wordsProperty: list, matching_mode: str = "auto"):
    """
    Performs template matching based on the specified mode.
    'auto' attempts to determine if text is printed or varied.
    'TM_SQDIFF_NORMED' uses standard template matching.
    'SIFT' uses SIFT feature matching with homography.
    """
    MIN_MATCH_COUNT_SIFT = 7 # Minimum number of good matches for initial SIFT consideration (can be tuned)
    MIN_INLIER_COUNT = 4 # Minimum number of inliers required for a valid homography (typically 4 or more)

    if template is None or template.size == 0:
        print("[WARNING] Template is empty or invalid. Skipping all matching operations.")
        return []

    # Extract SIFT features from the template once
    kp_template, des_template = extract_sift_features(template)
    if des_template is None or len(kp_template) < MIN_MATCH_COUNT_SIFT:
        print(f"[WARN] Not enough SIFT features in template for robust matching (found {len(kp_template)}).")
        # If template itself doesn't have enough features, SIFT won't work well.
        # Fallback to TM_SQDIFF_NORMED if SIFT is requested but template is poor.
        if matching_mode == "SIFT":
            print("[INFO] Falling back to TM_SQDIFF_NORMED due to insufficient template SIFT features.")
            return _tm_sqdiff_normed_matching(image, template, wordsProperty)
        # For auto mode, it will try TM_SQDIFF_NORMED anyway
        # For other modes or if still no features after fallback, return empty.
        return []

    foundWords = []

    for wp in wordsProperty:
        left, top = wp[0]
        right, bottom = wp[1]

        # Ensure valid coordinates for slicing
        if right <= left or bottom <= top:
            print(f"[WARN] Invalid word property coordinates: {wp}. Skipping.")
            continue

        # Slice candidate image region from the main image
        candidate_img = image[top:bottom, left:right]
        
        if candidate_img is None or candidate_img.size == 0:
            print(f"[WARN] Candidate image region is empty for {wp}. Skipping.")
            continue

        # Extract SIFT features from candidate region
        kp_candidate, des_candidate = extract_sift_features(candidate_img)
        if des_candidate is None:
            print(f"[DEBUG] No descriptors in candidate region {wp}. Skipping.")
            continue

        # Match features between template and candidate
        good_matches = match_sift_features(des_template, des_candidate)

        # Check if enough initial good matches are found
        if len(good_matches) > MIN_MATCH_COUNT_SIFT:
            # Extract locations of matched keypoints for homography calculation
            src_pts = np.float32([ kp_template[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_candidate[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

            # Find Homography using RANSAC (RANdom SAmple Consensus)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 

            if M is not None and mask is not None:
                # Get the number of inliers (good matches that fit the homography)
                matches_mask = mask.ravel().tolist()
                num_inliers = sum(matches_mask) # Count the True values in the mask

                # Verify if there are enough inliers to consider it a valid match
                if num_inliers >= MIN_INLIER_COUNT:
                    print(f"[INFO] Match found with {num_inliers} inliers for word property {wp}")
                    
                    # Transform the corners of the template to get the detected object's bounding box in the candidate image
                    h,w = template.shape[:2] # Ensure we get height and width from the template
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts,M)

                    # Get the min/max coordinates of the transformed rectangle
                    # Add the offset of the candidate image to get coordinates relative to the main image
                    min_x = int(np.min(dst[:,0,0]) + left)
                    max_x = int(np.max(dst[:,0,0]) + left)
                    min_y = int(np.min(dst[:,0,1]) + top)
                    max_y = int(np.max(dst[:,0,1]) + top)

                    # Append the refined bounding box (relative to original image)
                    foundWords.append(((min_x, min_y), (max_x, max_y)))
                else:
                    print(f"[DEBUG] Not enough inliers ({num_inliers}) for {wp}.")
            else:
                print(f"[DEBUG] No robust homography found for {wp}.")
        else:
            print(f"[DEBUG] Not enough initial good matches ({len(good_matches)}) for {wp}.")

    # Fallback to TM_SQDIFF_NORMED if SIFT (with homography) didn't find matches in 'auto' mode
    if matching_mode == "auto" and not foundWords:
        print("[INFO] No matches found with SIFT (with homography), trying TM_SQDIFF_NORMED as fallback.")
        foundWords = _tm_sqdiff_normed_matching(image, template, wordsProperty)
        
    return foundWords