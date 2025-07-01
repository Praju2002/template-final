import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont  
import preProcessing as pp
import wordExtract as we

# Minimum number of good matches required for SIFT (used as simple count now)
MIN_MATCH_COUNT = 7 
def is_nepali(text):
        return any('\u0900' <= ch <= '\u097F' for ch in text)
def createTemplate(word: str, fontSize: int):
    """
    Create a template image for the given word with specified font size.
    """
     # Corrected 'ch' to 'text'
    word_is_nepali = is_nepali(word)

    font_path = "./fonts/nepali.TTF" if word_is_nepali else "./fonts/arial.TTF"
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

        # For debugging template generation
        # cv2.imshow("Generated Template", wordImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        _, wordImage = cv2.threshold(wordImage, 127, 255, cv2.THRESH_BINARY)
    except Exception as e:
        print(f"[ERROR] Template image creation failed: {e}")
        return None

    if wordImage is None or wordImage.size == 0:
        print(f"[ERROR] Created word image is empty for word: {word}")
        return None

    averageHeight_Pixel, tallestText = pp.averageHeightOfLetters(image=wordImage)
    imageShape = wordImage.shape

    # Ensure tallestText indices are valid
    tallestText_top = max(0, min(tallestText[0], imageShape[0] - 1))
    tallestText_bottom = max(0, min(tallestText[1], imageShape[0])) # bottom is exclusive

    tallestLineImage = wordImage[tallestText_top:tallestText_bottom, 0:imageShape[1]]
    
    # Handle cases where tallestLineImage might be empty
    if tallestLineImage.size == 0:
        print("[WARNING] Tallest line image is empty. Cannot calculate average gap.")
        averageGap_Pixel = 1 # Default to a small gap to avoid division by zero or errors
    else:
        averageGap_Pixel = pp.averageGapOfLetter(image=tallestLineImage)

    iterationNumber = pp.requiredNumberOfIterations(averageGap=averageGap_Pixel,is_nepali_text=word_is_nepali)

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


def paddingCalculation(x: int):
    """
    Calculates padding required for the search word.
    """
    return int(np.round(-0.0006503 * x**2 + 0.3229 * x + 0.7658))

# def dynamic_threshold(area: int) -> float:
#     """
#     Returns a template matching threshold based on bounding box area.
#     Smaller area => lower threshold (more strict),
#     Larger area => higher threshold (more lenient).
#     """
#     # Avoid zero or negative areas
#     area = max(area, 1)

#     # Define min/max thresholds
#     min_thresh = 0.4
#     max_thresh = 0.75

#     # Use log-scaled range and normalize
#     # Adjust constants if the curve needs to shift
#     scale = (np.log10(area) - 1) / 2.0  # log10(10) -> 0; log10(1000) -> ~1.5

#     # Clamp scale between 0 and 1
#     scale = max(0.0, min(1.0, scale))

#     return min_thresh + (max_thresh - min_thresh) * scale


def dynamic_threshold(height: int , width: int) -> float:
    """
    Returns a template matching threshold based on bounding box area.
    Smaller area => lower threshold (more strict),
    Larger area => higher threshold (more lenient).
    """
    # Avoid zero or negative areas

    numberOfCharacters = width / (0.65 *height)

    numberOfCharacters = max(1, numberOfCharacters)

    print("estimate number of char",numberOfCharacters)


    # Define min/max thresholds
    min_thresh = 0.4
    max_thresh = 0.75

    log_min = np.log10(1)
    log_max = np.log10(45)
    log_val = np.log10(numberOfCharacters)  # Clamp to avoid log(0)

    scale = (log_val - log_min) / (log_max - log_min)
    scale = max(0.0, min(1.0, scale))  # Clamp to [0, 1]

    # Use log-scaled range and normalize
    # Adjust constants if the curve needs to shift
    #scale = (np.log10(area) - 1) / 2.0  # log10(10) -> 0; log10(1000) -> ~1.5

    threshold = min_thresh + (max_thresh - min_thresh) * scale
    print("theshold ",threshold)

    return threshold


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
    
    heightOfTemplate , widthOfTemplate = template.shape
    threshold  = dynamic_threshold(heightOfTemplate,widthOfTemplate)

    for wp in wordsProperty:
        left, top, right, bottom = wordPropertyTOdirectionConvertor(wp)
        width = right - left
        height = bottom - top

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

           
            # print(f"[DEBUG] TM_SQDIFF_NORMED: Area: {area}, Threshold: {threshold:.2f}, Score: {min_val:.2f}")

            print("min value",min_val)    
            if min_val < threshold:
                foundWords.append(wp)
        # else:
            # print(f"[SKIP] TM_SQDIFF_NORMED: Word property {wp} - section {sectionOfImage.shape}, template {scaledTemplate.shape}")

    return foundWords

def _sift_template_matching(image: np.ndarray, template: np.ndarray, wordsProperty: list[tuple]):
    """
    Performs template matching using SIFT features without homography.
    """
    foundWords = []

    if template is None or template.size == 0:
        print("[WARNING] Skipping SIFT matching: template is empty.")
        return []

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher() # Brute-Force Matcher

    # Convert template to grayscale if it's not already and ensure CV_8U
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
    template_gray = template_gray.astype(np.uint8) # Explicitly cast to uint8

    # Extract SIFT features from the template once
    kp_template, des_template = sift.detectAndCompute(template_gray, None)
    if des_template is None or len(kp_template) < MIN_MATCH_COUNT: # Ensure enough keypoints in template
        print(f"[WARNING] Not enough descriptors found in template ({len(kp_template)}). Cannot perform SIFT matching.")
        return []

    for wp in wordsProperty:
        left, top, right, bottom = wordPropertyTOdirectionConvertor(wp)
        
        # Ensure candidate image slice dimensions are valid before slicing
        if bottom <= top or right <= left:
            continue

        candidate_img = image[top:bottom, left:right]

        # Explicitly check if the sliced candidate image is empty
        if candidate_img.size == 0:
            continue

        # Convert candidate_img to grayscale if it's not already and ensure CV_8U
        if len(candidate_img.shape) == 3:
            candidate_img_gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
        else:
            candidate_img_gray = candidate_img
        candidate_img_gray = candidate_img_gray.astype(np.uint8) # Explicitly cast to uint8

        kp_candidate, des_candidate = sift.detectAndCompute(candidate_img_gray, None)

        if des_candidate is None or len(kp_candidate) < MIN_MATCH_COUNT: # Ensure enough keypoints in candidate
            continue

        try:
            matches = bf.knnMatch(des_template, des_candidate, k=2)
        except cv2.error as e:
            # This can happen if one of the descriptor sets is empty after filtering
            continue

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance: # Ratio test as per Lowe's paper
                good_matches.append(m)

        # Decision based solely on the number of good matches (no homography)
        if len(good_matches) >= MIN_MATCH_COUNT: # Use MIN_MATCH_COUNT for simple good match count
            foundWords.append(wp)
            # print(f"[DEBUG] SIFT Match Found for {wp} with {len(good_matches)} good matches.")

    return foundWords

def templateMatching(image: np.ndarray, template: np.ndarray, wordsProperty: list[tuple], matching_mode: str = "auto"):
    """
    Performs template matching based on the specified mode.
    'auto' attempts to determine if text is printed or varied.
    'TM_SQDIFF_NORMED' uses standard template matching.
    'SIFT' uses SIFT feature matching with homography.
    """
    if template is None or template.size == 0:
        print("[WARNING] Template is empty. Skipping all matching operations.")
        return []

    if matching_mode == "TM_SQDIFF_NORMED":
        print("[INFO] Using TM_SQDIFF_NORMED for template matching.")
        return _tm_sqdiff_normed_matching(image, template, wordsProperty)
    elif matching_mode == "SIFT":
        print("[INFO] Using SIFT for template matching.")
        return _sift_template_matching(image, template, wordsProperty)
    elif matching_mode == "auto":
        print("[INFO] Auto mode: Attempting TM_SQDIFF_NORMED first.")
        found = _tm_sqdiff_normed_matching(image, template, wordsProperty)
        
        if not found:
            print("[INFO] No matches found with TM_SQDIFF_NORMED, trying SIFT.")
            found = _sift_template_matching(image, template, wordsProperty)
        
        return found
    else:
        print(f"[ERROR] Invalid matching_mode: {matching_mode}. Defaulting to TM_SQDIFF_NORMED.")
        return _tm_sqdiff_normed_matching(image, template, wordsProperty)


def wordPropertyTOdirectionConvertor(wordsProperty: tuple):
    """
    Convert wordsProperty from ((left, top), (right, bottom)) to (left, top, right, bottom).
    """
    return (wordsProperty[0][0], wordsProperty[0][1], wordsProperty[1][0], wordsProperty[1][1])


def putRectangles(image: np.ndarray, wordProperty: list):
    """
    Draw rectangles on the image for each word property.
    """
    # Ensure image is mutable if it comes from base64 (might be read-only)
    if not image.flags['WRITEABLE']:
        image = image.copy()
        
    for leftTop_RightBottom in wordProperty:
        cv2.rectangle(image, pt1=leftTop_RightBottom[0], pt2=leftTop_RightBottom[1], color=(0, 255, 0), thickness=1)

    return image