import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont 
import preProcessing as pp
import wordExtract as we 

# Minimum number of good matches required for SIFT (general threshold)
MIN_MATCH_COUNT = 7 

def is_nepali(text):
    """
    Checks if a given string contains Nepali Unicode characters.
    Nepali Unicode block ranges from U+0900 to U+097F.
    """
    return any('\u0900' <= ch <= '\u097F' for ch in text)

def createTemplate(word: str, fontSize: int):
    """
    Create a template image for the given word with specified font size.
    """
    
    word_is_nepali = is_nepali(word)

    font_path = "./fonts/nepali.TTF" if word_is_nepali else "./fonts/arial.TTF"
    print(f"[DEBUG] Font path being used: {font_path}")
    try:
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

    tallestText_top = max(0, min(tallestText[0], imageShape[0] - 1))
    tallestText_bottom = max(0, min(tallestText[1], imageShape[0])) 

    tallestLineImage = wordImage[tallestText_top:tallestText_bottom, 0:imageShape[1]]
    
   
    if tallestLineImage.size == 0:
        print("[WARNING] Tallest line image is empty. Cannot calculate average gap.")
        averageGap_Pixel = 1
    else:
        averageGap_Pixel = pp.averageGapOfLetter(image=tallestLineImage)

    iterationNumber = pp.requiredNumberOfIterations(averageGap=averageGap_Pixel,is_nepali_text=word_is_nepali)

    print(averageHeight_Pixel, averageGap_Pixel, iterationNumber)

    smudgedImage = pp.prepareImageForWordExtraction(image=wordImage, iteration=iterationNumber)

    wordsProperty = we.wordExtract(image=smudgedImage,
                                  averageHeight=averageHeight_Pixel,
                                  smudgedIteration=iterationNumber)
    
   
    if not wordsProperty:
        print(f"[ERROR] No word properties extracted from template for word: {word}")
        return None

    
    wordsProperty_single = wordsProperty.pop()

    left, top, right, bottom = wordPropertyTOdirectionConvertor(wordsProperty_single)

  
    img_height, img_width = wordImage.shape[:2]
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)

    if left >= right or top >= bottom:
        print(f"[ERROR] Invalid bounding box for extracted word from template: ({left}, {top}, {right}, {bottom})")
        return None

    wordImage = wordImage[top:bottom, left:right]

    if wordImage.size == 0:
        print(f"[ERROR] Final template image is empty after word extraction and slicing for word: {word}")
        return None
    

    return wordImage


def paddingCalculation(x: int):
    """
    Calculates padding required for the search word.
    """
    return int(np.round(-0.0006503 * x**2 + 0.3229 * x + 0.7658))

def dynamic_threshold(area: int) -> float:
    """
    Returns a template matching threshold based on bounding box area.
    Smaller area => lower threshold (more strict),
    Larger area => higher threshold (more lenient).
    """
    area = max(area, 1)

    min_thresh = 0.35
    max_thresh = 0.5

    scale = (np.log10(area) - 1) / 2.0  

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


def _sift_template_matching(image: np.ndarray, template: np.ndarray, wordsProperty: list[tuple]):
    """
    Performs SIFT feature matching between template and candidate word regions.
    Optimized for text matching with reduced false positives.
    """
    foundWords = []

    if template is None or template.size == 0:
        print("[WARNING] Skipping SIFT matching: template is empty.")
        return []

    sift = cv2.SIFT_create(
        nfeatures=0,            
        contrastThreshold=0.04, 
        edgeThreshold=10,       
        sigma=1.6
    )
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template.copy()
    template_gray = template_gray.astype(np.uint8)

    template_gray = cv2.equalizeHist(template_gray)
  
    template_height, template_width = template_gray.shape
    if template_height < 25 or template_width < 50:
        scale_factor = 2.0
    else:
        scale_factor = 1.5
        
    if scale_factor != 1.0:
        template_gray = cv2.resize(template_gray, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    kp_template, des_template = sift.detectAndCompute(template_gray, None)

    if des_template is None or len(kp_template) < 6:  
        print(f"[WARNING] Not enough keypoints found in template ({len(kp_template) if kp_template else 0}). SIFT matching skipped.")
        return []

    print(f"[DEBUG] Template keypoints: {len(kp_template)}")

    match_count = 0
    template_area = template_height * template_width
    
    candidates_with_scores = []
    
    for wp in wordsProperty:
        left, top, right, bottom = wordPropertyTOdirectionConvertor(wp)

        if right <= left or bottom <= top:
            continue

        candidate_img = image[top:bottom, left:right]
        if candidate_img.size == 0:
            continue
        width = right - left
        height = bottom - top
        if width < 15 or height < 10:
            continue

        candidate_area = width * height
        size_ratio = candidate_area / template_area if template_area > 0 else 0
        if size_ratio < 0.3 or size_ratio > 3.0:  
            continue

        if len(candidate_img.shape) == 3:
            candidate_gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
        else:
            candidate_gray = candidate_img.copy()
        candidate_gray = candidate_gray.astype(np.uint8)

        candidate_gray = cv2.equalizeHist(candidate_gray)
        if scale_factor != 1.0:
            candidate_gray = cv2.resize(candidate_gray, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        kp_candidate, des_candidate = sift.detectAndCompute(candidate_gray, None)
        
        if des_candidate is None or len(kp_candidate) < 6:  
            continue

        try:
            matches = flann.knnMatch(des_template, des_candidate, k=2)
        except (cv2.error, ValueError):
            try:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des_template, des_candidate, k=2)
            except:
                continue

        valid_matches = [m for m in matches if len(m) == 2]
        if len(valid_matches) < 8: 
            continue

        good_matches = []
        for m, n in valid_matches:
            if m.distance < 0.65 * n.distance: 
                good_matches.append(m)

        if len(good_matches) < 10:  
            continue

        match_ratio = len(good_matches) / len(kp_template)
        candidate_coverage = len(good_matches) / len(kp_candidate)
        
        
        if len(kp_template) > 100:  
            min_ratio = 0.10    
            min_coverage = 0.12  
        elif len(kp_template) > 50: 
            min_ratio = 0.15
            min_coverage = 0.15
        else:  
            min_ratio = 0.25
            min_coverage = 0.20
        
        print(f"[DEBUG] Candidate {wp}: {len(good_matches)} matches, ratio: {match_ratio:.3f}, coverage: {candidate_coverage:.3f}, min_ratio: {min_ratio:.3f}")
        
        if (len(good_matches) >= 10 and 
            match_ratio >= min_ratio and 
            candidate_coverage >= min_coverage):
            
            quality_score = (match_ratio * 0.4 + 
                           candidate_coverage * 0.3 + 
                           len(good_matches) / 30.0 * 0.3) 
            
            candidates_with_scores.append((wp, len(good_matches), quality_score, match_ratio, candidate_coverage))

    candidates_with_scores.sort(key=lambda x: x[2], reverse=True)
    
    max_matches = min(5, len(candidates_with_scores))
    
    for i in range(max_matches):
        wp, good_match_count, score, ratio, coverage = candidates_with_scores[i]
        foundWords.append(wp)
        match_count += 1
        print(f"[MATCH] SIFT Match #{i+1} found at {wp} with {good_match_count} good matches (score: {score:.3f})")

    print(f"[DEBUG] SIFT found {match_count} matches out of {len(wordsProperty)} candidates")
    return foundWords

def preprocess_text_for_sift_minimal(img: np.ndarray) -> np.ndarray:
    """
    Very minimal preprocessing for text SIFT - just histogram equalization.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return cv2.equalizeHist(img)

def templateMatching(image: np.ndarray, template: np.ndarray, wordsProperty: list[tuple], matching_mode: str = "auto"):
    """
    Performs template matching using the specified mode.
    'auto' attempts to determine if text is printed or varied and applies appropriate method.
    'TM_SQDIFF_NORMED' uses standard template matching.
    'SIFT' uses SIFT feature matching.
    Expects wordsProperty as a list of ((left, top), (right, bottom)) tuples.
    """
    if template is None or template.size == 0:
        print("[WARNING] Template is empty. Skipping all matching operations.")
        return []
    
    if image is None or image.size == 0:
        print("[WARNING] Image for matching is empty. Skipping all matching operations.")
        return []

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.uint8)

    if matching_mode == "TM_SQDIFF_NORMED":
        print("[INFO] Using TM_SQDIFF_NORMED for template matching.")
        return _tm_sqdiff_normed_matching(image, template, wordsProperty)
    elif matching_mode == "TM_CCOEFF_NORMED":
        print("[INFO] Using TM_CCOEFF_NORMED for template matching.")
        return _tm_ccoeff_normed_matching(image, template, wordsProperty)
    elif matching_mode == "SIFT":
        print("[INFO] Using SIFT for template matching.")
        return _sift_template_matching(image, template, wordsProperty)
    elif matching_mode == "auto":
        print("[INFO] Auto mode: Attempting TM_CCOEFF_NORMED first.")
        found = _tm_ccoeff_normed_matching(image, template, wordsProperty)
        if not found:
            print("[INFO] No matches found with TM_SQDIFF_NORMED, trying TM_CCOEFF_NORMED.")
            found = _tm_sqdiff_normed_matching(image, template, wordsProperty)
        if not found:
            print("[INFO] No matches found with TM_CCOEFF_NORMED, trying SIFT.")
            found = _sift_template_matching(image, template, wordsProperty)
        return found
    else:
        print(f"[ERROR] Invalid matching_mode: {matching_mode}. Defaulting to TM_SQDIFF_NORMED.")
        return _tm_ccoeff_normed_matching(image, template, wordsProperty)


def wordPropertyTOdirectionConvertor(wordsProperty: tuple):
    """
    Convert a wordProperty tuple from ((left, top), (right, bottom)) to (left, top, right, bottom).
    This function ensures consistent unpacking throughout.
    """
    return (wordsProperty[0][0], wordsProperty[0][1], wordsProperty[1][0], wordsProperty[1][1])


def putRectangles(image: np.ndarray, wordProperty: list):
    """
    Draw rectangles on the image for each word property.
    Expects wordProperty as a list of ((left, top), (right, bottom)) tuples.
    """
    if not image.flags['WRITEABLE']:
        image = image.copy()
        
    for leftTop_RightBottom in wordProperty:
        cv2.rectangle(image, pt1=leftTop_RightBottom[0], pt2=leftTop_RightBottom[1], color=(0, 255, 0), thickness=1)

    return image

def _tm_ccoeff_normed_matching(image: np.ndarray, template: np.ndarray, wordsProperty: list[tuple]):
    """
    Performs template matching using TM_CCOEFF_NORMED.
    Returns word regions where the normalized correlation coefficient is high.
    """
    foundWords = []
    if template is None or template.size == 0:
        print("[WARNING] Skipping TM_CCOEFF_NORMED matching: template is empty.")
        return []

    for wp in wordsProperty:
        left, top, right, bottom = wordPropertyTOdirectionConvertor(wp)
        width = right - left
        height = bottom - top
        area = width * height

        if width <= 0 or height <= 0:
            continue

        try:
            scaledTemplate = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"[ERROR] Failed to resize template for TM_CCOEFF_NORMED: {e}")
            continue

        h_img, w_img = image.shape[:2]
        top_exp = max(0, top - 1)
        bottom_exp = min(h_img, bottom + 1)
        left_exp = max(0, left - 1)
        right_exp = min(w_img, right + 1)

        sectionOfImage = image[top_exp:bottom_exp, left_exp:right_exp]

        if sectionOfImage.shape[0] >= scaledTemplate.shape[0] and sectionOfImage.shape[1] >= scaledTemplate.shape[1]:
            similarityScore = cv2.matchTemplate(sectionOfImage, scaledTemplate, cv2.TM_CCOEFF_NORMED)
            max_val = cv2.minMaxLoc(similarityScore)[1]

            threshold = 0.63
            if max_val > threshold:
                foundWords.append(wp)

    return foundWords