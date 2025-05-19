import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont  
import preProcessing as pp
import wordExtract as we


def createTemplate(word: str, fontSize: int):
    """
    Create a template image for the given word with specified font size.
    """
    def is_nepali(text):
        return any('\u0900' <= ch <= '\u097F' for ch in text)

    font_path = "./fonts/nepali.TTF" if is_nepali(word) else "./fonts/arial.TTF"
    print(f"[DEBUG] Font path being used: {font_path}")
    try:
        fontSize = np.round(fontSize * 1.5).astype(int)
        font = ImageFont.truetype(font_path, fontSize)
    except Exception as e:
        print(f"[ERROR] Failed to load font: {e}")
        return None

    try:
        dummy = Image.new("L", (1, 1), color=255)
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), word, font=font)

        width = (bbox[2] - bbox[0] +2* fontSize)
        height = (bbox[3] - bbox[1] + fontSize)

        img = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(img)
        draw.text((width // 2, height // 2), word, fill="white", font=font, anchor="mm")
        
        wordImage = np.array(img)

        cv2.imshow("Generated Template", wordImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        _, wordImage = cv2.threshold(wordImage, 127, 255, cv2.THRESH_BINARY)
    except Exception as e:
        print(f"[ERROR] Template image creation failed: {e}")
        return None

    if wordImage is None or wordImage.size == 0:
        print(f"[ERROR] Created word image is empty for word: {word}")
        return None

    averageHeight_Pixel, tallestText = pp.averageHeightOfLetters(image=wordImage)
    imageShape = wordImage.shape

    tallestLineImage = wordImage[tallestText[0]:tallestText[1], 0:imageShape[1]]
    averageGap_Pixel = pp.averageGapOfLetter(image=tallestLineImage)

    iterationNumber = pp.requiredNumberOfIterations(averageGap=averageGap_Pixel)

    print(averageHeight_Pixel, averageGap_Pixel, iterationNumber)

    smudgedImage = pp.prepareImageForWordExtraction(image=wordImage, iteration=iterationNumber)

    wordsProperty = we.wordExtract(image=smudgedImage,
                                  averageHeight=averageHeight_Pixel,
                                  smudgedIteration=iterationNumber)
    
    # pop() returns the last item, assumes there is at least one
    wordsProperty = wordsProperty.pop()

    left, top, right, bottom = wordPropertyTOdirectionConvertor(wordsProperty)

    wordImage = wordImage[top:bottom, left:right]

    return wordImage


def paddingCalculation(x: int):
    """
    Calculates padding required for the search word.
    """
    return int(np.round(-0.0006503 * x**2 + 0.3229 * x + 0.7658))


# def templateMatching(image: np.ndarray, template: np.ndarray, wordsProperty: list[tuple]):
#     """
#     Perform template matching for all word properties on the image using the template.
#     Returns list of found word properties matching the template.
#     """

#     foundWords = []

#     if template is None or template.size == 0:
#         print("[WARNING] Skipping template matching: template is empty.")
#         return []

#     for wp in wordsProperty:
#         left, top, right, bottom = wordPropertyTOdirectionConvertor(wp)

#         # Resize template to fit word bounding box size
#         try:
#             scaledTemplate = cv2.resize(template, (right - left, bottom - top), interpolation=cv2.INTER_AREA)
#         except Exception as e:
#             print(f"[ERROR] Failed to resize template: {e}")
#             continue

#         # Expand bounding box by 1 pixel to avoid border issues
#         h, w = image.shape[:2]
#         top = max(0, top - 1)
#         bottom = min(h, bottom + 1)
#         left = max(0, left - 1)
#         right = min(w, right + 1)

#         sectionOfImage = image[top:bottom, left:right]

#         # Debug display to visually verify template and image section
#         cv2.imshow("Template", scaledTemplate)
#         cv2.imshow("Document Section", sectionOfImage)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         # Perform matching only if section is large enough
#         if sectionOfImage.shape[0] >= scaledTemplate.shape[0] and sectionOfImage.shape[1] >= scaledTemplate.shape[1]:
#             similarityScore = cv2.matchTemplate(sectionOfImage, scaledTemplate, cv2.TM_SQDIFF_NORMED)
#             min_val, _, _, _ = cv2.minMaxLoc(similarityScore)
#             print(f"[DEBUG] Similarity score : {min_val}")
#             if min_val < 1:
#                 foundWords.append(wp)
#         else:
#             print(f"[SKIP] Word property {wp} - section {sectionOfImage.shape}, template {scaledTemplate.shape}")

#     return foundWords
def extract_sift_features(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_sift_features(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

def templateMatching(image: np.ndarray, template: np.ndarray, wordsProperty: list[tuple]):
    foundWords = []

    if template is None or template.size == 0:
        print("[WARNING] Skipping template matching: template is empty.")
        return []

    # Extract SIFT features from the template once
    kp_template, des_template = extract_sift_features(template)
    if des_template is None:
        print("[WARNING] No descriptors found in template.")
        return []

    for wp in wordsProperty:
        left, top, right, bottom = wordPropertyTOdirectionConvertor(wp)

        # Extract candidate word image region
        candidate_img = image[top:bottom, left:right]

        # Extract SIFT features from candidate region
        kp_candidate, des_candidate = extract_sift_features(candidate_img)
        if des_candidate is None:
            print(f"[SKIP] No descriptors in candidate region {wp}")
            continue

        # Match features between template and candidate
        good_matches = match_sift_features(des_template, des_candidate)

        print(f"[DEBUG] Matches found: {len(good_matches)} for word property {wp}")

        # Threshold to decide if this candidate matches the template word
        if len(good_matches) > 1:  # Tune this threshold as needed
            foundWords.append(wp)

    return foundWords

def wordPropertyTOdirectionConvertor(wordsProperty: tuple):
    """
    Convert wordsProperty from ((left, top), (right, bottom)) to (left, top, right, bottom).
    """
    return (wordsProperty[0][0], wordsProperty[0][1], wordsProperty[1][0], wordsProperty[1][1])


def putRectangles(image: np.ndarray, wordProperty: list):
    """
    Draw rectangles on the image for each word property.
    """
    for leftTop_RightBottom in wordProperty:
        cv2.rectangle(image, pt1=leftTop_RightBottom[0], pt2=leftTop_RightBottom[1], color=(0, 255, 0), thickness=1)

    return image
