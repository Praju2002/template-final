import numpy as np
import cv2

def wordExtract(image : np.ndarray,averageHeight : int, smudgedIteration : int):
    """
    It takes an image (smudged image preferred) and figures out the words in it
    using connected components analysis for efficiency.

    average height of text is to verify whether or not a group of pixels are a word.
    smudged iteration is used to figure out how much shifting is required.
    """
    # Ensure image is binary (0 or 255) for connectedComponentsWithStats.
    # Assumes the input 'image' is already pre-processed to have a black background and white foreground.
    
    # Use cv2.connectedComponentsWithStats for efficient word extraction.
    # Connectivity 8 means it considers 8 neighbors (like your original getNewNeighbors logic).
    # cv2.CV_32S is the output label type.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)
    
    wordsProperty = []
    shift = shiftDueToSmudging(iteration=smudgedIteration)

    # Loop through all connected components (skip background, which is typically label 0).
    for i in range(1, num_labels):
        # Extract bounding box statistics directly from 'stats' array.
        x = stats[i, cv2.CC_STAT_LEFT]   # Leftmost x coordinate of the bounding box
        y = stats[i, cv2.CC_STAT_TOP]    # Topmost y coordinate of the bounding box
        w = stats[i, cv2.CC_STAT_WIDTH]  # Width of the bounding box
        h = stats[i, cv2.CC_STAT_HEIGHT] # Height of the bounding box

        # Construct the word property tuple: ((left,top),(right,bottom)).
        # Apply the 'shift' correction to the x-coordinates (left and right).
        # x + w gives the coordinate *after* the last pixel, so it's the right boundary.
        tempWordProperty = ((x - shift, y), (x + w - shift, y + h))
        
        # Use your existing wordVerification logic to filter out noise or unwanted components.
        if wordVerification(tempWordProperty, averageHeight=averageHeight):
            wordsProperty.append(tempWordProperty)
            
    return wordsProperty


# def wordVerification(wordProperty : tuple,averageHeight : int):
    """
    We receive the word property as ((left,top),(right,bottom)).

    This function helps to eliminate small noise that passed through
    pre-processing and prevent false word detections.

    For a group of pixels to be considered a word:
    - Its area (width * height) must not be too small.
    - It must not be disproportionately thin (either very tall and narrow, or very short and wide).
    """
    minHeight = np.round(averageHeight * 0.5)
    minArea = minHeight**2
    width = (wordProperty[1][0]- wordProperty[0][0])
    height = (wordProperty[1][1]-wordProperty[0][1])

    # Ensure width and height are positive to avoid errors in calculations.
    if width <= 0 or height <= 0:
        return False

    if(height * width < minArea):
        return False
    elif (height > minHeight and width < minHeight ): # Filters out thin vertical lines
        return False
    elif (height < minHeight and width > minHeight ): # Filters out thin horizontal lines
        return False
    else:
        return True

    aspect_ratio = height / width

    # Max aspect ratio: too tall and thin for a word (e.g., a vertical line or merged blob)
    # Tune MAX_HEIGHT_WIDTH_RATIO (e.g., 3.0 to 7.0). Start with 5.0, then try 4.0 or 3.0 if still an issue.
    MAX_HEIGHT_WIDTH_RATIO = 5.0 
    if aspect_ratio > MAX_HEIGHT_WIDTH_RATIO: 
        # print(f"[DEBUG] Filtered: Too tall (aspect_ratio={aspect_ratio:.2f}) for {wordProperty}") # Optional: for debugging
        return False

    # Min aspect ratio: too short and wide for a word (e.g., a horizontal splatter)
    # This is less likely the cause of your current problem, but good for completeness.
    MIN_HEIGHT_WIDTH_RATIO = 0.1 
    if aspect_ratio < MIN_HEIGHT_WIDTH_RATIO and width > (2 * averageHeight): # Only filter if it's also relatively wide
        # print(f"[DEBUG] Filtered: Too flat (aspect_ratio={aspect_ratio:.2f}) for {wordProperty}") # Optional: for debugging
        return False
    # --- END NEW IMPROVEMENT ---

    # Original checks (can keep as additional safeguard if desired, but aspect ratio is more general)
    # These are also good for catching extreme thin lines.
    if (height > (2 * averageHeight) and width < (averageHeight * 0.5) ): # Very tall and very narrow line
        return False
    elif (width > (3 * averageHeight) and height < (averageHeight * 0.5) ): # Very wide and very short line
        return False
    else:
        return True
def wordVerification(wordProperty : tuple, averageHeight : int):
    """
    We receive the word property as ((left,top),(right,bottom)).

    This function helps to eliminate small noise that passed through
    pre-processing and prevent false word detections.

    For a group of pixels to be considered a word:
    - Its area (width * height) must not be too small.
    - It must not be disproportionately thin (either very tall and narrow, or very short and wide).
    """
    minHeight = np.round(averageHeight * 0.5)
    minArea = minHeight**2
    width = (wordProperty[1][0] - wordProperty[0][0])
    height = (wordProperty[1][1] - wordProperty[0][1])

    # Ensure width and height are positive to avoid errors in calculations.
    if width <= 0 or height <= 0:
        return False

    area = height * width
    if area < minArea:
        return False

    # --- CRITICAL NEW IMPROVEMENT: Explicit Aspect Ratio Filtering ---
    aspect_ratio = height / width
    
    # Max aspect ratio: Filters out components that are too tall/thin.
    # This directly addresses the "tall green box" issue.
    # A typical word has an aspect ratio around 0.2 to 2.0. If it's too high,
    # it likely means characters or lines have merged vertically.
    # You might need to tune this value (e.g., from 5.0 down to 3.0 or 2.5) based on
    # how tall your Nepali words typically appear relative to their width.
    MAX_HEIGHT_WIDTH_RATIO = 5.0 
    if aspect_ratio > MAX_HEIGHT_WIDTH_RATIO: 
        return False
    
    # Min aspect ratio: Filters out components that are too short/wide.
    # This prevents identifying horizontal "splatters" or merged lines as words.
    MIN_HEIGHT_WIDTH_RATIO = 0.1 
    if aspect_ratio < MIN_HEIGHT_WIDTH_RATIO and width > (2 * averageHeight): 
        return False
    # --- END CRITICAL NEW IMPROVEMENT ---

    # Original checks (can keep as additional safeguard if desired)
    if (height > (2 * averageHeight) and width < (averageHeight * 0.5) ): 
        return False
    elif (width > (3 * averageHeight) and height < (averageHeight * 0.5) ): 
        return False
    else:
        return True

def shiftDueToSmudging( iteration :int):
    """
    Calculates how much shifting should be done to correct the errors of smudging.
    """
    if(iteration >= 1 ):
        return int(iteration/2)
    else:
        return 0