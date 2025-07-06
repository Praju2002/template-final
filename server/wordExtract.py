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
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)
    
    wordsProperty = []
    shift = shiftDueToSmudging(iteration=smudgedIteration)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]   
        y = stats[i, cv2.CC_STAT_TOP]    
        w = stats[i, cv2.CC_STAT_WIDTH]  
        h = stats[i, cv2.CC_STAT_HEIGHT] 

  
        tempWordProperty = ((x - shift, y), (x + w - shift, y + h))
   
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
    if width <= 0 or height <= 0:
        return False

    if(height * width < minArea):
        return False
    elif (height > minHeight and width < minHeight ): 
        return False
    elif (height < minHeight and width > minHeight ):
        return False
    else:
        return True

    aspect_ratio = height / width

    # Max aspect ratio: too tall and thin for a word (e.g., a vertical line or merged blob)
    MAX_HEIGHT_WIDTH_RATIO = 5.0 
    if aspect_ratio > MAX_HEIGHT_WIDTH_RATIO: 
        # print(f"[DEBUG] Filtered: Too tall (aspect_ratio={aspect_ratio:.2f}) for {wordProperty}") # Optional: for debugging
        return False

    # Min aspect ratio: too short and wide for a word (e.g., a horizontal splatter)
    MIN_HEIGHT_WIDTH_RATIO = 0.1 
    if aspect_ratio < MIN_HEIGHT_WIDTH_RATIO and width > (2 * averageHeight): 
        # print(f"[DEBUG] Filtered: Too flat (aspect_ratio={aspect_ratio:.2f}) for {wordProperty}") # Optional: for debugging
        return False

  
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

    if width <= 0 or height <= 0:
        return False

    area = height * width
    if area < minArea:
        return False

    aspect_ratio = height / width
  
    MAX_HEIGHT_WIDTH_RATIO = 5.0 
    if aspect_ratio > MAX_HEIGHT_WIDTH_RATIO: 
        return False
    
    MIN_HEIGHT_WIDTH_RATIO = 0.1 
    if aspect_ratio < MIN_HEIGHT_WIDTH_RATIO and width > (2 * averageHeight): 
        return False
  
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