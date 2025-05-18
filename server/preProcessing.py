import numpy as np
import cv2
import random

def backgroundBlackForegroundWhite(image: np.ndarray ):
    """
    makes sure that the image's background is black and foreground is white 
    """
    _, binaryImage = cv2.threshold(src=image, thresh=127, maxval=255,type= cv2.THRESH_BINARY )
   
    whitePixels = np.sum( binaryImage > 127)
    blackPixels = np.sum(binaryImage < 127)

    if whitePixels > blackPixels:
        # Text is black, background is white — invert it
        binaryImage = cv2.bitwise_not(binaryImage)

    return binaryImage


def prepareImageForWordExtraction(image: np.ndarray, iteration : int):
    """
    enter a image with black background and white letters and number of iteration 
    ps: removed morphological closing as they tended to change smudging drastically

    """
    processed = cv2.dilate(src= image,kernel=(3,3),iterations=1)
    #processed = cv2.morphologyEx(src= processed, op= cv2.MORPH_CLOSE, kernel=(3,3),iterations=1)

    
    processed = cv2.rotate(src= processed , rotateCode= cv2.ROTATE_90_CLOCKWISE)
        
    processed = cv2.dilate(src= processed,kernel=(3,3),iterations=iteration)
    #processed = cv2.morphologyEx(src= processed, op= cv2.MORPH_CLOSE, kernel=(3,3),iterations=iteration)
        
    processed = cv2.rotate(src= processed , rotateCode= cv2.ROTATE_90_COUNTERCLOCKWISE )

        

    return processed
        


def mixImage(image:np.ndarray , maks:np.ndarray):
    """
    takes 2 images and gives the average of to by mixing them
    """
    imageSize = image.shape

    for h in range(0,imageSize[0]):
        for w in range(0,imageSize[1]):
            image[h][w] = int(image[h][w] + maks[h][w])/2
    
    return image


def averageHeightOfLetters(image : np.ndarray):
    averageHeight = 0
    currentHeight = 0
    NumberOfLine = 0
    imageShape = image.shape
    heightAtWhichLineStarts = 0
    tallestText = (0,0)
        
    sumOfCurrentPixel = 0
    sumOfNextPixel = np.sum(image[0])

    for i in range(0, imageShape[0] - 1):
        sumOfCurrentPixel = sumOfNextPixel
        sumOfNextPixel = np.sum(image[i + 1])

        # why? it ran into the Unexpected case in line detection so... removed
        # currentPixelIs0 = sumOfCurrentPixel == 0
        # nextPixelIs0 = sumOfNextPixel == 0

        currentPixelIs0 = True if(sumOfCurrentPixel ==0 )else False
        nextPixelIs0 = True if(sumOfNextPixel == 0 )else False

        typeCase = (currentPixelIs0, nextPixelIs0)

        match typeCase:
            case (True, True):
                continue
            case (True, False):
                NumberOfLine += 1
                heightAtWhichLineStarts =i

            case (False, True):
                currentHeight += 1

                if NumberOfLine == 0:
                    print("Warning: Detected line end before start — skipping.")
                    continue
                averageHeight = ((averageHeight * (NumberOfLine - 1)) + currentHeight) / NumberOfLine
                
                #logic for tallest text 
                if((tallestText[1]-tallestText[0])<(i-heightAtWhichLineStarts)):
                    tallestText = (heightAtWhichLineStarts,i)
                                
                currentHeight = 0
            
            case (False, False):
                currentHeight += 1
            case _:
                print("Unexpected case in line detection")

    if NumberOfLine == 0:
        print("Warning: No text lines detected.")
        return 0  # or None if you want to force error handling

    return (int(averageHeight),tallestText)


def requiredNumberOfIterations(averageGap : int):
    """
     where x = average gap between characters in pixel

    this was derived from curve fitting of data extracted from samples.
    # random number to account for any unforseen situation
    """  
     #temp  = 0.5030 * averageGap + -0.3333
    random_number = random.uniform(0.6, 0.8)

    temp = averageGap * random_number

    return np.round(temp).astype(int)



def averageGapOfLetter(image :np.ndarray):
    tallestLineImage_Rotated= cv2.rotate(src= image , rotateCode= cv2.ROTATE_90_CLOCKWISE)
    averageGap_Pixel,_= averageHeightOfLetters(image=tallestLineImage_Rotated)
    return averageGap_Pixel 
