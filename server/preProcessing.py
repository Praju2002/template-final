import numpy as np
import cv2

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
    """
    processed = cv2.dilate(src= image,kernel=(3,3),iterations=1)
    processed = cv2.morphologyEx(src= processed, op= cv2.MORPH_CLOSE, kernel=(3,3),iterations=1)

    
    processed = cv2.rotate(src= processed , rotateCode= cv2.ROTATE_90_CLOCKWISE)
        
    processed = cv2.dilate(src= processed,kernel=(3,3),iterations=iteration)
    processed = cv2.morphologyEx(src= processed, op= cv2.MORPH_CLOSE, kernel=(3,3),iterations=iteration)
        
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
    """
        you enter image and returns the average height of letters

        there are 4 cases here
        case 1: current pixel = background, next pixel = background 
            do nothing

        case 2: current pixel = background, next pixel = line 
            you are going to entered a line 
                increment number of line

        case 3: current pixel = line, next pixel = background 
            you are going to exit a line 
                increment current height
                update average height
                update current height
                
        case 4: current pixel = line, next pixel = line 
            you are in line
                increment current height


         0 is True
    """
    
    averageHeight = 0
    currentHeight = 0
    NumberOfLine = 0
    imageShape = image.shape
        
    sumOfCurrentPixel = 0
    sumOfNextPixel = np.sum(image[0])

    for i in range(0,imageShape[0]-1):

        sumOfCurrentPixel = sumOfNextPixel
        sumOfNextPixel = np.sum(image[i+1])

        currentPixelIs0 = True if(sumOfCurrentPixel ==0 )else False
        nextPixelIs0 = True if(sumOfNextPixel == 0 )else False

        typeCase = (currentPixelIs0 , nextPixelIs0)

        match typeCase:

            case (True,True):     #in background
                continue

            case (True,False):     #entering line
                NumberOfLine +=1

            case (False,True):      #exiting line 
                currentHeight +=1
                averageHeight = ((averageHeight * (NumberOfLine -1))+currentHeight )/NumberOfLine
                currentHeight = 0

            case (False ,False):   #in line
                currentHeight +=1

            case _:
                print("law as been broken")

    return int(averageHeight)


def requiredNumberOfIterations(x : int):
    """
    where x = height of text in pixel

    this was derived from curve fitting of data extracted from samples.
    """ 
    temp  = 0.0001929 *x**2 + 0.1286 *x - 0.4234
    
    return np.round(temp).astype(int)


    

