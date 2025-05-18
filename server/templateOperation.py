import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont  
import preProcessing as pp
import wordExtract as we



def createTemplate(word :str , fontSize : int ):
    """
    this is to create template for the word
    word = string of the word
    fontSize = int of height of text in pixel
    wordProperty = tuple of ((left,top),(right,bottom)) position of the word
    """    
    

    # Create an image
    fontSize = np.round(fontSize*1.5 ).astype(int)           # when resizing it would have maximum property
    #padding = paddingCalculation(fontSize)
    font = ImageFont.truetype("arial.ttf", fontSize)


    dummy = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), word, font=font)
        
    # Calculate size with padding
    width = (bbox[2] - bbox[0] + fontSize)  
    height =(bbox[3] - bbox[1] +fontSize ) 

    img = Image.new("L", (width, height), "black")
    draw = ImageDraw.Draw(img)

    # Load custom font

    # Draw text with stroke and anchor
    draw.text((int(width/2) ,int(height/2)), 
              text= word, 
              fill="white", 
              font=font,
              anchor="mm")

    wordImage = np.array(img)

    _, wordImage = cv2.threshold(src=wordImage, thresh=127, maxval=255,type= cv2.THRESH_BINARY )

    averageHeight_Pixel,tallestText= pp.averageHeightOfLetters(image=wordImage)
    imageShape = wordImage.shape

    tallestLineImage = wordImage[tallestText[0]:tallestText[1],0:imageShape[1]]
    averageGap_Pixel= pp.averageGapOfLetter(image=tallestLineImage)

    iterationNumber = pp.requiredNumberOfIterations(averageGap= averageGap_Pixel)

    print(averageHeight_Pixel,averageGap_Pixel,iterationNumber)

    smudgedImage = pp.prepareImageForWordExtraction(image=wordImage,iteration= iterationNumber)
   
    #find words from the images 

    wordsProperty = we.wordExtract(image=smudgedImage , 
                                averageHeight= averageHeight_Pixel,
                                smudgedIteration= iterationNumber)
    
    wordsProperty = wordsProperty.pop()

    left,top,right,bottom = wordPropertyTOdirectionConvertor(wordsProperty=wordsProperty)

    wordImage = wordImage[top:bottom,left:right ]    

    return wordImage


def paddingCalculation (x :int):
    """
    calcualtes the padding required for the search word
    """
    return int(np.round(-0.0006503 *x**2 + 0.3229 * x + 0.7658))
    

def templateMatching(image : np.ndarray , template : np.ndarray , wordsProperty : list[tuple]):
    """
    It gets image and template of the word and the position of the word.

    
    then template is rescaled to the size of the word using wp.

    using wp we find the section of that image.
        note: the extracted section would 1px wider in all for accuracy purposes,
    then that section is template matched using cv2 and the greatest score is associated.

    repeat for all possible words and if the accuracy is low then word is discarded.

    and returns (wordProperty , similarity score)

    """

    foundWords = []
    
    

    for wp in wordsProperty:

        left,top,right,bottom = wordPropertyTOdirectionConvertor(wp)

        #to flush out any pervious image data in the variable 

        scaledTemplate = np.zeros((0, 0), dtype=np.uint8)
        # scaledTemplate = np.zeros((0, 0), dtype=np.uint8)


        scaledTemplate = cv2.resize(src= template , dsize=(right-left,bottom-top) , interpolation= cv2.INTER_AREA)

        # sectionOfImage = image[top-1:bottom+1,left-1:right+1]


        # similarityScore = cv2.matchTemplate(image=sectionOfImage,templ= scaledTemplate,method=cv2.TM_SQDIFF_NORMED)

       
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarityScore)

        # if(min_val <1):
        #     foundWords.append(wp)
        h,w = image.shape[:2]
        top=max(0,top-1)
        bottom=min(h,bottom+1)
        left=max(0,left-1)
        right=min(w,right+1)
        sectionOfImage=image[top:bottom,left:right]

        if sectionOfImage.shape[0] >=scaledTemplate.shape[0] and sectionOfImage.shape[1] >= scaledTemplate.shape[1]:
            similarityScore=cv2.matchTemplate(image=sectionOfImage,templ=scaledTemplate,method=cv2.TM_SQDIFF_NORMED
            )
            min_val, _, _, _ = cv2.minMaxLoc(similarityScore)
            if min_val <1:
                foundWords.append(wp)
        else:
            print(f"[SKIP] Word property {wp} - section {sectionOfImage.shape} ,template {scaledTemplate.shape} ")
        # print(f"min_val: {min_val} , max_val: {max_val} , min_loc: {min_loc} , max_loc: {max_loc}")
    
    return foundWords


def wordPropertyTOdirectionConvertor(wordsProperty :tuple):
    """ we receive the word property as
    ((left,top),(right,bottom))
    
    returns  left, top , right bottom
    """
    return (wordsProperty[0][0] ,wordsProperty[0][1] ,wordsProperty[1][0] ,wordsProperty[1][1] )


def putRectangles(image : np.ndarray ,wordProperty: list):
    """
    it receives image and list fo word property and puts rectangle on them and returns that image
    """

    for leftTop_RightBottom in wordProperty:
        cv2.rectangle(img=image ,pt1= leftTop_RightBottom[0], 
                      color=(0,255,0),pt2= leftTop_RightBottom[1],
                      thickness= 1)

    return image    






    


