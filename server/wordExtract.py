import numpy as np
import cv2

def wordExtract(image : np.ndarray,averageHeight : int, smudgedIteration : int):
    """
    it takes image ( smudged image preferred) and figures out the words are in the image

    average height of text is to verify weather or not a group of pixels are a word

    smudged iteration is used to figure out how much shifting is required

    """
    imageShape = image.shape
    pixelCoordinate = []
    wordsProperty = []
    whitePixel = 255
    shift = shiftDueToSmudging(iteration=smudgedIteration)

    #loads all the pixels coordinate of image into pixel coordinate and pop one after another 
    for h in range(1, imageShape[0]-1):
        for w in range(1, imageShape[1]-1):
            pixelCoordinate.append((h,w))
    

    while(len(pixelCoordinate)>0):
        currentPixel = pixelCoordinate.pop()
        if(image[currentPixel[0]][currentPixel[1]] == whitePixel): # found 1st pixel of word
            print("word might be in " , currentPixel)
            #remove from total pixel list and add to white pixel ine list
            whitePixelList = []
            neighborList = []
            
            whitePixelList.append(currentPixel)
            neighborList = getNewNeighbors(coordinate= currentPixel , 
                                                     pixelCoordinate=pixelCoordinate,
                                                     currentNeighbor=neighborList)
            
        else:#found a background pixel
            continue

        #one the pixel of word is detected so gather all the white pixels
        while(len(neighborList) > 0):
            currentPixel = neighborList.pop()
            if(image[currentPixel[0]][currentPixel[1]] == whitePixel): # found another of word
                #remove from total pixel list and add to white pixel ine list                
                whitePixelList.append(pixelCoordinate.pop(pixelCoordinate.index(currentPixel)))
                #as Some pixels are already in
                neighborList.extend (getNewNeighbors(coordinate= currentPixel , 
                                                     pixelCoordinate=pixelCoordinate,
                                                     currentNeighbor=neighborList))
                
            else:
                #found a background pixel
                 pixelCoordinate.pop(pixelCoordinate.index(currentPixel))

        """
        #all of pixels of word has been collected in white pixels
            
        we now have list of tuples of adjacent white pixels so now we calculate
        the top left corner of the box in which the word will fit 
        height and width of the word in pixels
        """
        tempWordProperty = wordPropertyExtraction(pixelList = whitePixelList,shift=shift)
          
        #check if the word is valid or not
        # if true then pixel is word else not  word 
        if(wordVerification(tempWordProperty , averageHeight=averageHeight)):
            print("Verified")
            wordsProperty.append(tempWordProperty)
        else:
            print("rejected")
            continue
    
    return wordsProperty


def getNewNeighbors(coordinate : tuple, pixelCoordinate : list , currentNeighbor : list):
    """
    it get the coordinates the current pixels position and initiates the 8 neighbors of the pixels and the filters out
    all the pixels that are already dealt with and are not is current neighbor list
    """
    xcord = coordinate[0]
    ycord = coordinate[1]
    cleanNeighbor = [] 
                
    #8 neighbors of current pixels
    allNeighbor = [
    (x, y)
    for x in range(xcord - 1, xcord + 2)
    for y in range(ycord - 1, ycord + 2)
    if (x, y) != (xcord, ycord)]

    for neighbor in allNeighbor:
        if(pixelCoordinate.__contains__(neighbor) and not currentNeighbor.__contains__(neighbor)):
            cleanNeighbor.append(neighbor)
    
    return cleanNeighbor
    

def wordPropertyExtraction(pixelList : list,shift : int):
    """
    takes a list of tuples as coordinates and returns positional property of the word in the format
    (left,top),(right,bottom)
    
    note : it also takes account for shifting of pixels due to smudging and corrects it
    
    """
    x= []
    y = []
    for pix in pixelList:
        x.append(pix[1])
        y.append(pix[0])
    
    left = min(x)
    right = max(x)
    top = min(y)
    bottom = max(y)

    return((left-shift,top),(right-shift,bottom))


def wordVerification(wordProperty : tuple,averageHeight : int):
    """
    we receive the word property as
    ((left,top),(right,bottom))

    this is relative operations but this helps to eliminate any small noise that
    pass thought the pre processing and help us prevent any false words

    for a group of pixels to be considered a word 
    if the width and height is less than minHeight = averageHeight *0.3
    if the area covered by pixel is less than 0.2 where area =  minHeight**2
    """
    minHeight = np.round(averageHeight * 0.5)
    minArea = minHeight**2
    width = (wordProperty[1][0]- wordProperty[0][0])
    height = (wordProperty[1][1]-wordProperty[0][1])

    if(height* width < minArea):
        return False
    elif (height > minHeight and width < minHeight ):
        return False
    elif (height < minHeight and width > minHeight ):
        return False
    else:
        return True
    


def shiftDueToSmudging( iteration :int):
    """
    it calculates how much shifting should be done to correct the errors of smudging
    """
    if(iteration >= 1 ):
        return int(iteration/2)
    else:
        return 0







                        
                

                    
                

               

                

                



                


                



                    


