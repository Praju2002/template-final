import os
import uuid
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont
from flask_cors import CORS
import preProcessing as pp
import wordExtract as we
import templateOperation as tempOp


app = Flask(__name__)
CORS(app , supports_credentials=True)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ==== API Endpoint ====
@app.route("/upload", methods=["POST"])
def upload():
    image_file = request.files["image"]
    word = request.form.get("word")

    if not image_file or not word:
        return jsonify({"error": "Missing image or word"}), 400

    # Read image
    img_array = np.frombuffer(image_file.read(), np.uint8)
    originalImage = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    greyImage = cv2.imdecode(buf=img_array , flags=cv2.IMREAD_GRAYSCALE )

    image = pp.backgroundBlackForegroundWhite(image=greyImage)

    averageHeight_Pixel,tallestText= pp.averageHeightOfLetters(image=image)
    imageShape = image.shape

    tallestLineImage = image[tallestText[0]:tallestText[1],0:imageShape[1]]
    averageGap_Pixel= pp.averageGapOfLetter(image=tallestLineImage)

    iterationNumber = pp.requiredNumberOfIterations(averageGap= averageGap_Pixel)
    print("avg ht, gap and iteration", averageHeight_Pixel,averageGap_Pixel , iterationNumber)

    smudgedImage = pp.prepareImageForWordExtraction(image=image,iteration= iterationNumber)

    wordsProperty = we.wordExtract(image=smudgedImage , 
                               averageHeight= averageHeight_Pixel,
                               smudgedIteration= iterationNumber)
    
    print("creating template")

    template = tempOp.createTemplate(word= word ,fontSize= averageHeight_Pixel)

    print("template has been created \n template matching images")


    foundWords = tempOp.templateMatching(image=image , template= template , wordsProperty=wordsProperty)

    print(foundWords)

    #to prevent overwriting 
    finalImage=originalImage.copy()

    if(len(foundWords) > 0):
        finalImage = tempOp.putRectangles(image= finalImage , wordProperty= foundWords)
    
    else:
        cv2.putText(img= finalImage , text= "No match found",
                    org=(20,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale= 0.6, color=(0,255,0),
                    thickness=2)

    # Convert final image to base64
    _, buffer = cv2.imencode('.png', finalImage)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"image_base64": base64_img})


@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)