DocuFind

Overview:
This project is a Flask-based web application that detects and highlights specific words in images using computer vision techniques. It supports both English and Nepali text, leveraging OpenCV for image preprocessing, word extraction, and template matching. The system processes uploaded images, identifies regions containing a specified search word, and returns base64-encoded images showing the processing stages and final results with highlighted matches. A testing framework evaluates performance using precision, recall, and F1-score metrics based on ground truth data.
Features

Multilingual Support: Handles English and Nepali text with appropriate font selection.
Image Preprocessing: Converts images to binary (black background, white foreground) and applies smudging for robust word extraction.
Word Extraction: Uses connected components analysis to identify word regions efficiently.
Template Matching: Supports multiple methods (TM_SQDIFF_NORMED, TM_CCOEFF_NORMED, SIFT) with dynamic thresholding.
Web Interface: Flask API for uploading images and search words, returning processed images and match coordinates.
Testing Framework: Evaluates detection accuracy using IoU-based metrics on a ground truth dataset.


Requirements

Python 3.8+
Libraries: numpy, opencv-python, Pillow, flask, flask-cors, requests
Font files: arial.TTF (English), nepali.TTF (Nepali) in the fonts/ directory
Optional: Test images and scanned.json for running tests

Installation

Clone the Repository:
git clone <https://github.com/Praju2002/template-final.git>
cd template-final

Install Dependencies:
pip install numpy opencv-python Pillow flask flask-cors requests


Create Directories:
mkdir uploads results



Usage
Running the Flask Application

Start the Flask server:
python app.py



Contributing
Contributions are welcome! Please submit issues or pull requests to the repository. Ensure code follows PEP 8 style guidelines and includes tests for new features.
Contact
For questions or feedback, please contact Praju Khanal at prajukhanal21@gmail.com.
**📸**  
!(./client/assets/Screenshot 2025-07-14 122347.png)  
!(./client/assets/Screenshot 2025-07-14 122248.png)  
!(./client/assets/Screenshot 2025-07-14 122334.png)  
