import json
import requests
import os
import cv2
import numpy as np

FLASK_API_URL = "http://127.0.0.1:5000/upload"
GROUND_TRUTH_FILE ="scanned.json"
TEST_IMAGES_DIR = "test/scanned/"

TOTAL_TRUE_POSITIVES = 0
TOTAL_FALSE_POSITIVES = 0
TOTAL_FALSE_NEGATIVES = 0


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = float(box1_area + box2_area - intersection_area)

    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def run_all_tests():
    global TOTAL_TRUE_POSITIVES, TOTAL_FALSE_POSITIVES, TOTAL_FALSE_NEGATIVES
    try:
        with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file '{GROUND_TRUTH_FILE}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{GROUND_TRUTH_FILE}'. Check file format.")
        return

    print(f"Loaded {len(test_cases)} test cases from ground truth.")

    for i, test_case in enumerate(test_cases):
        image_path = os.path.join(TEST_IMAGES_DIR, os.path.basename(test_case['image_filename']))
        search_word = test_case['search_word']
        ground_truth_boxes = test_case['ground_truth_boxes'] 

        print(f"\n--- Testing Case {i+1}/{len(test_cases)}: {os.path.basename(image_path)} (Word: '{search_word}') ---")

        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at '{image_path}'. Skipping this test case.")
            continue

        try:
            with open(image_path, 'rb') as img_file:
                files = {'image': (os.path.basename(image_path), img_file, 'image/png')} 
                data = {'word': search_word}
                response = requests.post(FLASK_API_URL, files=files, data=data)
                response.raise_for_status() 

            api_response = response.json()
          
            detected_boxes = []
            if "foundWords" in api_response and isinstance(api_response["foundWords"], list):
                for fw in api_response["foundWords"]:
                    
                    if isinstance(fw, list) and len(fw) == 2 and isinstance(fw[0], (list, tuple)) and isinstance(fw[1], (list, tuple)):
                        detected_boxes.append([fw[0][0], fw[0][1], fw[1][0], fw[1][1]])
                    elif isinstance(fw, list) and len(fw) == 4:
                        detected_boxes.append(fw)
                    elif isinstance(fw, dict) and 'box' in fw and isinstance(fw['box'], list) and len(fw['box']) == 4:
                         detected_boxes.append(fw['box'])

            print(f"Ground Truth Boxes: {ground_truth_boxes}")
            print(f"Detected Boxes: {detected_boxes}")

            current_tps = 0
            current_fps = 0
            current_fns = 0

            matched_gt_indices = [False] * len(ground_truth_boxes)

            for det_box in detected_boxes:
                is_true_positive = False
                for gt_idx, gt_box in enumerate(ground_truth_boxes):
                    if not matched_gt_indices[gt_idx]:
                        iou = calculate_iou(det_box, gt_box)
                        
                        IOU_THRESHOLD = 0.4 
                        
                        if iou >= IOU_THRESHOLD:
                            current_tps += 1
                            is_true_positive = True
                            matched_gt_indices[gt_idx] = True 
                            break 
                if not is_true_positive:
                    current_fps += 1 

            for gt_idx, matched in enumerate(matched_gt_indices):
                if not matched:
                    current_fns += 1

            TOTAL_TRUE_POSITIVES += current_tps
            TOTAL_FALSE_POSITIVES += current_fps
            TOTAL_FALSE_NEGATIVES += current_fns

            print(f"TP: {current_tps}, FP: {current_fps}, FN: {current_fns}")

        except requests.exceptions.RequestException as e:
            print(f"Error making API request for {image_path}: {e}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON response for {image_path}. Response text: {response.text}")
        except Exception as e:
            print(f"An unexpected error occurred for {image_path}: {e}")

    print("\n--- Overall Results ---")
    print(f"Total True Positives (TP): {TOTAL_TRUE_POSITIVES}")
    print(f"Total False Positives (FP): {TOTAL_FALSE_POSITIVES}")
    print(f"Total False Negatives (FN): {TOTAL_FALSE_NEGATIVES}")

    precision = TOTAL_TRUE_POSITIVES / (TOTAL_TRUE_POSITIVES + TOTAL_FALSE_POSITIVES) if (TOTAL_TRUE_POSITIVES + TOTAL_FALSE_POSITIVES) > 0 else 0.0
    recall = TOTAL_TRUE_POSITIVES / (TOTAL_TRUE_POSITIVES + TOTAL_FALSE_NEGATIVES) if (TOTAL_TRUE_POSITIVES + TOTAL_FALSE_NEGATIVES) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

if __name__ == "__main__":
    run_all_tests()