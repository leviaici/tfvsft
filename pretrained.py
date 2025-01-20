import os
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

# Initialize the MTCNN model
mtcnn = MTCNN(keep_all=True, thresholds=[0.1, 0.2, 0.2])

# Function to predict face bounding boxes and confidence scores
def predict_faces_with_confidence(image_path):
    try:
        # Open the image
        image = Image.open(image_path).convert('RGB')
        
        # Detect faces (bounding boxes and probabilities)
        boxes, probabilities = mtcnn.detect(image)
        
        # Return bounding boxes and confidence scores
        if boxes is not None:
            return boxes.tolist(), probabilities.tolist()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    return [], []

# Function to process all images in a folder and save results
def process_images(image_folder, output_folder, max_counter=20):
    all_boxes = []       # Store all bounding boxes
    all_scores = []      # Store all confidence scores
    all_file_names = []  # Store all image names
    count = 0
    
    # Iterate through all files in the folder
    for image_name in os.listdir(image_folder):
        if count > max_counter:
            break
        count += 1
        image_path = os.path.join(image_folder, image_name)
        print(f"\rProcessing: {image_name}")
        
        # Predict bounding boxes and confidence scores
        boxes, scores = predict_faces_with_confidence(image_path)
        
        # Append results to respective lists
        if boxes: 
            for box, score in zip(boxes, scores):
                all_boxes.append(box)
                all_scores.append(score)
                all_file_names.append(image_name)
    
    # Convert lists to NumPy arrays with dtype=int32
    all_boxes = np.array(all_boxes, dtype=np.int32)
    all_scores = np.array(all_scores, dtype=object)
    all_file_names = np.array(all_file_names, dtype=object)

    # Save the results to .npy files
    np.save(os.path.join(output_folder, "detections_all_faces.npy"), all_boxes)
    np.save(os.path.join(output_folder, "scores_all_faces.npy"), all_scores)
    np.save(os.path.join(output_folder, "file_names_all_faces.npy"), all_file_names)
    print("Results saved as .npy files.")

# Main function
if __name__ == "__main__":
    # Path to the folder containing images
    image_folder = "validare/validare/"
    
    # Path to save the output .npy files
    output_folder = "output/"
    os.makedirs(output_folder, exist_ok=True)
    
    # Process images and save results
    process_images(image_folder, output_folder, max_counter=200)