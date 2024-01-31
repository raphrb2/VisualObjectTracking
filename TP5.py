import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from tools.KalmanFilter import KalmanFilter
from tools.bbox_operations import *
from tools.utils import *
from scipy.optimize import linear_sum_assignment
import cv2
import os
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from scipy.spatial.distance import cosine
import pandas as pd

def calculate_similarity(box1_features, box2_features):
    # Using cosine similarity
    return 1 - cosine(box1_features, box2_features)


def extract_features(image, box):
    """
    Uses a CNN model (ResNet50) to extract visual features from image portions delimited by bounding boxes. 
    These features will be used to evaluate the visual similarity between the tracked objects
    """

    # Load ResNet50 with pretrained weights
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()

    # Transform pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Convert box coordinates to integers and crop the image
    x, y, w, h = map(int, [box[0], box[1], box[2], box[3]])
    crop_img = Image.fromarray(image[y:y+h, x:x+w])
    crop_img = transform(crop_img)

    # Extract features
    with torch.no_grad():
        features = model(crop_img.unsqueeze(0))

    return features.numpy().flatten()

def load_frame_image(frame_number, base_path="ADL-Rundle-6/img1"):
    """
    Load an image corresponding to a specific frame number.

    Args:
    - frame_number (int): The frame number to load.
    - base_path (str): The base path where the images are stored.

    Returns:
    - np.array: The loaded image as a NumPy array, or None if the image does not exist.
    """
    filename = f"{str(frame_number).zfill(6)}.jpg"
    image_path = os.path.join(base_path, filename)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    image = cv2.imread(image_path)
    return image



class Tracker:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.kalman_filters = {}  # Kalman filters for each track
        self.visual_features = {}  # Visual features for each track
        self.original_boxes = {}  # Original bounding boxes for each track
        self.centroids = {}  # Centroids for each track
        self.max_id = 0  # Maximum assigned track ID
        self.dt = dt  # Time step
        self.u_x = u_x  # Estimated acceleration in x
        self.u_y = u_y  # Estimated acceleration in y
        self.std_acc = std_acc  # Standard deviation of acceleration
        self.x_std_meas = x_std_meas  # Measurement noise in x
        self.y_std_meas = y_std_meas  # Measurement noise in y

    def update_tracker(self, box, image, id=None):
        """
        Updates or creates a new tracker with the given bounding box and image.

        Args:
            box (tuple): The bounding box.
            image (np.array): The image frame.
            id (int, optional): The ID of the tracker. Defaults to None.

        Returns:
            int: The ID of the tracker.
        """
        centroid = box2centroid(box)
        if id is None:
            self.max_id += 1
            id = self.max_id
            kf = KalmanFilter(self.dt, self.u_x, self.u_y, self.std_acc, self.x_std_meas, self.y_std_meas)
        else:
            kf = self.kalman_filters.get(id, KalmanFilter(self.dt, self.u_x, self.u_y, self.std_acc, self.x_std_meas, self.y_std_meas))

        kf.update(np.array([[centroid[0]], [centroid[1]]]))
        self.kalman_filters[id] = kf

        features = extract_features(image, box)
        self.visual_features[id] = features

        self.original_boxes[id] = (box[2], box[3])
        self.centroids[id] = centroid

        return id

    def predict(self):
        """
        Predicts the next state for all trackers.

        Returns:
            dict: A dictionary of predicted centroids for each tracker.
        """
        predictions = {}
        for id, kf in self.kalman_filters.items():
            kf.predict()
            pred_centroid = (kf.x[0, 0], kf.x[1, 0])
            predictions[id] = pred_centroid
        return predictions


def main():
    try:
        det = pd.read_csv('ADL-Rundle-6/det/det.txt', sep=',', index_col=0)
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return

    frames = det.index.unique()
    tracker = Tracker(0.1, 1, 1, 1, 0.1, 0.1)

    # Create results file
    results_filename = 'results/nn_results.txt'
    with open(results_filename, 'w') as results_file:
        results_file.write("frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z\n")

        for frame_number in frames:
            frame_data = det.loc[frame_number]
            frame_image = load_frame_image(frame_number)

            if frame_number == 1:
                for _, row in frame_data.iterrows():
                    box = (row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height'])
                    track_id = tracker.update_tracker(box, frame_image)
                    results_file.write(f"{frame_number},{track_id},{box[0]},{box[1]},{box[2]},{box[3]},1,-1,-1,-1\n")
            else:
                preds = tracker.predict()
                pred_f = [centroid2box(preds[id], tracker.original_boxes[id][0], tracker.original_boxes[id][1]) for id in preds]
                frame_data_boxes = frame_to_boxes(frame_data)
                similarity_mat = similarity_matrix_kalman(pred_f, frame_data_boxes)
                row_ind, col_ind = linear_sum_assignment(similarity_mat, maximize=True)

                for row, col in zip(row_ind, col_ind):
                    track_id = list(tracker.kalman_filters.keys())[row]
                    box = frame_data.iloc[col]
                    new_centroid = box2centroid((box['bb_left'], box['bb_top'], box['bb_width'], box['bb_height']))
                    tracker.kalman_filters[track_id].update(np.array([[new_centroid[0]], [new_centroid[1]]]))
                    tracker.centroids[track_id] = new_centroid
                    tracker.original_boxes[track_id] = (box['bb_width'], box['bb_height'])

                    # Write tracking results
                    results_file.write(f"{frame_number},{track_id},{box['bb_left']},{box['bb_top']},{box['bb_width']},{box['bb_height']},1,-1,-1,-1\n")

            boxes = [centroid2box(tracker.centroids[id], tracker.original_boxes[id][0], tracker.original_boxes[id][1]) for id in tracker.centroids]
            box_display_2(boxes, frame_number, list(tracker.kalman_filters.keys()))

if __name__ == '__main__':
    main()




