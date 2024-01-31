import numpy as np
import pandas as pd
import cv2
from tools.KalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from tools.bbox_operations import *
from tools.utils import *

'''
    Kalman Tracker for TP4
'''
        

class Tracker:
    '''
    Attributes:

        kalman_filters: dict
            Dictionary of Kalman filters, indexed by track id
        original_boxes: dict
            Dictionary of original bounding boxes, indexed by track id
        centroids: dict
            Dictionary of centroids, indexed by track id
        max_id: int
            Maximum track id
        dt: float
            Time step
        u_x: float
            Acceleration in x direction
        u_y: float
            Acceleration in y direction
        std_acc: float
            Standard deviation of acceleration
        x_std_meas: float
            Standard deviation of measurement in x direction
        y_std_meas: float
            Standard deviation of measurement in y direction
    '''
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.kalman_filters = {}
        self.original_boxes = {}
        self.centroids = {}
        self.max_id = 0
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.x_std_meas = x_std_meas
        self.y_std_meas = y_std_meas

    def update_tracker(self, box, id=None):
        centroid = box2centroid(box)
        if id is None:
            self.max_id += 1
            id = self.max_id
            kf = KalmanFilter(self.dt, self.u_x, self.u_y, self.std_acc, self.x_std_meas, self.y_std_meas)
        else:
            kf = self.kalman_filters.get(id, KalmanFilter(self.dt, self.u_x, self.u_y, self.std_acc, self.x_std_meas, self.y_std_meas))
        
        kf.update(np.array([[centroid[0]], [centroid[1]]]))
        self.kalman_filters[id] = kf
        self.original_boxes[id] = (box[2], box[3])
        self.centroids[id] = centroid
        return id

    def predict(self):
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
    results_filename = 'results/kalman_results.csv'
    with open(results_filename, 'w') as results_file:
        results_file.write("frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z\n")
        for frame_number in frames:
            frame_data = det.loc[frame_number]
            if frame_number == 1:
                for _, row in frame_data.iterrows():
                    box = (row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height'])
                    track_id = tracker.update_tracker(box)
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