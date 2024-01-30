import pandas as pd
from bbox_operations import *
from assignments import greedy
from utils import *
import numpy as np
import cv2

'''
    Greedy Tracker for TP2
'''

def main():
    det = pd.read_csv('ADL-Rundle-6/det/det.txt', sep=',', index_col=0)
    frames = det.index.unique()
    sigma_iou = 0.2
    tracks = update_tracks(0, [], len(det.loc[1]))

    frame_dfs = [det.loc[i] for i in frames]

    for i in range(len(frames) - 1):
        f1 = frame_dfs[i]
        f2 = frame_dfs[i + 1]
        similarity_matrix = similarity_matrix_iou(f1, f2)

        assignments = greedy(similarity_matrix, tracks, sigma_iou)
        tracks = update_tracks(max(tracks), assignments, len(f2))
        display_boxes(frame_to_boxes(f2), frames[i + 1], tracks)
        cv2.waitKey(10)

if __name__ == '__main__':
    main()


