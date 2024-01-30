import os
import pandas as pd
from bbox_operations import *
from assignments import hungarian
from utils import *
import numpy as np
import cv2

'''
    Hungarian Tracker for TP3
'''

def main():
    try:
        os.remove('results/hungarian_res.csv')
    except OSError:
        pass

    det = pd.read_csv('ADL-Rundle-6/det/det.txt', sep=',', index_col=0)
    frames = det.index.unique()
    sigma_iou = 0.2
    tracks = update_tracks(0, [], len(det.loc[1]))

    frame_dfs = [det.loc[i] for i in frames]

    with open('results/hungarian_res.csv', 'a') as f:
        # add the header
        f.write("frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z\n")
        for i in range(len(frames) - 1):
            f1 = frame_dfs[i]
            f2 = frame_dfs[i + 1]
            similarity_matrix = similarity_matrix_iou(f1, f2)

            assignments = hungarian(similarity_matrix, tracks, sigma_iou)
            tracks = update_tracks(max(tracks), assignments, len(f2))
            display_boxes(frame_to_boxes(f2), frames[i + 1], tracks)
            cv2.waitKey(10)

            for col, id in assignments:
                f.write(f"{i+1},{id},{','.join(map(str, f2.iloc[col][['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']]))}\n")

if __name__ == '__main__':
    main()
