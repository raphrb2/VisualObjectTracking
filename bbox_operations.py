import numpy as np

'''
    Bounding Box Operations for TP2/3/4

    bbox = (x, y, w, h)
'''

def compute_iou(b1, b2):
    coin_x_max = max(b1[0][0], b2[0][0])
    coin_y_max = max(b1[0][1], b2[0][1])
    coin_x_min = min(b1[0][0] + b1[1], b2[0][0] + b2[1])
    coin_y_min = min(b1[0][1] + b1[2], b2[0][1] + b2[2])

    iou = max(0, (coin_x_min - coin_x_max) * (coin_y_min - coin_y_max))

    surface = (b1[1] * b1[2]) + (b2[1] * b2[2]) - iou

    return iou / surface


def similarity_matrix_iou(f1, f2):
    coords_f1 = f1[['bb_left', 'bb_top']].values
    dimensions_f1 = f1[['bb_width', 'bb_height']].values
    coords_f2 = f2[['bb_left', 'bb_top']].values
    dimensions_f2 = f2[['bb_width', 'bb_height']].values

    # init
    similarity_matrix = np.zeros((len(f1), len(f2)))

    # compute similarity matrix
    for i in range(len(f1)):
        for j in range(len(f2)):
            b1 = (coords_f1[i], dimensions_f1[i][0], dimensions_f1[i][1])
            b2 = (coords_f2[j], dimensions_f2[j][0], dimensions_f2[j][1])
            similarity_matrix[i, j] = compute_iou(b1, b2)

    return similarity_matrix

def similarity_matrix_kalman(f1, f2):
    similarity_matrix = np.zeros((len(f1), len(f2)))
    for i in range(len(f1)):
        for j in range(len(f2)):
            b1 = (f1[i][0:2], f1[i][2], f1[i][3])
            b2 = (f2[j][0:2], f2[j][2], f2[j][3])
            similarity_matrix[i, j] = compute_iou(b1, b2)
    return similarity_matrix