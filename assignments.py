import numpy as np
from scipy.optimize import linear_sum_assignment


# method for TP2
def greedy(mat, tracks, threshold):
    assignments = []
    used = set()
    for i in range(mat.shape[1]):
        max_indices = np.argsort(-(mat[:, i]))
        for idx in max_indices:
            if mat[idx, i] < threshold:
                break
            if idx not in used:
                assignments.append((i, tracks[idx]))
                used.add(idx)
                break
    return assignments


# method for TP3
def hungarian(mat, tracks, threshold):
    clear_mat = np.where(mat < threshold, 0, mat)
    row_indices, col_indices = linear_sum_assignment(-clear_mat)
    assignments = [(col_indices[i], tracks[row_indices[i]]) for i in range(len(row_indices)) if clear_mat[row_indices[i], col_indices[i]] != 0]
    return assignments




