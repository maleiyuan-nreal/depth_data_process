import numpy as np

# nreal left 内参
nreal_left_K = np.array([
            [238, 0, 320],
            [0, 238, 240],
            [0, 0, 1],
        ], dtype=np.float32)


nreal_right_K = np.array([
            [238, 0, 320],
            [0, 238, 240],
            [0, 0, 1],
        ], dtype=np.float32)

# nreal left 畸变
nreal_left_distCoeffs = np.float32([6.5248813573164491e-02, -2.4770151684803149e-02, 4.5283449521595666e-03, 3.5708289268269745e-04, 0., 0., 0., 0.])



nreal_row = 480 
nreal_col = 640