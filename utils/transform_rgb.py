import numpy as np



def get_transform_matrix(nreal_left_K, dataset_K):
    nreal_left_K_inv = np.linalg.pinv(nreal_left_K)
    transform_matrix = np.matmul(dataset_K, nreal_left_K_inv)
    return transform_matrix


def transform_rgb(image, transform_matrix):
    image_row = 480
    image_col = 640
    channel = 3
    x = np.arange(0, image_col)
    y = np.arange(0, image_row)
    mesh_x, mesh_y = np.meshgrid(x, y)
    xy = np.zeros((np.size(mesh_x), 1, 2), dtype=np.float32)
    xy[:, 0, 0] = np.reshape(mesh_x, -1)
    xy[:, 0, 1] = np.reshape(mesh_y, -1)
    
    z = np.ones([xy.shape[0], 1], dtype=xy.dtype)
    xyz_undistorted = np.hstack([xy[:, 0, :], z])
    
    
    projected_image = np.zeros((image_row, image_col, channel))
    for item in xyz_undistorted:
        col, row = item[0], item[1]
        ans = np.matmul(transform_matrix, item)
        new_col, new_row = ans[0], ans[1]
        if new_col>=0 and new_col<image.shape[1] and new_row>=0 and new_row<image.shape[0]:
            projected_image[int(row), int(col), :] = image[int(new_row), int(new_col), :]
        
    projected_image = projected_image.astype('uint8') 
    return projected_image