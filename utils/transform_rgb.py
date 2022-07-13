import numpy as np
import cv2

from bfuncs.images import SE3, project_depth_image_to_point_cloud, project_poind_cloud_to_depth_image


def project_rgb(depth, image, nreal_glass_info, need_inverse=False):
    """_summary_

    Args:
        depth : 深度图
        image : 另外一目的image
        
        need_inverse : dpeth为左目需要inverse
    """
    K_left = np.float32(nreal_glass_info['K_left'])
    K_right = np.float32(nreal_glass_info['K_right'])
    kc_left = np.float32(nreal_glass_info['kc_left'])
    kc_right = np.float32(nreal_glass_info['kc_right'])
    left_p_right = np.float32(nreal_glass_info['left_p_right'])
    left_q_right = np.float32(nreal_glass_info['left_q_right'])
    if need_inverse:
        K_in = K_left
        kc_in = kc_left
        K_out = K_right
        kc_out = kc_right
    else:
        K_in = K_right
        kc_in = kc_right
        K_out = K_left
        kc_out = kc_left
    
    xyz_undistorted = project_depth_image_to_point_cloud(depth, K_in, distCoeffs_depth=kc_in)

    xyz_undistorted = xyz_undistorted[xyz_undistorted[:, 2] <= 100, :]
    xyz_undistorted = xyz_undistorted[xyz_undistorted[:, 2] > 0, :]
    left_2_right_SE3 = SE3(rotation=left_q_right, translation=left_p_right)
    
    
    if need_inverse:
        # left_2_right_SE3 = left_2_right_SE3.inverse()
        xyz_undistorted_nreal_left = left_2_right_SE3.inverse_transform_point_cloud(xyz_undistorted)
    else:
        xyz_undistorted_nreal_left = left_2_right_SE3.transform_point_cloud(xyz_undistorted)

    rvec = np.array([0, 0, 0], dtype=np.float32) # rotation vector
    tvec = np.array([0, 0, 0], dtype=np.float32)
    uv_distored_nreal_left, _ = cv2.projectPoints(xyz_undistorted_nreal_left, rvec=rvec, tvec=tvec, cameraMatrix=K_out, distCoeffs=kc_out)
    uv_distored_nreal_left = uv_distored_nreal_left[:, 0, :]


    map1_x = uv_distored_nreal_left[:, 0].reshape(image.shape[0],image.shape[1])
    map1_y = uv_distored_nreal_left[:, 1].reshape(image.shape[0],image.shape[1])
    rgb_trans = cv2.remap(image, map1_x, map1_y,
                                    interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))
    return rgb_trans