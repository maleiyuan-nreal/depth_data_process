import cv2
import numpy as np


def pretty_depth(depth_path):
    """
    用PIL是无法读取uint16格式的数据的,详情见:
    https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    统一使用opencv读, 指定读取格式cv2.IMREAD_UNCHANGED
    
    """
    
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth.dtype == np.uint16:
        return depth
    depth_shape = depth.shape
    if len(depth_shape) == 3:
        assert (depth[...,0] == depth[...,1]).all()
        assert (depth[...,0] == depth[...,2]).all()
        depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
    elif len(depth_shape) == 2:
        pass
    else:
        raise NotImplementedError
    depth = np.clip(depth, 0, 2**16 - 1)
    uint16_depth = depth.astype(np.uint16)
    assert (depth == uint16_depth).all(), print(depth_path+" wrong!")
    return uint16_depth