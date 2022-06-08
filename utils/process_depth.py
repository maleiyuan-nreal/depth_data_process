import cv2
import os
import numpy as np


def pretty_depth(depth_path):
    """
    用PIL是无法读取uint16格式的数据的,详情见:
    https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    统一使用opencv读, 指定读取格式cv2.IMREAD_UNCHANGED
    Return:
        单通道
        uint16
    """
    
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = np.clip(depth, 0, 2**16 - 1)
    
    depth_shape = depth.shape
    if len(depth_shape) == 3:
        assert (depth[...,0] == depth[...,1]).all()
        assert (depth[...,0] == depth[...,2]).all()
        depth = depth[...,0]
    elif len(depth_shape) == 2:
        pass
    else:
        raise NotImplementedError
    uint16_depth = depth.astype(np.uint16)
    
    assert (depth == uint16_depth).all(), print(depth_path+" wrong!")
    assert uint16_depth.max() < 2**16
    assert uint16_depth.min() >= 0
    
    return uint16_depth


def process_depth(args, ori_depth_path, ouput_depth_path):
    """
    关于深度图的处理
    requirement: 单通道+uint16
    其他: 待扩展
    """
    # 统一处理depth到uint16类型
    uint16_depth = pretty_depth(ori_depth_path)
    # 保存为png无损格式
    cv2.imwrite(os.path.join(args.output_path, ouput_depth_path), uint16_depth)