import cv2
import os
import numpy as np
import h5py


def pretty_depth(depth):
    """
    用PIL是无法读取uint16格式的数据的,详情见:
    https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    统一使用opencv读, 指定读取格式cv2.IMREAD_UNCHANGED
    Return:
        单通道
        uint16
    """
    
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
    assert (depth == uint16_depth).all()
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
    if ori_depth_path.endswith("jpg") or ori_depth_path.endswith("png"):
        depth = cv2.imread(ori_depth_path, cv2.IMREAD_UNCHANGED)
        output_depth = pretty_depth(depth)
        
    
    elif ori_depth_path.endswith("h5"):
        hdf5_file_read = h5py.File(ori_depth_path,'r')
        depth = hdf5_file_read.get('/depth')
        
        output_depth = np.array(depth)
        hdf5_file_read.close()
        
        # if np.sum(depth > 1e-8) > 10:
        #     depth[ depth > np.percentile(depth[depth > 1e-8], 98)] = 0
        #     depth[ depth < np.percentile(depth[depth > 1e-8], 1)] = 0
    
    elif ori_depth_path.endswith("npy"):
        output_depth = np.load(ori_depth_path)
    
    else:
        raise NotImplementedError
    # 保存为png无损格式
    cv2.imwrite(os.path.join(args.output_path, ouput_depth_path), output_depth)