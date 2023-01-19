import cv2
import os
import re


import numpy as np
import h5py
import OpenEXR
import Imath


def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    channel_names = list(File.header()['channels'].keys())
    if not all(x in ["G", "R", "B"] for x in channel_names):
        return

    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if (CNum > 1):
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.fromstring(File.channel(c, PixType),
                            dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1], Size[0], CNum), dtype=np.float32)
    if (CNum == 1):
        hdr[:, :, 0] = np.reshape(Pixels[0], (Size[1], Size[0]))
    else:
        hdr[:, :, 0] = np.reshape(Pixels[0], (Size[1], Size[0]))
        hdr[:, :, 1] = np.reshape(Pixels[1], (Size[1], Size[0]))
        hdr[:, :, 2] = np.reshape(Pixels[2], (Size[1], Size[0]))
    return hdr


def load_pfm(file):
    """
    ref:https://github.com/YoYo000/MVSNet/blob/4c4aa5e2336a214e4bde2de31c9a46f55a8150c5/mvsnet/preprocess.py#L175
    """
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = file.readline().decode('UTF-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float((file.readline()).decode('UTF-8').rstrip())
    if scale < 0:  # little-endian
        data_type = '<f'
    else:
        data_type = '>f'  # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data


def load_exr(filename):
    """
    ref:https://github.com/blackjack2015/IRS/blob/4fbbf8be8e1c0b18e978beb81b054c610c4fad73/dataloader/EXRloader.py#L37
    """
    hdr = exr2hdr(filename)
    if hdr is None:
        return
    h, w, c = hdr.shape
    if c == 1:
        hdr = np.squeeze(hdr)
    return hdr


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
        assert (depth[..., 0] == depth[..., 1]).all()
        assert (depth[..., 0] == depth[..., 2]).all()
        depth = depth[..., 0]
    elif len(depth_shape) == 2:
        pass
    else:
        raise NotImplementedError
    uint16_depth = depth.astype(np.uint16)
    assert (depth == uint16_depth).all()
    assert uint16_depth.max() < 2**16
    assert uint16_depth.min() >= 0

    return uint16_depth


def process_depth(args, ori_depth_path, ouput_depth_path, mask_path):
    """
    关于深度图的处理
    requirement: 单通道+uint16
    其他: 待扩展
    """
    mask = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    # 统一处理depth到uint16类型
    if ori_depth_path.endswith("jpg") or ori_depth_path.endswith("png"):
        depth = cv2.imread(ori_depth_path, cv2.IMREAD_UNCHANGED)
        if mask is not None:
            depth = depth*(mask == 255)
        if args.dataset == "HR-WSI":
            output_depth = np.clip(1.0 / (depth + 1e-6) * 65535, 0, 2**16 - 1).astype(np.uint16)
        else:
            output_depth = pretty_depth(depth)

    elif ori_depth_path.endswith("h5"):
        hdf5_file_read = h5py.File(ori_depth_path, 'r')
        depth = hdf5_file_read.get('/depth')

        output_depth = np.array(depth)
        hdf5_file_read.close()
        output_depth = np.clip(output_depth * 1000, 0, 2**16 - 1).astype(np.uint16)

        # if np.sum(depth > 1e-8) > 10:
        #     depth[ depth > np.percentile(depth[depth > 1e-8], 98)] = 0
        #     depth[ depth < np.percentile(depth[depth > 1e-8], 1)] = 0

    elif ori_depth_path.endswith("npy"):
        output_depth = np.load(ori_depth_path)
        output_depth = np.clip(output_depth, 0, 50)
        output_depth = (output_depth * 1000).astype("uint16")

    elif ori_depth_path.endswith("exr"):
        output_depth = load_exr(ori_depth_path)

    else:
        raise NotImplementedError
    # 保存为png无损格式
    assert output_depth is not None
    cv2.imwrite(os.path.join(args.output_path, ouput_depth_path), output_depth)
