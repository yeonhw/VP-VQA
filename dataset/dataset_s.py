import os
import re
import json
import numpy as np
import subprocess
import torch
from torch.utils.data import DataLoader, Dataset
import logging
import py360convert
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor


class CropSegment(object):
    r"""
    Crop a clip along the spatial axes, i.e. h, w
    DO NOT crop along the temporal axis

    args:
        size_x: horizontal dimension of a segment
        size_y: vertical dimension of a segment
        stride_x: horizontal stride between segments
        stride_y: vertical stride between segments
    return:
        clip (tensor): dim = (N, C, D, H=size_y, W=size_x). N are segments number by applying sliding window with given window size and stride
    """

    def __init__(self, size_x, size_y, stride_x, stride_y):

        self.size_x = size_x
        self.size_y = size_y
        self.stride_x = stride_x
        self.stride_y = stride_y

    def __call__(self, clip):

        # input dimension [C, D, H, W]
        channel = clip.shape[0]
        depth = clip.shape[1]

        clip = clip.unfold(2, self.size_x, self.stride_x)
        clip = clip.unfold(3, self.size_y, self.stride_y)
        clip = clip.permute(2, 3, 0, 1, 4, 5)
        clip = clip.contiguous().view(-1, channel, depth, self.size_x, self.size_y)

        return clip

class CropSegment0_s(object):

    def __init__(self, size_x, size_y, stride_u, stride_v):

        self.size_x = size_x
        self.size_y = size_y
        self.stride_u = stride_u
        self.stride_v = stride_v

    def __call__(self, clip):

        # out1 = py360convert.e2p(clip, fov_deg=(90, 90), u_deg=0, v_deg=0,
        #                            out_hw=(360, 360), in_rot_deg=0, mode='bilinear')

        frames_frames = []  #每个视频33帧的crop

        # print(len(clip.shape)) # 4 torch.Size([1, 33, 720, 1440])
        for i in range(clip.shape[1]): # i=33 帧数 第一个维度：33
            clip_frame = clip[0, i] # 获取第 i 帧，形状为 (720, 1440)
            # print(len(clip_frame.shape)) # 2
            # print('shape for clip_frame:{shape}'.format(shape=clip_frame.shape))  # torch.Size([720, 1440])
            cropped_frames = []  # 每幀視頻的crop
            with open('./tool/test.txt', "r") as f:
                j = 0
                for line in f:
                    j = j + 1
                    coord = eval(line.strip())  # 解析坐标
                    x, y = coord
                    crop = py360convert.e2p(clip_frame, (90, 90), x, y, (self.size_x, self.size_y), in_rot_deg=0, mode='bilinear') # ndarray：(112,112)
                    # print('type of crop:{type}'.format(type=crop.dtype))
                    # print(f'type of crop:{type(crop)}') # type of crop:<class 'numpy.ndarray'> float 数组
                    # print(f'length of crop:{len(crop)}')
                    # print(f'length of crop[0]:{len(crop[0])}')
                    # print(f'crop:{crop}')
                    plt.imshow(crop, cmap='gray')  # 如果是灰度图像，使用 cmap='gray'
                    plt.colorbar()  # 显示颜色条
                    print(x,y)
                    plt.show()  # 显示图像
                    crop = crop.reshape(1, 1, self.size_x, self.size_y)
                    # print(f'type of crop = crop.reshape(1, 1, self.size_x, self.size_y):{type(crop)}') #<class 'numpy.ndarray'>
                    # print(f'shape of crop = crop.reshape(1, 1, self.size_x, self.size_y):{crop.shape}')#(1, 1, 112, 112)
                    cropped_frames.append(crop) # 应该是145个crop
                cropped_frames = np.concatenate(cropped_frames, axis=1)  #
                cropped_frames = cropped_frames.reshape(1, 1, j, self.size_x, self.size_y)
                # print('length of ret:{length}'.format(length=len(cropped_frames)))#
            frames_frames.append(cropped_frames)
        frames_frames = np.concatenate(frames_frames, axis=1)
        frames_frames = torch.from_numpy(np.asarray(frames_frames))
        frames_frames= frames_frames.permute(2, 0, 1, 3, 4)
        # print(f'shape of frames_frames:{frames_frames.shape}')


        return cropped_frames # 目标形状： torch.Size([136, 1, 33, 112, 112])

        # return clip

# print调试时用这个
# class CropSegment_s:
#     def __init__(self, size_x, size_y, stride_u, stride_v):
#         self.size_x = size_x
#         self.size_y = size_y
#         self.stride_u = stride_u
#         self.stride_v = stride_v
#
#         # 预加载坐标
#         with open('./tool/coordinates2.txt', "r") as f:
#             self.coordinates = [tuple(map(int, line.strip()[1:-1].split(','))) for line in f]
#         self.num_crops = len(self.coordinates)  # 145 个 crop
#
#     def __call__(self, clip):
#         num_frames = clip.shape[1]  # 获取帧数（33）
#
#         # 预分配 `frames_frames` 以避免 append 开销
#         frames_frames = np.empty((self.num_crops, 1, num_frames, self.size_x, self.size_y), dtype=np.float32)
#         # logging.info("1")
#         for i in range(num_frames):
#             clip_frame = clip[0, i]  # 形状 (720, 1440)
#             # logging.info("2")
#             # plt.imshow(clip_frame, cmap='gray')  # 假设是灰度图，彩色图需修改 reshape 和cmap 参数
#             # plt.title(f"Frame {i}")
#             # plt.axis('off')
#             # plt.show()
#
#             # 预分配 `cropped_frames`，减少 `append` 操作
#             cropped_frames = np.empty((1, self.num_crops, self.size_x, self.size_y), dtype=np.float32)
#             # logging.info("3")
#
#
#             for j, (x, y) in enumerate(self.coordinates):
#                 cropped_frames[0, j] = py360convert.e2p(
#                     clip_frame, (90, 90), x, y, (self.size_x, self.size_y), in_rot_deg=0, mode='bilinear'
#                 )
#             # logging.info("4")
#                 # plt.imshow(cropped_frames[0, j], cmap='gray')  # 假设是灰度图，彩色图需修改 reshape 和cmap 参数
#                 # plt.title(f"cropped_frames[0, {j}]")
#                 # plt.axis('off')
#                 # plt.show()
#
#             # 存入 `frames_frames`
#             frames_frames[:, 0, i, :, :] = cropped_frames[0]
#             # logging.info("5")
#
#         frames_frames = torch.from_numpy(frames_frames)
#         # logging.info("6")
#         # frames_frames = frames_frames.permute(0, 2, 1, 3, 4)  # 目标形状：[num_crops, 1, num_frames, 112, 112]
#         frames_frames = frames_frames.permute(0, 1, 2, 3, 4)
#         # logging.info("7")
#         # print(f'shape of frames_frames:{frames_frames.shape}')
#
#         return frames_frames

class CropSegment_s:
    def __init__(self, size_x, size_y, stride_u, stride_v):
        self.size_x = size_x
        self.size_y = size_y
        self.stride_u = stride_u
        self.stride_v = stride_v

        with open('./tool/coordinates2.txt', "r") as f:
            self.coordinates = [tuple(map(int, line.strip()[1:-1].split(','))) for line in f]
        self.num_crops = len(self.coordinates)  # 145 个 crop

    def __call__(self, clip):
        num_frames = clip.shape[1]  # 获取帧数（33）

        frames_frames = np.empty((self.num_crops, 1, num_frames, self.size_x, self.size_y), dtype=np.float32)
        for i in range(num_frames):
            clip_frame = clip[0, i]  # 形状 (720, 1440)
            cropped_frames = np.empty((1, self.num_crops, self.size_x, self.size_y), dtype=np.float32)

            for j, (x, y) in enumerate(self.coordinates):
                cropped_frames[0, j] = py360convert.e2p(
                    clip_frame, (90, 90), x, y, (self.size_x, self.size_y), in_rot_deg=0, mode='bilinear'
                )

            frames_frames[:, 0, i, :, :] = cropped_frames[0]
        frames_frames = torch.from_numpy(frames_frames)
        frames_frames = frames_frames.permute(0, 1, 2, 3, 4)

        return frames_frames




class CropSegment1_s:
    def __init__(self, size_x, size_y, stride_u, stride_v, num_workers=4):
        self.size_x = size_x
        self.size_y = size_y
        self.stride_u = stride_u
        self.stride_v = stride_v
        self.num_workers = num_workers

        with open('./tool/coordinates2.txt', "r") as f:
            self.coordinates = [tuple(map(int, line.strip()[1:-1].split(','))) for line in f]
        self.num_crops = len(self.coordinates)  # 145 个 crop

    def process_frame(self, frame):
        """ 并行处理单个帧 """
        cropped_frames = np.array([
            py360convert.e2p(frame, (90, 90), x, y, (self.size_x, self.size_y), in_rot_deg=0, mode='bilinear')
            for x, y in self.coordinates
        ], dtype=np.float32)
        return cropped_frames

    def __call__(self, clip):
        num_frames = clip.shape[1]  # 获取帧数（33）
        # print(num_frames)
        frames_list = Parallel(n_jobs=self.num_workers)(
            delayed(self.process_frame)(clip[0, i]) for i in range(num_frames))

        # 合并结果
        frames_frames = np.stack(frames_list, axis=2)  # (num_crops, size_x, size_y, num_frames)
        frames_frames = frames_frames[:, np.newaxis, :, :, :]  # (num_crops, 1, num_frames, size_x, size_y)
        # print("shape for frames_frames = frames_frames[:, np.newaxis, :, :, :]:{shape}".format(shape=frames_frames.shape))
        frames_frames = torch.from_numpy(frames_frames)  # 先转换为 Tensor
        frames_frames = frames_frames.permute(0, 1, 3, 2, 4)  # 调整维度
        # print(
            # "shape for frames_frames = frames_frames[:, np.newaxis, :, :, :]:{shape}".format(shape=frames_frames.shape))
        return frames_frames


class CropSegment20_s:
    def __init__(self, size_x, size_y, stride_u, stride_v, num_workers=4):
        self.size_x = size_x
        self.size_y = size_y
        self.stride_u = stride_u
        self.stride_v = stride_v
        self.num_workers = num_workers if num_workers else os.cpu_count()

        # 使用 numpy 读取，提高效率
        self.coordinates = np.loadtxt('./tool/coordinates2.txt', delimiter=',', dtype=int)
        self.num_crops = len(self.coordinates)

    def process_frame(self, frame):
        """ 处理单帧 """
        return np.stack([
            py360convert.e2p(frame, (90, 90), x, y, (self.size_x, self.size_y), in_rot_deg=0, mode='bilinear')
            for x, y in self.coordinates
        ], axis=0).astype(np.float32)

    def __call__(self, clip):
        num_frames = clip.shape[1]  # 获取帧数（33）
        frames_list = Parallel(n_jobs=self.num_workers, prefer="threads")(
            delayed(self.process_frame)(clip[0, i]) for i in range(num_frames)
        )

        # NumPy 直接处理数据维度，减少转换开销
        frames_frames = np.stack(frames_list, axis=2)  # (num_crops, size_x, size_y, num_frames)
        frames_frames = frames_frames[:, np.newaxis, :, :, :]  # (num_crops, 1, num_frames, size_x, size_y)

        return torch.from_numpy(frames_frames).permute(0, 1, 3, 2, 4)

class CropSegment2_s:
    def __init__(self, size_x, size_y, stride_u, stride_v, num_workers=4):
        self.size_x = size_x
        self.size_y = size_y
        self.stride_u = stride_u
        self.stride_v = stride_v
        self.num_workers = num_workers if num_workers else os.cpu_count()

        # 使用 numpy 读取，提高效率
        self.coordinates = np.loadtxt('./tool/sampling_coords_15overlap.txt', dtype=float) # delimiter=',',
        self.num_crops = len(self.coordinates)

    def process_frame(self, frame):
        """ 处理单帧 """
        return np.stack([
            py360convert.e2p(frame, (30, 30), x, y, (self.size_x, self.size_y), in_rot_deg=0, mode='bilinear')
            for x, y in self.coordinates
        ], axis=0).astype(np.float32)

    def __call__(self, clip):
        num_frames = clip.shape[1]  # 获取帧数（33）
        frames_list = Parallel(n_jobs=self.num_workers, prefer="threads")(
            delayed(self.process_frame)(clip[0, i]) for i in range(num_frames)
        )

        # NumPy 直接处理数据维度，减少转换开销
        frames_frames = np.stack(frames_list, axis=2)  # (num_crops, size_x, size_y, num_frames)
        frames_frames = frames_frames[:, np.newaxis, :, :, :]  # (num_crops, 1, num_frames, size_x, size_y)


        return torch.from_numpy(frames_frames).permute(0, 1, 3, 2, 4)


import cv2
from pathlib import Path
import numpy as np


def visualize_patch(ref, save_dir, patch_idx=0, frame_idx=0):
    """
    直接保存 ref_img 到文件夹，按 001.png, 002.png 自动命名
    ref: torch tensor [num_crops, 1, T, H, W]
    save_dir: 保存目录
    patch_idx: 要保存的 patch 索引
    frame_idx: 要保存的时间帧索引
    """
    # 确保目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 提取 ref_img (numpy array)
    ref_img = ref[patch_idx, 0, frame_idx].detach().cpu().numpy()

    # 归一化到 [0, 255]（如果 ref_img 是 float32 且值在 [0,1] 或 [-1,1]）
    if ref_img.dtype == np.float32:
        ref_img = (ref_img * 255).clip(0, 255).astype(np.uint8)

    # 自动生成文件名（001.png, 002.png, ...）
    existing_files = list(Path(save_dir).glob("*.png"))
    next_num = len(existing_files) + 1
    save_path = Path(save_dir) / f"{next_num:03d}.png"

    # 保存图像（OpenCV）
    cv2.imwrite(str(save_path), ref_img)


class CropSegment22_s:
    def __init__(self, size_x, size_y, stride_u, stride_v, num_workers=4):
        self.size_x = size_x
        self.size_y = size_y
        self.stride_u = stride_u
        self.stride_v = stride_v
        self.num_workers = num_workers if num_workers else os.cpu_count()

        # 使用 numpy 读取，提高效率
        self.coordinates = np.loadtxt('./tool/coordinates4.txt', delimiter=',', dtype=int)
        self.num_crops = len(self.coordinates)

    def process_frame(self, frame):
        """ 处理单帧 """
        return np.stack([
            py360convert.e2p(frame, (45, 45), x, y, (self.size_x, self.size_y), in_rot_deg=0, mode='bilinear')
            for x, y in self.coordinates
        ], axis=0).astype(np.float32)

    def __call__(self, clip):
        num_frames = clip.shape[1]  # 获取帧数（33）
        frames_list = Parallel(n_jobs=self.num_workers, prefer="threads")(
            delayed(self.process_frame)(clip[0, i]) for i in range(num_frames)
        )

        # NumPy 直接处理数据维度，减少转换开销
        frames_frames = np.stack(frames_list, axis=2)  # (num_crops, size_x, size_y, num_frames)
        frames_frames = frames_frames[:, np.newaxis, :, :, :]  # (num_crops, 1, num_frames, size_x, size_y)
        frames_frames = torch.from_numpy(frames_frames).permute(0, 1, 3, 2, 4)
        for patchnum in range(0, 76):
            visualize_patch(frames_frames, save_dir= './crops/4fov26', patch_idx=patchnum, frame_idx=0)

        return frames_frames


class VideoDataset(Dataset):
    r"""
    A Dataset for a folder of videos

    args:
        subj_score_file (str): path to the subjective score file. It contains train/test split, ref list, dis list, fps list and mos list
        directory (str): the path to the directory containing all videos
        mode (str, optional): determines whether to read train/test data
        channel (int, optional): number of channels of a sample
        size_x: horizontal dimension of a segment
        size_y: vertical dimension of a segment
        stride_x: horizontal stride between segments
        stride_y: vertical stride between segments
    """

    def __init__(self, subj_score_file, directory, mode='train', channel=1, size_x=112, size_y=112, stride_u=30, stride_v=15, transform=None):

        with open(subj_score_file, "r") as f:
            data = json.load(f)
        self.video_dir = directory
        data = data[mode]
        self.ref = data['ref']
        self.dis = data['dis']
        self.label = data['mos']
        self.framerate = data['fps']
        self.frame_height = data['height']
        self.frame_width = data['width']
        self.channel = channel
        self.size_x = size_x
        self.size_y = size_y
        # self.stride_x = stride_x
        # self.stride_y = stride_y
        self.stride_u = stride_u
        self.stride_v = stride_v
        self.transform = transform

    def __getitem__(self, index):

        ref = os.path.join(self.video_dir, self.ref[index])
        ref1 = ref
        dis = os.path.join(self.video_dir, self.dis[index])
        dis1 = dis
        label = float(self.label[index])
        framerate = int(self.framerate[index])
        frame_height = int(self.frame_height[index])
        frame_width = int(self.frame_width[index])

        if framerate <= 30:
            stride_t = 2
        elif framerate <= 60:
            stride_t = 4
        else:
            raise ValueError('Unsupported fps')

        if ref.endswith(('.YUV', '.yuv')):
            ref = self.load_1yuv(ref, frame_height, frame_width, stride_t)
        elif ref.endswith(('.mp4')):
            ref = self.load_encode(ref, frame_height, frame_width, stride_t)
        else:
            raise ValueError('Unsupported video format')
        # logging.info("index:{index}, ref:{ref}, over load_1yuv".format(index=index, ref=ref1))



        if dis.endswith(('.YUV', '.yuv')):
            dis = self.load_1yuv(dis, frame_height, frame_width, stride_t)
        elif dis.endswith(('.mp4')):
            dis = self.load_encode(dis, frame_height, frame_width, stride_t)
        else:
            raise ValueError('Unsupported video format')

        spherical_crop = CropSegment2_s(self.size_x, self.size_y, self.stride_u, self.stride_v)
        ref = spherical_crop(ref)
        # logging.info("index:{index}, ref:{ref}, over spherical_crop".format(index=index, ref=ref1))
        # print("shape for ref = spatial_crop(ref):{shape}".format(shape=ref.shape))
        dis = spherical_crop(dis)
        # logging.info("index:{index}, dis:{dis}, over spherical_crop".format(index=index, dis=dis1))
        # print("shape for dis = spatial_crop(dis):{shape}".format(shape=dis.shape))
        # for patchnum in range(0, 76):
        #     visualize_patch(ref, dis, patch_idx=patchnum, frame_idx=0)


        ref = torch.from_numpy(np.asarray(ref))
        dis = torch.from_numpy(np.asarray(dis))
        label = torch.from_numpy(np.asarray(label))
        # print("shape for final_dis:{shape}".format(shape=dis.shape))
        # logging.info("index:{index}, ref:{ref}, over dataset".format(index=index, ref=ref1))

        return ref, dis, label

    def load_yuv(self, file_path, frame_height, frame_width, stride_t, start=0):
        r"""
        Load frames on-demand from raw video, currently supports only yuv420p

        args:
            file_path (str): path to yuv file
            frame_height
            frame_width
            stride_t (int): sample the 1st frame from every stride_t frames
            start (int): index of the 1st sampled frame
        return:
            ret (tensor): contains sampled frames (Y channel). dim = (C, D, H, W)
        """

        bytes_per_frame = int(frame_height * frame_width * 1.5)
        frame_count = os.path.getsize(file_path) / bytes_per_frame

        ret = []
        count = 0

        with open(file_path, 'rb') as f:
            while count < frame_count:
                if count % stride_t == 0:
                    offset = count * bytes_per_frame
                    f.seek(offset, 0)
                    frame = f.read(frame_height * frame_width)
                    frame = np.frombuffer(frame, "uint8")
                    frame = frame.astype('float32') / 255.
                    frame = frame.reshape(1, 1, frame_height, frame_width)
                    ret.append(frame)
                count += 1

        ret = np.concatenate(ret, axis=1)
        ret = torch.from_numpy(np.asarray(ret))

        return ret

    def load_1yuv(self, file_path, frame_height, frame_width, stride_t, start=0):
        r"""
        Load frames on-demand from raw video, currently supports only yuv420p

        args:
            file_path (str): path to yuv file
            frame_height
            frame_width
            stride_t (int): sample the 1st frame from every stride_t frames
            start (int): index of the 1st sampled frame
        return:
            ret (tensor): contains sampled frames (Y channel). dim = (C, D, H, W)


        """

        bytes_per_frame = int(frame_height * frame_width * 1.5)  # 计算每帧的字节数
        frame_count = os.path.getsize(file_path) / bytes_per_frame # 计算视频一共有多少帧

        ret = []
        B = []
        # count = 0
        A = np.linspace(1, frame_count - 2, 11)
        A = np.array(A).astype(dtype=int).tolist()
        for centre_idx in A:
            # for i in range(centre_idx-7, centre_idx+1):
            #     B.append(i)
            B = B + list(range(centre_idx - 1, centre_idx + 2, 1)) # 抽到的11帧再加上前后各1帧

        with open(file_path, 'rb') as f: # ref or dis 的路径
            for count in B:
            # while count < frame_count:
            #     if count % stride_t == 0: # 按照stride_t来进行抽帧
                offset = count * bytes_per_frame
                f.seek(offset, 0)
                frame = f.read(frame_height * frame_width)
                frame = np.frombuffer(frame, "uint8")
                # 原始像素数据可视化（未归一化的 uint8）

                frame = frame.astype('float32') / 255.
                frame = frame.reshape(1, 1, frame_height, frame_width)
                ret.append(frame)
                # count += 1
        # print('length of ret:{length}'.format(length=len(ret))) #33
        ret = np.concatenate(ret, axis=1)  # 将 ret 中的所有帧数据沿第 1 维度（a xis=1）合并，形成一个更大的数组。
        # print('shape for ret:{length}'.format(length=len(ret)))
        # ret = torch.from_numpy(np.asarray(ret))  # 将合并后的 NumPy 数组转换为 PyTorch 张量。

        return ret

    def load_encode(self, file_path, frame_height, frame_width, stride_t, start=0):
        r"""
        Load frames on-demand from encode bitstream

        args:
            file_path (str): path to yuv file
            frame_height
            frame_width
            stride_t (int): sample the 1st frame from every stride_t frames
            start (int): index of the 1st sampled frame
        return:
            ret (array): contains sampled frames. dim = (C, D, H, W)
        """

        enc_path = file_path
        enc_name = re.split('/', enc_path)[-1]

        yuv_name = enc_name.replace('.mp4', '.yuv')
        yuv_path = os.path.join('/dockerdata/tmp/', yuv_name)
        cmd = "ffmpeg -y -i {src} -f rawvideo -pix_fmt yuv420p -vsync 0 -an {dst}".format(src=enc_path, dst=yuv_path)
        subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        ret = self.load_yuv(yuv_path, frame_height, frame_width, stride_t, start=0)

        return ret

    def __len__(self):
        return len(self.dis)


if __name__ == '__main__':

    root_dir = os.path.dirname(os.path.realpath(__file__))
    subj_score_file = os.path.join(root_dir, 'csiq_subj_score.json')
    video_dir = '/dockerdata/CSIQ_YUV'
    csiq_dataset = VideoDataset(subj_score_file, video_dir)
    print(len(csiq_dataset))
