# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications
import os
import random
# import OpenEXR
import json
import h5py
import numpy as np
from copy import deepcopy
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from src.utils.dataloader_sparse_depth import get_sparse_depth, get_grid_coordinate, get_rotate_translate
from src.models.penet_util import AddCoordsNp

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})



def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


def to_tensor(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError(
            'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # img = torch.from_numpy(pic.copy().transpose((2, 0, 1)))
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)

    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


class TT_DS_compose_paper(object):
    def __init__(self, args, mode):
        if mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False)


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = deepcopy(args)
        self.args.mode = mode
        if mode == 'online_eval':
            md = 'test'

        if md == 'test':
            fname = args.filenames_file_eval
            with open(fname, 'r') as fh:
                pairs = []
                for line in fh:
                    line = line.rstrip()
                    words = line.split()
                    pairs.append((words[0], words[1]))

                self.sample_list = pairs

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        self.position = AddCoordsNp(288, 512)
        self.grid_coordinates = get_grid_coordinate()


    def __getitem__(self, idx):
        if self.args.mode == 'online_eval':
            input, target = self.sample_list[idx]  #   input = image       target = depth_gt
            target = os.path.join(self.args.data_path_eval, target)

        if idx == 0 or idx == len(self.sample_list) - 1:

            before_rgb,before_depth = self.sample_list[idx]
            before_rgb = os.path.join(self.args.data_path_eval, before_rgb)
            before_depth = os.path.join(self.args.data_path_eval, before_depth)

            now_rgb, now_depth = self.sample_list[idx]
            now_rgb = os.path.join(self.args.data_path_eval, now_rgb)
            now_depth = os.path.join(self.args.data_path_eval, now_depth)

            next_rgb, next_depth = self.sample_list[idx]
            next_rgb = os.path.join(self.args.data_path_eval, next_rgb)
            next_depth = os.path.join(self.args.data_path_eval, next_depth)

        else:
            before_rgb,before_depth = self.sample_list[idx - 1]
            before_rgb = os.path.join(self.args.data_path_eval, before_rgb)
            before_depth = os.path.join(self.args.data_path_eval, before_depth)

            now_rgb,now_depth = self.sample_list[idx]
            now_rgb = os.path.join(self.args.data_path_eval, now_rgb)
            now_depth = os.path.join(self.args.data_path_eval, now_depth)

            next_rgb,next_depth = self.sample_list[idx + 1]
            next_rgb = os.path.join(self.args.data_path_eval, next_rgb)
            next_depth = os.path.join(self.args.data_path_eval, next_depth)

        before_rgb_in = cv2.imread(before_rgb)
        before_image = cv2.cvtColor(before_rgb_in, cv2.COLOR_BGR2RGB)
        now_rgb_in = cv2.imread(now_rgb)
        now_image = cv2.cvtColor(now_rgb_in, cv2.COLOR_BGR2RGB)
        next_rgb_in = cv2.imread(next_rgb)
        next_image = cv2.cvtColor(next_rgb_in, cv2.COLOR_BGR2RGB)

        before_depth_gt = np.load(before_depth)
        now_depth_gt = np.load(now_depth)
        next_depth_gt = np.load(next_depth)

        before_depth_gt[before_depth_gt > self.args.max_depth] = self.args.max_depth #original
        before_depth_gt[before_depth_gt < self.args.min_depth] = self.args.min_depth
        now_depth_gt[now_depth_gt > self.args.max_depth] = self.args.max_depth #original
        now_depth_gt[now_depth_gt < self.args.min_depth] = self.args.min_depth
        next_depth_gt[next_depth_gt > self.args.max_depth] = self.args.max_depth #original
        next_depth_gt[next_depth_gt < self.args.min_depth] = self.args.min_depth

        before_image = Image.fromarray(before_image, mode='RGB')
        now_image = Image.fromarray(now_image, mode='RGB')
        next_image = Image.fromarray(next_image, mode='RGB')

        before_depth_gt = Image.fromarray(before_depth_gt.astype('float32'), mode='F')
        now_depth_gt = Image.fromarray(now_depth_gt.astype('float32'), mode='F')
        next_depth_gt = Image.fromarray(next_depth_gt.astype('float32'), mode='F')

        before_image = before_image.resize((512, 288))
        now_image = now_image.resize((512, 288))
        next_image = next_image.resize((512, 288))

        before_depth_gt = before_depth_gt.resize((512, 288))
        now_depth_gt = now_depth_gt.resize((512, 288))
        next_depth_gt = next_depth_gt.resize((512, 288))

        before_image = np.array(before_image, dtype=np.float32)
        now_image = np.array(now_image, dtype=np.float32)
        next_image = np.array(next_image, dtype=np.float32)

        before_depth_gt = np.array(before_depth_gt, dtype=np.float32)
        before_depth_gt = np.expand_dims(before_depth_gt, axis=2)

        now_depth_gt = np.array(now_depth_gt, dtype=np.float32)
        now_depth_gt = np.expand_dims(now_depth_gt, axis=2)

        next_depth_gt = np.array(next_depth_gt, dtype=np.float32)
        next_depth_gt = np.expand_dims(next_depth_gt, axis=2)

        image = []
        image.append(before_image)
        image.append(now_image)
        image.append(next_image)
        image = np.array(image,dtype=np.float32)

        depth_gt = []
        depth_gt.append(before_depth_gt)
        depth_gt.append(now_depth_gt)
        depth_gt.append(next_depth_gt)
        depth_gt = np.array(depth_gt)

        image[0] = np.array(image[0], dtype=np.float32) / 255.0
        image[1] = np.array(image[1], dtype=np.float32) / 255.0
        image[2] = np.array(image[2], dtype=np.float32) / 255.0

        fname = self.sample_list[idx][0]
        image_path = fname[fname.rfind('/') + 1:].replace('h5', 'jpg')
        image_folder = fname[:fname.rfind('/')]

        sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': True,
                  'image_path': image_path, 'image_folder': image_folder}

        if self.transform:
            sample = self.transform(sample)

        sample['path'] = target
        sparse_depth_0 = get_sparse_depth(torch.from_numpy(get_rotate_translate(depth_gt[0]).transpose(2, 0, 1)),torch.from_numpy(image[0].transpose(2, 0, 1)))
        sparse_depth_1 = get_sparse_depth(torch.from_numpy(get_rotate_translate(depth_gt[1]).transpose(2, 0, 1)),torch.from_numpy(image[1].transpose(2, 0, 1)))
        sparse_depth_2 = get_sparse_depth(torch.from_numpy(get_rotate_translate(depth_gt[2]).transpose(2, 0, 1)),torch.from_numpy(image[2].transpose(2, 0, 1)))

        sparse_depth = torch.stack([sparse_depth_0, sparse_depth_1, sparse_depth_2], dim=0)

        grid_depth = torch.zeros_like(sparse_depth)

        fr = torch.zeros(1200, 4)
        mask = torch.zeros(1200)
        fh = torch.zeros(1200, 16)  # [patch_num, 16]
        patch_info = torch.zeros(1200, 16)
        pos = torch.from_numpy(self.position.call()).permute(2, 0, 1)

        # for conv kernel = [4,6,8], pad/kernel, max_patch/kernel, all_patch/kernel
        sample['additional'] = {
            'hist_data': fh.to(torch.float),
            'rect_data': fr.to(torch.float),
            'mask': mask,
            'patch_info': patch_info,
            'image_ori': to_tensor(image[1]),
            'sparse_depth': sparse_depth,
            'position': pos,
            'grid_depth': grid_depth,
        }

        return sample

    def __len__(self):
        return len(self.sample_list)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image = sample['image'] # (480,640,3)
        image_0 = self.to_tensor(image[0]) # (3,480,640)
        image_0 = self.normalize(image_0)

        image_1 = self.to_tensor(image[1])
        image_1 = self.normalize(image_1)

        image_2 = self.to_tensor(image[2])
        image_2 = self.normalize(image_2)

        new_image = torch.stack([image_0,image_1,image_2],dim=0)

        depth = sample['depth']

        has_valid_depth = sample['has_valid_depth']
        a = self.to_tensor(depth[0])
        b = self.to_tensor(depth[1])
        c = self.to_tensor(depth[2])
        new_depth = torch.stack([a, b, c], dim=0)

        return {'image': new_image, 'depth': new_depth, 'has_valid_depth': has_valid_depth,
                'image_path': sample['image_path'], 'image_folder': sample['image_folder']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            # img = torch.from_numpy(pic.copy().transpose((2, 0, 1)))
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)  # 480*640*3 -> 640,480,3

        img = img.transpose(0, 1).transpose(0, 2).contiguous() # 640,480,3 -> 480,640,3 -> 3,640,480
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
