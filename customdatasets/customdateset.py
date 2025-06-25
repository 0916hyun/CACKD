import os
import random
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms.functional as F
import random

label_color_map = {"Sky": [128, 128, 128],
                  "Building": [128, 0, 0],
                  "Pole": [192, 192, 128],
                  "Road": [128, 64, 128],
                  "Pavement": [0, 0, 192],
                  "Tree": [128, 128, 0],
                  "SignSymbol": [192, 128, 128],
                  "Fence": [64, 64, 128],
                  "Car": [64, 0, 128],
                  "Pedestrian": [64, 64, 0],
                  "Bicyclist": [0, 128, 192],
                  "Void": [0, 0, 0]}



class TrainLoadDatasetMulti(Dataset):
    def __init__(self, inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map):
        super(TrainLoadDatasetMulti, self).__init__()
        self.inp_dir = inp_dir
        self.tar_dir = tar_dir
        self.flare_dir = flare_dir
        self.cam_dir = cam_dir
        self.mask_dir = mask_dir
        self.fr_masked_dir = fr_masked_dir
        self.seg_dir = seg_dir
        self.label_color_map = label_color_map
        self.length = len(self.inp_dir)
        self.crop_size = 256

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index_ = index % self.length
        inp_path = self.inp_dir[index_]
        tar_path = self.tar_dir[index_]
        flare_path = self.flare_dir[index_]
        cam_path = self.cam_dir[index_]
        mask_path = self.mask_dir[index_]
        fr_masked_path = self.fr_masked_dir[index_]
        seg_path = self.seg_dir[index_]

        inp_img = Image.open(inp_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")
        flare_img = Image.open(flare_path).convert("RGB")
        cam_img = Image.open(cam_path).convert("L")
        mask_img = Image.open(mask_path).convert("L")
        fr_masked_img = Image.open(fr_masked_path).convert("RGB")
        seg_img = Image.open(seg_path).convert("L")

        i, j, h, w = T.RandomCrop.get_params(inp_img, output_size=(self.crop_size, self.crop_size))

        inp_img = F.crop(inp_img, i, j, h, w)
        tar_img = F.crop(tar_img, i, j, h, w)
        flare_img = F.crop(flare_img, i, j, h, w)
        fr_masked_img = F.crop(fr_masked_img, i, j, h, w)
        cam_img = F.crop(cam_img, i, j, h, w)
        mask_img = F.crop(mask_img, i, j, h, w)
        seg_img = F.crop(seg_img, i, j, h, w)

        if random.random() > 0.5:
            inp_img = F.hflip(inp_img)
            tar_img = F.hflip(tar_img)
            flare_img = F.hflip(flare_img)
            fr_masked_img = F.hflip(fr_masked_img)
            cam_img = F.hflip(cam_img)
            mask_img = F.hflip(mask_img)
            seg_img = F.hflip(seg_img)


        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)
        flare_img = F.to_tensor(flare_img)
        fr_masked_img = F.to_tensor(fr_masked_img)
        cam_img = F.to_tensor(cam_img)
        mask_img = F.to_tensor(mask_img)

        seg_img = torch.from_numpy(np.array(seg_img)).long()

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, flare_img, cam_img, mask_img, fr_masked_img, seg_img, filename


class TrainLoadDatasetMulti_k(Dataset):
    def __init__(self, inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map):
        super(TrainLoadDatasetMulti_k, self).__init__()
        self.inp_dir = inp_dir
        self.tar_dir = tar_dir
        self.flare_dir = flare_dir
        self.cam_dir = cam_dir
        self.mask_dir = mask_dir
        self.fr_masked_dir = fr_masked_dir
        self.seg_dir = seg_dir
        self.length = len(self.inp_dir)
        self.crop_size = (288, 960)
        self.label_color_map = {
            "Sky": [128, 128, 128],
            "Building": [128, 0, 0],
            "Pole": [192, 192, 128],
            "Road": [128, 64, 128],
            "Pavement": [0, 0, 192],
            "Tree": [128, 128, 0],
            "SignSymbol": [192, 128, 128],
            "Fence": [64, 64, 128],
            "Car": [64, 0, 128],
            "Pedestrian": [64, 64, 0],
            "Bicyclist": [0, 128, 192],
            "Void": [0, 0, 0],
        }

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index_ = index % self.length
        inp_path = self.inp_dir[index_]
        tar_path = self.tar_dir[index_]
        flare_path = self.flare_dir[index_]
        cam_path = self.cam_dir[index_]
        mask_path = self.mask_dir[index_]
        fr_masked_path = self.fr_masked_dir[index_]
        seg_path = self.seg_dir[index_]

        inp_img = Image.open(inp_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")
        flare_img = Image.open(flare_path).convert("RGB")
        cam_img = Image.open(cam_path).convert("L")
        mask_img = Image.open(mask_path).convert("L")
        fr_masked_img = Image.open(fr_masked_path).convert("RGB")
        seg_img = Image.open(seg_path).convert("RGB")

        i, j, h, w = T.RandomCrop.get_params(inp_img, output_size=self.crop_size)

        inp_img = F.crop(inp_img, i, j, h, w)
        tar_img = F.crop(tar_img, i, j, h, w)
        flare_img = F.crop(flare_img, i, j, h, w)
        fr_masked_img = F.crop(fr_masked_img, i, j, h, w)
        cam_img = F.crop(cam_img, i, j, h, w)
        mask_img = F.crop(mask_img, i, j, h, w)
        seg_img = F.crop(seg_img, i, j, h, w)

        if random.random() > 0.5:
            inp_img = F.hflip(inp_img)
            tar_img = F.hflip(tar_img)
            flare_img = F.hflip(flare_img)
            fr_masked_img = F.hflip(fr_masked_img)
            cam_img = F.hflip(cam_img)
            mask_img = F.hflip(mask_img)
            seg_img = F.hflip(seg_img)

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)
        flare_img = F.to_tensor(flare_img)
        fr_masked_img = F.to_tensor(fr_masked_img)
        cam_img = F.to_tensor(cam_img)
        mask_img = F.to_tensor(mask_img)


        seg_img = np.array(seg_img)
        h, w, _ = seg_img.shape
        seg_onehot = torch.zeros((len(self.label_color_map), h, w), dtype=torch.float32)

        for i, color in enumerate(self.label_color_map.values()):
            color_array = np.array(color, dtype=np.uint8)
            mask = np.all(seg_img == color_array, axis=-1)
            seg_onehot[i][mask] = 1

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, flare_img, cam_img, mask_img, fr_masked_img, seg_onehot, filename


class ValTestLoadDatasetMulti(Dataset):
    def __init__(self, inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map):
        super(ValTestLoadDatasetMulti, self).__init__()
        self.inp_dir = inp_dir
        self.tar_dir = tar_dir
        self.flare_dir = flare_dir
        self.cam_dir = cam_dir
        self.mask_dir = mask_dir
        self.fr_masked_dir = fr_masked_dir
        self.seg_dir = seg_dir
        self.label_color_map = label_color_map
        self.length = len(self.inp_dir)
        self.crop_size = 256

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index_ = index % self.length
        inp_path = self.inp_dir[index_]
        tar_path = self.tar_dir[index_]
        flare_path = self.flare_dir[index_]
        cam_path = self.cam_dir[index_]
        mask_path = self.mask_dir[index_]
        fr_masked_path = self.fr_masked_dir[index_]
        seg_path = self.seg_dir[index_]

        inp_img = Image.open(inp_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")
        flare_img = Image.open(flare_path).convert("RGB")
        cam_img = Image.open(cam_path).convert("L")
        mask_img = Image.open(mask_path).convert("L")
        fr_masked_img = Image.open(fr_masked_path).convert("RGB")
        seg_img = Image.open(seg_path).convert("L")

        i, j, h, w = T.RandomCrop.get_params(inp_img, output_size=(self.crop_size, self.crop_size))

        inp_img = F.crop(inp_img, i, j, h, w)
        tar_img = F.crop(tar_img, i, j, h, w)
        flare_img = F.crop(flare_img, i, j, h, w)
        fr_masked_img = F.crop(fr_masked_img, i, j, h, w)
        cam_img = F.crop(cam_img, i, j, h, w)
        mask_img = F.crop(mask_img, i, j, h, w)
        seg_img = F.crop(seg_img, i, j, h, w)

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)
        flare_img = F.to_tensor(flare_img)
        fr_masked_img = F.to_tensor(fr_masked_img)
        cam_img = F.to_tensor(cam_img)
        mask_img = F.to_tensor(mask_img)

        seg_img = torch.from_numpy(np.array(seg_img)).long()

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, flare_img, cam_img, mask_img, fr_masked_img, seg_img, filename

class ValTestLoadDatasetMulti_k(Dataset):
    def __init__(self, inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map):
        super(ValTestLoadDatasetMulti_k, self).__init__()
        self.inp_dir = inp_dir
        self.tar_dir = tar_dir
        self.flare_dir = flare_dir
        self.cam_dir = cam_dir
        self.mask_dir = mask_dir
        self.fr_masked_dir = fr_masked_dir
        self.seg_dir = seg_dir
        self.label_color_map = {
            "Sky": [128, 128, 128],
            "Building": [128, 0, 0],
            "Pole": [192, 192, 128],
            "Road": [128, 64, 128],
            "Pavement": [0, 0, 192],
            "Tree": [128, 128, 0],
            "SignSymbol": [192, 128, 128],
            "Fence": [64, 64, 128],
            "Car": [64, 0, 128],
            "Pedestrian": [64, 64, 0],
            "Bicyclist": [0, 128, 192],
            "Void": [0, 0, 0],
        }
        self.length = len(self.inp_dir)
        self.crop_size = (288, 960)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index_ = index % self.length
        inp_path = self.inp_dir[index_]
        tar_path = self.tar_dir[index_]
        flare_path = self.flare_dir[index_]
        cam_path = self.cam_dir[index_]
        mask_path = self.mask_dir[index_]
        fr_masked_path = self.fr_masked_dir[index_]
        seg_path = self.seg_dir[index_]

        inp_img = Image.open(inp_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")
        flare_img = Image.open(flare_path).convert("RGB")
        cam_img = Image.open(cam_path).convert("L")
        mask_img = Image.open(mask_path).convert("L")
        fr_masked_img = Image.open(fr_masked_path).convert("RGB")
        seg_img = Image.open(seg_path).convert("RGB")

        i, j, h, w = T.RandomCrop.get_params(inp_img, output_size=self.crop_size)

        inp_img = F.crop(inp_img, i, j, h, w)
        tar_img = F.crop(tar_img, i, j, h, w)
        flare_img = F.crop(flare_img, i, j, h, w)
        fr_masked_img = F.crop(fr_masked_img, i, j, h, w)
        cam_img = F.crop(cam_img, i, j, h, w)
        mask_img = F.crop(mask_img, i, j, h, w)
        seg_img = F.crop(seg_img, i, j, h, w)

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)
        flare_img = F.to_tensor(flare_img)
        fr_masked_img = F.to_tensor(fr_masked_img)
        cam_img = F.to_tensor(cam_img)
        mask_img = F.to_tensor(mask_img)

        seg_img = np.array(seg_img)
        h, w, _ = seg_img.shape
        seg_onehot = torch.zeros((len(self.label_color_map), h, w), dtype=torch.float32)

        for i, color in enumerate(self.label_color_map.values()):
            color_array = np.array(color, dtype=np.uint8)
            mask = np.all(seg_img == color_array, axis=-1)
            seg_onehot[i][mask] = 1

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, flare_img, cam_img, mask_img, fr_masked_img, seg_onehot, filename


class TrainSegDataset_c(Dataset):
    def __init__(self, inp_dir, seg_dir, crop_size=256):
        super(TrainSegDataset_c, self).__init__()
        self.inp_dir = inp_dir
        self.seg_dir = seg_dir
        self.length = len(inp_dir)
        self.crop_size = crop_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = index % self.length
        inp_path = self.inp_dir[idx]
        seg_path = self.seg_dir[idx]

        inp_img = Image.open(inp_path).convert('RGB')
        seg_img = Image.open(seg_path).convert('L')

        i, j, h, w = T.RandomCrop.get_params(
            inp_img, output_size=(self.crop_size, self.crop_size)
        )
        inp_img = F.crop(inp_img, i, j, h, w)
        seg_img = F.crop(seg_img, i, j, h, w)

        if random.random() > 0.5:
            inp_img = F.hflip(inp_img)
            seg_img = F.hflip(seg_img)

        inp_img = F.to_tensor(inp_img)
        seg_img = torch.from_numpy(np.array(seg_img)).long()

        filename = os.path.splitext(os.path.basename(inp_path))[0]
        return inp_img, seg_img, filename


class ValTestSegDataset_c(Dataset):
    def __init__(self, inp_dir, seg_dir, crop_size=256):
        super(ValTestSegDataset_c, self).__init__()
        self.inp_dir = inp_dir
        self.seg_dir = seg_dir
        self.length = len(inp_dir)
        self.crop_size = crop_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = index % self.length
        inp_path = self.inp_dir[idx]
        seg_path = self.seg_dir[idx]

        inp_img = Image.open(inp_path).convert('RGB')
        seg_img = Image.open(seg_path).convert('L')

        i, j, h, w = T.RandomCrop.get_params(
            inp_img, output_size=(self.crop_size, self.crop_size)
        )
        inp_img = F.crop(inp_img, i, j, h, w)
        seg_img = F.crop(seg_img, i, j, h, w)

        inp_img = F.to_tensor(inp_img)
        seg_img = torch.from_numpy(np.array(seg_img)).long()

        filename = os.path.splitext(os.path.basename(inp_path))[0]
        return inp_img, seg_img, filename

class MultiValTestLoadDatasetV2_c(Dataset):
    def __init__(self, inp_dir, seg_label_dir):
        super(MultiValTestLoadDatasetV2_c, self).__init__()
        self.inp_dir = inp_dir
        self.seg_label_dir = seg_label_dir
        self.length = len(inp_dir)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = index % self.length
        inp_path = self.inp_dir[idx]
        seg_path = self.seg_label_dir[idx]

        inp_img = Image.open(inp_path).convert('RGB')
        seg_img = Image.open(seg_path).convert('L')

        inp_img = F.to_tensor(inp_img)
        seg_img = torch.from_numpy(np.array(seg_img)).long()

        filename = os.path.splitext(os.path.basename(inp_path))[0]
        return inp_img, seg_img, filename

class TrainSegDataset_k(Dataset):
    def __init__(self, inp_dir, seg_dir, label_color_map, crop_size=(288, 960)):
        super(TrainSegDataset_k, self).__init__()
        self.inp_dir = inp_dir
        self.seg_dir = seg_dir
        self.length = len(inp_dir)
        self.crop_size = crop_size
        self.label_color_map = label_color_map

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = index % self.length
        inp_path = self.inp_dir[idx]
        seg_path = self.seg_dir[idx]
        inp_img = Image.open(inp_path).convert('RGB')
        seg_img = Image.open(seg_path).convert('RGB')
        i, j, h, w = T.RandomCrop.get_params(inp_img, output_size=self.crop_size)
        inp_img = F.crop(inp_img, i, j, h, w)
        seg_img = F.crop(seg_img, i, j, h, w)
        if random.random() > 0.5:
            inp_img = F.hflip(inp_img)
            seg_img = F.hflip(seg_img)
        inp_img = F.to_tensor(inp_img)
        seg_arr = np.array(seg_img)
        H, W, _ = seg_arr.shape
        seg_onehot = torch.zeros((len(self.label_color_map), H, W), dtype=torch.float32)
        for idx_label, color in enumerate(self.label_color_map.values()):
            mask = np.all(seg_arr == np.array(color), axis=-1)
            seg_onehot[idx_label][mask] = 1
        filename = os.path.splitext(os.path.basename(inp_path))[0]
        return inp_img, seg_onehot, filename

class ValTestSegDataset_k(Dataset):
    def __init__(self, inp_dir, seg_dir, label_color_map, crop_size=(288, 960)):
        super(ValTestSegDataset_k, self).__init__()
        self.inp_dir = inp_dir
        self.seg_dir = seg_dir
        self.length = len(inp_dir)
        self.crop_size = crop_size
        self.label_color_map = label_color_map

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = index % self.length
        inp_path = self.inp_dir[idx]
        seg_path = self.seg_dir[idx]
        inp_img = Image.open(inp_path).convert('RGB')
        seg_img = Image.open(seg_path).convert('RGB')
        i, j, h, w = T.RandomCrop.get_params(inp_img, output_size=self.crop_size)
        inp_img = F.crop(inp_img, i, j, h, w)
        seg_img = F.crop(seg_img, i, j, h, w)
        inp_img = F.to_tensor(inp_img)
        seg_arr = np.array(seg_img)
        H, W, _ = seg_arr.shape
        seg_onehot = torch.zeros((len(self.label_color_map), H, W), dtype=torch.float32)
        for idx_label, color in enumerate(self.label_color_map.values()):
            mask = np.all(seg_arr == np.array(color), axis=-1)
            seg_onehot[idx_label][mask] = 1
        filename = os.path.splitext(os.path.basename(inp_path))[0]
        return inp_img, seg_onehot, filename

class MultiValTestLoadDatasetV2_k(Dataset):
    def __init__(self, inp_dir, seg_label_dir, label_color_map):
        super(MultiValTestLoadDatasetV2_k, self).__init__()
        self.inp_dir = inp_dir
        self.seg_label_dir = seg_label_dir
        self.length = len(inp_dir)
        self.label_color_map = label_color_map

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = index % self.length
        inp_path = self.inp_dir[idx]
        seg_path = self.seg_label_dir[idx]
        inp_img = Image.open(inp_path).convert('RGB')
        seg_img = Image.open(seg_path).convert('RGB')
        inp_img = F.to_tensor(inp_img)
        seg_arr = np.array(seg_img)
        H, W, _ = seg_arr.shape
        seg_onehot = torch.zeros((len(self.label_color_map), H, W), dtype=torch.float32)
        for idx_label, color in enumerate(self.label_color_map.values()):
            mask = np.all(seg_arr == np.array(color), axis=-1)
            seg_onehot[idx_label][mask] = 1
        filename = os.path.splitext(os.path.basename(inp_path))[0]
        return inp_img, seg_onehot, filename

class multi_ValTestLoadDataset_v2(Dataset):
    def __init__(self, inp_dir, tar_dir, cam_dir, mask_dir, fr_masked_dir, seg_label_dir):
        super(multi_ValTestLoadDataset_v2, self).__init__()

        self.inp_dir = inp_dir
        self.tar_dir = tar_dir
        self.cam_dir = cam_dir
        self.mask_dir = mask_dir
        self.fr_masked_dir = fr_masked_dir
        self.seg_label_dir = seg_label_dir
        self.length = len(self.inp_dir)
        self.crop_size = 256

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index_ = index % self.length

        inp_path = self.inp_dir[index_]
        tar_path = self.tar_dir[index_]
        cam_path = self.cam_dir[index_]
        mask_path = self.mask_dir[index_]
        fr_masked_path = self.fr_masked_dir[index_]
        seg_label_path = self.seg_label_dir[index_]

        inp_img = Image.open(inp_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")
        cam_img = Image.open(cam_path).convert("L")
        mask_img = Image.open(mask_path).convert("L")
        fr_masked_img = Image.open(fr_masked_path).convert("RGB")
        seg_label_img = Image.open(seg_label_path)

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)
        fr_masked_img = F.to_tensor(fr_masked_img)
        cam_img = F.to_tensor(cam_img)
        mask_img = F.to_tensor(mask_img)
        seg_label_img = torch.from_numpy(np.array(seg_label_img)).long()

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, cam_img, mask_img, fr_masked_img, seg_label_img, filename

class multi_ValTestLoadDataset_v2_k(Dataset):
    def __init__(self, inp_dir, tar_dir, cam_dir, mask_dir, fr_masked_dir, seg_label_dir, label_color_map):
        super(multi_ValTestLoadDataset_v2_k, self).__init__()

        self.inp_dir = inp_dir
        self.tar_dir = tar_dir
        self.cam_dir = cam_dir
        self.mask_dir = mask_dir
        self.fr_masked_dir = fr_masked_dir
        self.seg_label_dir = seg_label_dir
        self.label_color_map = {
            "Sky": [128, 128, 128],
            "Building": [128, 0, 0],
            "Pole": [192, 192, 128],
            "Road": [128, 64, 128],
            "Pavement": [0, 0, 192],
            "Tree": [128, 128, 0],
            "SignSymbol": [192, 128, 128],
            "Fence": [64, 64, 128],
            "Car": [64, 0, 128],
            "Pedestrian": [64, 64, 0],
            "Bicyclist": [0, 128, 192],
            "Void": [0, 0, 0],
        }
        self.length = len(self.inp_dir)
        self.crop_size = (1152, 352)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index_ = index % self.length

        inp_path = self.inp_dir[index_]
        tar_path = self.tar_dir[index_]
        cam_path = self.cam_dir[index_]
        mask_path = self.mask_dir[index_]
        fr_masked_path = self.fr_masked_dir[index_]
        seg_label_path = self.seg_label_dir[index_]

        inp_img = Image.open(inp_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")
        cam_img = Image.open(cam_path).convert("L")
        mask_img = Image.open(mask_path).convert("L")
        fr_masked_img = Image.open(fr_masked_path).convert("RGB")
        seg_label_img = Image.open(seg_label_path).convert("RGB")

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)
        fr_masked_img = F.to_tensor(fr_masked_img)
        cam_img = F.to_tensor(cam_img)
        mask_img = F.to_tensor(mask_img)

        seg_label_img = np.array(seg_label_img)
        h, w, _ = seg_label_img.shape
        seg_onehot = torch.zeros((len(self.label_color_map), h, w), dtype=torch.float32)

        for i, color in enumerate(self.label_color_map.values()):
            color_array = np.array(color, dtype=np.uint8)
            mask = np.all(seg_label_img == color_array, axis=-1)
            seg_onehot[i][mask] = 1

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, cam_img, mask_img, fr_masked_img, seg_onehot, filename