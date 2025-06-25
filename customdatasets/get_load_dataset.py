from .customdateset import *

def get_train_dataset_multi(inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map):
    return TrainLoadDatasetMulti(inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map)

def get_train_dataset_multi_k(inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map):
    return TrainLoadDatasetMulti_k(inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map)

def get_val_test_dataset_multi(inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map):
    return ValTestLoadDatasetMulti(inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map)

def get_val_test_dataset_multi_k(inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map):
    return ValTestLoadDatasetMulti_k(inp_dir, tar_dir, flare_dir, cam_dir, mask_dir, fr_masked_dir, seg_dir, label_color_map)

def get_train_dataset_c(inp_dir, seg_dir, crop_size=256):
    return TrainSegDataset_c(inp_dir, seg_dir, crop_size)

def get_val_test_dataset_c(inp_dir, seg_dir, crop_size=256):
    return ValTestSegDataset_c(inp_dir, seg_dir, crop_size)

def get_multi_val_test_dataset_v2_c(inp_dir, seg_label_dir):
    return MultiValTestLoadDatasetV2_c(inp_dir, seg_label_dir)

def get_train_dataset_k(inp_dir, seg_dir, label_color_map, crop_size=(288, 960)):
    return TrainSegDataset_k(inp_dir, seg_dir, label_color_map, crop_size)

def get_val_test_dataset_k(inp_dir, seg_dir, label_color_map, crop_size=(288, 960)):
    return ValTestSegDataset_k(inp_dir, seg_dir, label_color_map, crop_size)

def get_multi_val_test_dataset_v2_k(inp_dir, seg_label_dir, label_color_map):
    return MultiValTestLoadDatasetV2_k(inp_dir, seg_label_dir, label_color_map)


