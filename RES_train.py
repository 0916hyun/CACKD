import torch
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from customdatasets.get_load_dataset import *
from RES_trainer import Trainer

import sys
sys.path.append('C:/Users/8138/anaconda3/lib/site-packages')
import os



def train(args):
    DB = args.database
    flare_train_dataset_path = args.train_input_path
    flare_train_label_path = args.train_label_path
    flare_val_dataset_path = args.valid_input_path
    flare_val_label_path = args.valid_label_path
    flare_train_flare_path = args.train_flare_path
    flare_val_flare_path = args.valid_flare_path
    cam_train_dataset_path = args.train_cam_path
    mask_train_dataset_path = args.train_mask_path
    fr_masked_train_dataset_path = args.train_fr_masked_path
    cam_val_dataset_path = args.valid_cam_path
    mask_val_dataset_path = args.valid_mask_path
    fr_masked_val_dataset_path = args.valid_fr_masked_path
    weights_save_path = args.weights_save_path
    val_results_path = args.val_results_path
    root = args.data_root
    save_root = args.save_root
    device = args.device
    lr = args.lr
    NUM_EPOCH = args.num_epoch
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    DB = args.database
    flare_test_dataset_path = args.test_input_path
    flare_test_label_path = args.test_label_path
    cam_test_dataset_path = args.test_cam_path
    mask_test_dataset_path = args.test_mask_path
    fr_masked_test_dataset_path = args.test_fr_masked_path
    seg_label_test_dataset_path = args.test_seg_label_path
    test_save_root = args.test_save_root
    device = args.device




    if DB == 'CamVid':
        num_classes = 12
    elif DB == 'KITTI':
        num_classes = 12
    else:
        raise ValueError("Unknown database. Please set the num_classes accordingly.")

    label_color_map = {
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
        "Void": [0, 0, 0]
    }

    # 시드 설정
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    # 파일 경로 설정
    flare_fold_train_dataset_path = sorted([os.path.join(root, flare_train_dataset_path, file)
                                            for file in os.listdir(os.path.join(root, flare_train_dataset_path))])
    flare_fold_train_label_path = sorted([os.path.join(root, flare_train_label_path, file)
                                          for file in os.listdir(os.path.join(root, flare_train_label_path))])
    flare_fold_train_flare_path = sorted([os.path.join(root, flare_train_flare_path, file)
                                          for file in os.listdir(os.path.join(root, flare_train_flare_path))])
    flare_fold_valid_dataset_path = sorted([os.path.join(root, flare_val_dataset_path, file)
                                            for file in os.listdir(os.path.join(root, flare_val_dataset_path))])
    flare_fold_valid_label_path = sorted([os.path.join(root, flare_val_label_path, file)
                                          for file in os.listdir(os.path.join(root, flare_val_label_path))])
    flare_fold_valid_flare_path = sorted([os.path.join(root, flare_val_flare_path, file)
                                          for file in os.listdir(os.path.join(root, flare_val_flare_path))])
    cam_fold_train_dataset_path = sorted([os.path.join(root, cam_train_dataset_path, file)
                                          for file in os.listdir(os.path.join(root, cam_train_dataset_path))])
    mask_fold_train_dataset_path = sorted([os.path.join(root, mask_train_dataset_path, file)
                                           for file in os.listdir(os.path.join(root, mask_train_dataset_path))])
    fr_masked_fold_train_dataset_path = sorted([os.path.join(root, fr_masked_train_dataset_path, file)
                                                for file in os.listdir(os.path.join(root, fr_masked_train_dataset_path))])
    cam_fold_valid_dataset_path = sorted([os.path.join(root, cam_val_dataset_path, file)
                                          for file in os.listdir(os.path.join(root, cam_val_dataset_path))])
    mask_fold_valid_dataset_path = sorted([os.path.join(root, mask_val_dataset_path, file)
                                           for file in os.listdir(os.path.join(root, mask_val_dataset_path))])
    fr_masked_fold_valid_dataset_path = sorted([os.path.join(root, fr_masked_val_dataset_path, file)
                                                for file in os.listdir(os.path.join(root, fr_masked_val_dataset_path))])
    seg_fold_train_label_path = sorted([os.path.join(root, args.train_seg_path, file) for file in os.listdir(os.path.join(root, args.train_seg_path))])
    seg_fold_valid_label_path = sorted([os.path.join(root, args.valid_seg_path, file) for file in os.listdir(os.path.join(root, args.valid_seg_path))])

    flare_fold_test_dataset_path = sorted([os.path.join(root, flare_test_dataset_path, file)
                                            for file in os.listdir(os.path.join(root, flare_test_dataset_path))])
    flare_fold_test_label_path = sorted([os.path.join(root, flare_test_label_path, file)
                                          for file in os.listdir(os.path.join(root, flare_test_label_path))])
    cam_fold_test_dataset_path = sorted([os.path.join(root, cam_test_dataset_path, file)
                                          for file in os.listdir(os.path.join(root, cam_test_dataset_path))])
    mask_fold_test_dataset_path = sorted([os.path.join(root, mask_test_dataset_path, file)
                                           for file in os.listdir(os.path.join(root, mask_test_dataset_path))])
    fr_masked_fold_test_dataset_path = sorted([os.path.join(root, fr_masked_test_dataset_path, file)
                                                for file in
                                                os.listdir(os.path.join(root, fr_masked_test_dataset_path))])
    seg_label_test_dataset_path = sorted([os.path.join(root, seg_label_test_dataset_path, file)
                                               for file in
                                               os.listdir(os.path.join(root, seg_label_test_dataset_path))])


    if DB == 'CamVid':
        train_datasets = get_train_dataset_multi(
            inp_dir=flare_fold_train_dataset_path,
            tar_dir=flare_fold_train_label_path,
            flare_dir=flare_fold_train_flare_path,
            cam_dir=cam_fold_train_dataset_path,
            mask_dir=mask_fold_train_dataset_path,
            fr_masked_dir=fr_masked_fold_train_dataset_path,
            seg_dir=seg_fold_train_label_path,
            label_color_map=label_color_map
        )
        valid_datasets = get_val_test_dataset_multi(
            inp_dir=flare_fold_valid_dataset_path,
            tar_dir=flare_fold_valid_label_path,
            flare_dir=flare_fold_valid_flare_path,
            cam_dir=cam_fold_valid_dataset_path,
            mask_dir=mask_fold_valid_dataset_path,
            fr_masked_dir=fr_masked_fold_valid_dataset_path,
            seg_dir=seg_fold_valid_label_path,
            label_color_map=label_color_map
        )
        test_datasets = multi_ValTestLoadDataset_v2(
            inp_dir=flare_fold_test_dataset_path,
            tar_dir=flare_fold_test_label_path,
            cam_dir=cam_fold_test_dataset_path,
            mask_dir=mask_fold_test_dataset_path,
            fr_masked_dir=fr_masked_fold_test_dataset_path,
            seg_label_dir=seg_label_test_dataset_path
        )

    elif DB == 'KITTI':
        train_datasets = get_train_dataset_multi_k(
            inp_dir=flare_fold_train_dataset_path,
            tar_dir=flare_fold_train_label_path,
            flare_dir=flare_fold_train_flare_path,
            cam_dir=cam_fold_train_dataset_path,
            mask_dir=mask_fold_train_dataset_path,
            fr_masked_dir=fr_masked_fold_train_dataset_path,
            seg_dir=seg_fold_train_label_path,
            label_color_map=label_color_map
        )
        valid_datasets = get_val_test_dataset_multi_k(
            inp_dir=flare_fold_valid_dataset_path,
            tar_dir=flare_fold_valid_label_path,
            flare_dir=flare_fold_valid_flare_path,
            cam_dir=cam_fold_valid_dataset_path,
            mask_dir=mask_fold_valid_dataset_path,
            fr_masked_dir=fr_masked_fold_valid_dataset_path,
            seg_dir=seg_fold_valid_label_path,
            label_color_map=label_color_map
        )
        test_datasets = multi_ValTestLoadDataset_v2_k(
            inp_dir=flare_fold_test_dataset_path,
            tar_dir=flare_fold_test_label_path,
            cam_dir=cam_fold_test_dataset_path,
            mask_dir=mask_fold_test_dataset_path,
            fr_masked_dir=fr_masked_fold_test_dataset_path,
            seg_label_dir=seg_label_test_dataset_path,
            label_color_map=label_color_map
        )


    train_loader = DataLoader(train_datasets, batch_size=train_batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_datasets, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_datasets, batch_size=eval_batch_size, shuffle=False, drop_last=False)


    trainer = Trainer(save_root, weights_save_path, val_results_path, device, lr,
                      NUM_EPOCH, num_classes, label_color_map).to(device)
    trainer.train(train_loader, valid_loader, test_loader, resume_from_checkpoint=args.resume_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train flare removal')
    # Dataset paths
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of datasets')
    parser.add_argument('--database', type=str, required=True, help='Dataset name (e.g., CAMVID)')
    parser.add_argument('--train_input_path', type=str, required=True, help='Directory of training input images')
    parser.add_argument('--train_label_path', type=str, required=True, help='Directory of training label images')
    parser.add_argument('--valid_input_path', type=str, required=True, help='Directory of validation input images')
    parser.add_argument('--valid_label_path', type=str, required=True, help='Directory of validation label images')
    parser.add_argument('--train_flare_path', type=str, required=True, help='Directory of flare-augmented training images')
    parser.add_argument('--valid_flare_path', type=str, required=True, help='Directory of flare-augmented validation images')
    parser.add_argument('--train_cam_path', type=str, required=True, help='Directory of CAM outputs for training set')
    parser.add_argument('--train_mask_path', type=str, required=True, help='Directory of mask files for training set')
    parser.add_argument('--train_fr_masked_path', type=str, required=True, help='Directory of combined masks for training set')
    parser.add_argument('--valid_cam_path', type=str, required=True, help='Directory of CAM outputs for validation set')
    parser.add_argument('--valid_mask_path', type=str, required=True, help='Directory of mask files for validation set')
    parser.add_argument('--valid_fr_masked_path', type=str, required=True, help='Directory of combined masks for validation set')
    parser.add_argument('--train_seg_path', type=str, required=True, help='Directory of segmentation labels for training')
    parser.add_argument('--valid_seg_path', type=str, required=True, help='Directory of segmentation labels for validation')

    # Test set paths
    parser.add_argument('--test_input_path', type=str, required=True, help='Directory of test input images')
    parser.add_argument('--test_label_path', type=str, required=True, help='Directory of test label images')
    parser.add_argument('--test_cam_path', type=str, required=True, help='Directory of CAM outputs for test set')
    parser.add_argument('--test_mask_path', type=str, required=True, help='Directory of mask files for test set')
    parser.add_argument('--test_fr_masked_path', type=str, required=True, help='Directory of combined masks for test set')
    parser.add_argument('--test_seg_label_path', type=str, required=True, help='Directory of segmentation labels for test set')

    # Output and checkpoints
    parser.add_argument('--weights_save_path', type=str, required=True, help='Directory to save trained weights and checkpoints')
    parser.add_argument('--val_results_path', type=str, required=True, help='Directory to save validation results')
    parser.add_argument('--test_save_root', type=str, required=True, help='Directory to save test results')
    parser.add_argument('--save_root', type=str, required=True, help='Root directory for all outputs')

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Computation device (cuda or cpu)')

    args = parser.parse_args()
    train(args)
