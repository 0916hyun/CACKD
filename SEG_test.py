import torch
import os
import argparse
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from customdatasets.get_load_dataset import *
from segmentation_models_pytorch import Unet
import pandas as pd
from utils.stream_metrics import StreamSegMetrics
import csv
import torch.nn as nn
from ptflops import get_model_complexity_info
from segmentation_models_pytorch.metrics.functional import get_stats, f1_score

import torch.nn.functional as F

def resize_inference(model, img, device, n_classes, target_size=(352, 480)):
    B, _, H, W = img.shape
    img_resized = torch.nn.functional.interpolate(img, size=target_size, mode='bilinear', align_corners=False).to(
        device)

    preds_resized = model(img_resized)
    if isinstance(preds_resized, tuple):
        preds_resized = preds_resized[0]

    preds = torch.nn.functional.interpolate(preds_resized, size=(H, W), mode='bilinear', align_corners=False)

    return preds


def test(args):
    DB = args.database
    flare_test_dataset_path = args.test_input_path
    flare_test_label_path = args.test_label_path
    cam_test_dataset_path = args.test_cam_path
    mask_test_dataset_path = args.test_mask_path
    fr_masked_test_dataset_path = args.test_fr_masked_path
    seg_label_test_dataset_path = args.test_seg_label_path
    weights_save_path = args.weights_save_path
    best_model_pth = args.best_model_pth
    test_results_path = args.test_results_path
    root = args.data_root
    save_root = args.save_root
    device = args.device
    eval_batch_size = args.eval_batch_size

    model_weights_save = os.path.join(save_root, weights_save_path)
    result_test_images_save = os.path.join(save_root, test_results_path)
    os.makedirs(result_test_images_save, exist_ok=True)

    class_iou_save_path = os.path.join(model_weights_save, 'test_class_iou.csv')
    with open(class_iou_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch"] + [f'class_{i}_IoU' for i in range(12)])

    test_metrics_csv = os.path.join(model_weights_save, 'test_metrics.csv')
    with open(test_metrics_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "seg_loss", "mIoU", "F1-score", "Pixel Accuracy", "Class Accuracy"])

    flare_fold_test_dataset_path = sorted([os.path.join(root, flare_test_dataset_path, file) for file in
                                           os.listdir(os.path.join(root, flare_test_dataset_path))])
    flare_fold_test_label_path = sorted([os.path.join(root, flare_test_label_path, file) for file in
                                         os.listdir(os.path.join(root, flare_test_label_path))])
    cam_fold_test_dataset_path = sorted([os.path.join(root, cam_test_dataset_path, file) for file in
                                         os.listdir(os.path.join(root, cam_test_dataset_path))])
    mask_fold_test_dataset_path = sorted([os.path.join(root, mask_test_dataset_path, file) for file in
                                          os.listdir(os.path.join(root, mask_test_dataset_path))])
    fr_masked_fold_test_dataset_path = sorted([os.path.join(root, fr_masked_test_dataset_path, file) for file in
                                               os.listdir(os.path.join(root, fr_masked_test_dataset_path))])
    seg_label_test_dataset_path = sorted([os.path.join(root, seg_label_test_dataset_path, file) for file in
                                          os.listdir(os.path.join(root, seg_label_test_dataset_path))])

    label_color_map = {
        "Sky": [128, 128, 128],
        "Building": [70, 70, 70],
        "Pole": [153, 153, 153],
        "Road": [128, 64, 128],
        "Sidewalk": [244, 35, 232],
        "Vegetation": [107, 142, 35],
        "SignSymbol": [220, 220, 0],
        "Fence": [190, 153, 153],
        "Car": [0, 0, 142],
        "Pedestrian": [220, 20, 60],
        "Bicyclist": [119, 11, 32],
        "Void": [0, 0, 0],
    }

    if DB == 'CamVid':
        test_datasets = multi_ValTestLoadDataset_v2(
            inp_dir=flare_fold_test_dataset_path,
            tar_dir=flare_fold_test_label_path,
            cam_dir=cam_fold_test_dataset_path,
            mask_dir=mask_fold_test_dataset_path,
            fr_masked_dir=fr_masked_fold_test_dataset_path,
            seg_label_dir=seg_label_test_dataset_path
        )
    elif DB == 'KITTI':
        test_datasets = multi_ValTestLoadDataset_v2_k(
            inp_dir=flare_fold_test_dataset_path,
            tar_dir=flare_fold_test_label_path,
            cam_dir=cam_fold_test_dataset_path,
            mask_dir=mask_fold_test_dataset_path,
            fr_masked_dir=fr_masked_fold_test_dataset_path,
            seg_label_dir=seg_label_test_dataset_path,
            label_color_map=label_color_map
        )

    test_loader = DataLoader(test_datasets, batch_size=eval_batch_size, shuffle=False, drop_last=False)

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=12,
    ).to(device)

    checkpoint = torch.load(os.path.join(model_weights_save, best_model_pth), map_location=device)
    model.load_state_dict(checkpoint['seg_model_state_dict'])
    epoch = checkpoint.get('epoch', 0)

    model.eval()

    metrics = StreamSegMetrics(12)

    seg_loss_fn = nn.CrossEntropyLoss(ignore_index=11).to(device)

    total_seg_loss = 0.0
    f1_list = []
    pixel_acc_list = []
    class_acc_list = []
    num_batches = len(test_loader)
    print(model)

    epoch_start_time = time.time()
    print('======> Test Start')

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            test_input, test_label, cam_data, mask_data, fr_masked_image, seg_labels, file_name = data
            test_input = test_input.to(device)
            test_label = seg_labels.to(device)

            if test_label.shape[1] == 12:
                test_label = test_label.argmax(dim=1)



            test_out = resize_inference(model, test_input, device, 12, target_size=(352, 480))
            test_out_softmax = torch.nn.functional.softmax(test_out, dim=1)
            preds = test_out_softmax.argmax(dim=1)

            metrics.update(test_label.cpu().numpy(), preds.cpu().numpy())

            preds_tensor = preds.clone()
            labels_tensor = test_label.clone()
            labels_tensor[labels_tensor == 11] = -1

            tp, fp, fn, tn = get_stats(preds_tensor, labels_tensor, mode="multiclass", num_classes=12, ignore_index=-1)

            f1 = f1_score(tp, fp, fn, tn, reduction="macro").item()
            f1_list.append(f1)

            valid_tp = tp[:11]
            valid_fp = fp[:11]
            pixel_acc = (valid_tp.sum() / (valid_tp.sum() + valid_fp.sum())).item()
            pixel_acc_list.append(pixel_acc)

            valid_mask = (valid_tp + valid_fp) > 0
            if valid_mask.any():
                class_acc = (valid_tp[valid_mask] / (valid_tp[valid_mask] + valid_fp[valid_mask])).mean().item()
            else:
                class_acc = 0.0
            class_acc_list.append(class_acc)

        avg_seg_loss = total_seg_loss / num_batches
        avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        avg_pixel_acc = sum(pixel_acc_list) / len(pixel_acc_list) if pixel_acc_list else 0.0
        avg_class_acc = sum(class_acc_list) / len(class_acc_list) if class_acc_list else 0.0

        results = metrics.get_results()
        mIoU = results['Mean IoU']
        print('Epoch: {}\tTime: {:.4f}\tmIoU: {:.4f}'.format(
            epoch, time.time() - epoch_start_time, mIoU
        ))
        print('Epoch: {}\tTime: {:.4f}\tmIoU: {:.4f}'.format(
            epoch, time.time() - epoch_start_time, results['Mean IoU']
        ))

        class_iou = results['Class IoU']
        with open(class_iou_save_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch] + [class_iou[i] for i in range(12)])

        with open(test_metrics_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_seg_loss, mIoU, avg_f1, avg_pixel_acc, avg_class_acc])

        Test_history = {
            'mIoU': [mIoU],
            'Class IoU': [class_iou],
            'seg_loss': [avg_seg_loss],
            'F1-score': [avg_f1],
            'Pixel Accuracy': [avg_pixel_acc],
            'Class Accuracy': [avg_class_acc]
        }
        Test_history_df = pd.DataFrame(Test_history)
        Test_history_df.to_csv(os.path.join(model_weights_save, 'test_history_segmentation.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Segmentation Model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--database', type=str, required=True, help='Database name (e.g., CamVid)')
    parser.add_argument('--test_input_path', type=str, required=True, help='Path to test input images directory')
    parser.add_argument('--test_label_path', type=str, required=True, help='Path to test label masks directory')
    parser.add_argument('--test_cam_path', type=str, required=True, help='Path to directory containing CAM outputs')
    parser.add_argument('--test_mask_path', type=str, required=True, help='Path to directory containing mask files')
    parser.add_argument('--test_fr_masked_path', type=str, required=True, help='Path to directory containing combined masks')
    parser.add_argument('--test_seg_label_path', type=str, required=True, help='Path to directory containing segmentation labels')
    parser.add_argument('--weights_save_path', type=str, required=True, help='Directory to save model weights and checkpoints')
    parser.add_argument('--best_model_pth', type=str, required=True, help='Filename of the best model checkpoint (.pth)')
    parser.add_argument('--test_results_path', type=str, required=True, help='Directory to save test results')
    parser.add_argument('--save_root', type=str, required=True, help='Root directory for all output files')
    parser.add_argument('--device', type=str, default='cuda', help='Computation device (e.g., cuda or cpu)')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Batch size for evaluation')

    args = parser.parse_args()

    test(args)

