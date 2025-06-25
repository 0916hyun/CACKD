import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from customdatasets.get_load_dataset import *
from unet_trainer_origin import Trainer


def get_model(model_name: str, n_classes: int, device):
    model_name = model_name.lower()
    if model_name == "unet":
        return smp.Unet(
            encoder_name="resnet50", encoder_weights="imagenet",
            in_channels=3, classes=n_classes
        ).to(device)
    if model_name == "unetpp":
        return smp.UnetPlusPlus(
            encoder_name="resnet50", encoder_weights="imagenet",
            in_channels=3, classes=n_classes
        ).to(device)
    raise ValueError(f"Unsupported model: {model_name}")


def train(args):
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    def sorted_file_list(sub_dir):
        full_dir = os.path.join(args.data_root, sub_dir)
        return sorted(os.path.join(full_dir, f) for f in os.listdir(full_dir))

    train_img_paths = sorted_file_list(args.train_input_path)
    train_seg_paths = sorted_file_list(args.train_seg_path)
    valid_img_paths = sorted_file_list(args.valid_input_path)
    valid_seg_paths = sorted_file_list(args.valid_seg_path)

    if args.database == "CamVid":
        num_classes = 12
        label_color_map = {}
        train_dataset = get_train_dataset_c(
            inp_dir=train_img_paths,
            seg_dir=train_seg_paths,
            crop_size=256
        )
        valid_dataset = get_val_test_dataset_c(
            inp_dir=valid_img_paths,
            seg_dir=valid_seg_paths,
            crop_size=256
        )
    elif args.database == "KITTI":
        num_classes = 12
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
        train_dataset = get_train_dataset_k(
            inp_dir=train_img_paths,
            seg_dir=train_seg_paths,
            label_color_map=label_color_map,
            crop_size=(288, 960)
        )
        valid_dataset = get_val_test_dataset_k(
            inp_dir=valid_img_paths,
            seg_dir=valid_seg_paths,
            label_color_map=label_color_map,
            crop_size=(288, 960)
        )
    else:
        raise ValueError("database must be 'CamVid' or 'KITTI'")

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False
    )

    device = args.device
    model = get_model(args.model, num_classes, device)

    trainer = Trainer(
        save_root=args.save_root,
        save_model_weights_path=args.weights_save_path,
        save_val_results_path=args.val_results_path,
        device=device,
        lr=args.lr,
        num_epochs=args.num_epoch,
        label_color_map=label_color_map,
        n_classes=num_classes,
        model_type=args.model,
        model=model
    ).to(device)

    trainer.train(
        train_loader,
        valid_loader,
        resume_from_checkpoint=args.resume_checkpoint
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model (CamVid / KITTI)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--database", choices=["CamVid", "KITTI"], required=True)
    parser.add_argument("--train_input_path", type=str, required=True)
    parser.add_argument("--train_seg_path", type=str, required=True)
    parser.add_argument("--valid_input_path", type=str, required=True)
    parser.add_argument("--valid_seg_path", type=str, required=True)
    parser.add_argument("--model", choices=["unet", "unetpp"], default="unet")
    parser.add_argument("--weights_save_path", type=str, default="./weights")
    parser.add_argument("--val_results_path", type=str, default="./val_results")
    parser.add_argument("--save_root", type=str, default="./runs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    args = parser.parse_args()
    train(args)
