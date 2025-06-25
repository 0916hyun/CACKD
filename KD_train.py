import argparse
from customdatasets.get_load_dataset import *
from KD_trainer import Trainer
import sys
sys.path.append('C:/Users/8138/anaconda3/lib/site-packages')
import os


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

def train(args):
    DB = args.database
    flare_train_dataset_path = args.train_input_path #0
    flare_train_label_path = args.train_label_path #1
    flare_train_flare_path = args.train_flare_path #2
    cam_train_dataset_path = args.train_cam_path #3
    mask_train_dataset_path = args.train_mask_path #4
    fr_masked_train_dataset_path = args.train_fr_masked_path #5
    label_train_dataset_path = args.train_seg_path #6

    flare_val_dataset_path = args.valid_input_path #0
    flare_val_label_path = args.valid_label_path #1
    flare_val_flare_path = args.valid_flare_path  #2
    cam_val_dataset_path = args.valid_cam_path  #3
    mask_val_dataset_path = args.valid_mask_path #4
    fr_masked_val_dataset_path = args.valid_fr_masked_path #5
    label_val_dataset_path = args.valid_seg_path #6

    flare_test_dataset_path = args.test_input_path #0
    flare_test_label_path = args.test_label_path #1
    cam_test_dataset_path = args.test_cam_path #2
    mask_test_dataset_path = args.test_mask_path #3
    fr_masked_test_dataset_path = args.test_fr_masked_path #4
    seg_label_test_dataset_path = args.test_seg_label_path #5

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
    test_save_root = args.test_save_root
    device = args.device




    if DB == 'CamVid':
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
    elif DB == 'KITTI':
        num_classes = 12  # KITTI에 맞게 수정
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
    else:
        raise ValueError("Unknown database. Please set the num_classes accordingly.")

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
    cam_fold_train_dataset_path = sorted([os.path.join(root, cam_train_dataset_path, file)
                                          for file in os.listdir(os.path.join(root, cam_train_dataset_path))])
    mask_fold_train_dataset_path = sorted([os.path.join(root, mask_train_dataset_path, file)
                                           for file in os.listdir(os.path.join(root, mask_train_dataset_path))])
    fr_masked_fold_train_dataset_path = sorted([os.path.join(root, fr_masked_train_dataset_path, file)
                                                for file in os.listdir(os.path.join(root, fr_masked_train_dataset_path))])
    seg_fold_train_label_path = sorted([os.path.join(root, label_train_dataset_path, file)
                                        for file in os.listdir(os.path.join(root, label_train_dataset_path))])

    flare_fold_valid_dataset_path = sorted([os.path.join(root, flare_val_dataset_path, file)
                                            for file in os.listdir(os.path.join(root, flare_val_dataset_path))])
    flare_fold_valid_label_path = sorted([os.path.join(root, flare_val_label_path, file)
                                          for file in os.listdir(os.path.join(root, flare_val_label_path))])
    flare_fold_valid_flare_path = sorted([os.path.join(root, flare_val_flare_path, file)
                                          for file in os.listdir(os.path.join(root, flare_val_flare_path))])
    cam_fold_valid_dataset_path = sorted([os.path.join(root, cam_val_dataset_path, file)
                                          for file in os.listdir(os.path.join(root, cam_val_dataset_path))])
    mask_fold_valid_dataset_path = sorted([os.path.join(root, mask_val_dataset_path, file)
                                           for file in os.listdir(os.path.join(root, mask_val_dataset_path))])
    fr_masked_fold_valid_dataset_path = sorted([os.path.join(root, fr_masked_val_dataset_path, file)
                                                for file in os.listdir(os.path.join(root, fr_masked_val_dataset_path))])
    seg_fold_valid_label_path = sorted([os.path.join(root, label_val_dataset_path, file)
                                        for file in os.listdir(os.path.join(root, label_val_dataset_path))])

    flare_fold_test_dataset_path = sorted([os.path.join(root, flare_test_dataset_path, file)
                                            for file in os.listdir(os.path.join(root, flare_test_dataset_path))])
    flare_fold_test_label_path = sorted([os.path.join(root, flare_test_label_path, file)
                                          for file in os.listdir(os.path.join(root, flare_test_label_path))])
    cam_fold_test_dataset_path = sorted([os.path.join(root, cam_test_dataset_path, file)
                                          for file in os.listdir(os.path.join(root, cam_test_dataset_path))])
    mask_fold_test_dataset_path = sorted([os.path.join(root, mask_test_dataset_path, file)
                                           for file in os.listdir(os.path.join(root, mask_test_dataset_path))])
    fr_masked_fold_test_dataset_path = sorted([os.path.join(root, fr_masked_test_dataset_path, file)
                                                for file in os.listdir(os.path.join(root, fr_masked_test_dataset_path))])
    seg_fold_test_label_path = sorted([os.path.join(root, seg_label_test_dataset_path, file)
                                               for file in os.listdir(os.path.join(root, seg_label_test_dataset_path))])


    # 데이터셋 로드
    if args.database == 'CamVid':
        train_datasets = get_train_dataset_multi(
            inp_dir=flare_fold_train_dataset_path,
            tar_dir=flare_fold_train_label_path,
            flare_dir=flare_fold_train_flare_path,  # dummy
            cam_dir=cam_fold_train_dataset_path,
            mask_dir=mask_fold_train_dataset_path,
            fr_masked_dir=fr_masked_fold_train_dataset_path,
            seg_dir=seg_fold_train_label_path,  # 세그멘테이션 라벨 경로
            label_color_map=label_color_map
        )
        valid_datasets = get_val_test_dataset_multi(
            inp_dir=flare_fold_valid_dataset_path,
            tar_dir=flare_fold_valid_label_path,
            flare_dir=flare_fold_valid_flare_path,  # dummy
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
            seg_label_dir=seg_fold_test_label_path
        )

    elif args.database == 'KITTI':
        train_datasets = get_train_dataset_multi_k(
            inp_dir=flare_fold_train_dataset_path,
            tar_dir=flare_fold_train_label_path,
            flare_dir=flare_fold_train_flare_path,  # 추가: flare_dir
            cam_dir=cam_fold_train_dataset_path,
            mask_dir=mask_fold_train_dataset_path,
            fr_masked_dir=fr_masked_fold_train_dataset_path,
            seg_dir=seg_fold_train_label_path,  # 세그멘테이션 더미 경로 전달
            label_color_map=label_color_map
        )
        valid_datasets = get_val_test_dataset_multi_k(
            inp_dir=flare_fold_valid_dataset_path,
            tar_dir=flare_fold_valid_label_path,
            flare_dir=flare_fold_valid_flare_path,  # 추가: flare_dir
            cam_dir=cam_fold_valid_dataset_path,
            mask_dir=mask_fold_valid_dataset_path,
            fr_masked_dir=fr_masked_fold_valid_dataset_path,
            seg_dir=seg_fold_valid_label_path,  # 세그멘테이션 더미 경로 전달
            label_color_map=label_color_map
        )
        test_datasets = multi_ValTestLoadDataset_v2_k(
            inp_dir=flare_fold_test_dataset_path,
            tar_dir=flare_fold_test_label_path,
            cam_dir=cam_fold_test_dataset_path,
            mask_dir=mask_fold_test_dataset_path,
            fr_masked_dir=fr_masked_fold_test_dataset_path,
            seg_label_dir=seg_fold_test_label_path,
            label_color_map=label_color_map
        )
    # train_subset = Subset(train_datasets, [0])
    # valid_subset = Subset(valid_datasets, [0])
    # test_subset = Subset(test_datasets, [0])
    #
    # train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True, drop_last=False)
    # valid_loader = DataLoader(valid_subset, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    # test_loader = DataLoader(test_subset, batch_size=eval_batch_size, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_datasets, batch_size=train_batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_datasets, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_datasets, batch_size=eval_batch_size, shuffle=False, drop_last=False)

    trainer = Trainer(
        save_root=save_root,
        save_model_weights_path=weights_save_path,
        save_val_results_path=val_results_path,
        device=device,
        lr=lr,
        num_epochs=NUM_EPOCH,
        n_classes=num_classes,
        label_color_map=label_color_map,
        temperature=4.0,
        teacher_checkpoint_path=args.teacher_checkpoint_path
    ).to(device)

    trainer.train(train_loader, valid_loader, test_loader, resume_from_checkpoint=args.resume_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train flare removal')
    parser.add_argument('--data_root', type=str, default='C:/workspace/Datas/')
    parser.add_argument('--database', type=str, default='CamVid')

    parser.add_argument('--train_input_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_train')
    parser.add_argument('--train_label_path', type=str, default='D:/workspace/Restoration/Unetpp_CAMVID_fold1_resnet50/case1/train')
    parser.add_argument('--train_flare_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_train/')
    parser.add_argument('--train_cam_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_train_CAM/highres_cam')
    parser.add_argument('--train_mask_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_train_CAM/mask')
    parser.add_argument('--train_fr_masked_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_train_CAM/mask_combine')
    parser.add_argument('--train_seg_path', type=str, default='C:/workspace/Datas/camvid_label_12_2fold_v2/fold1_train/labels')

    parser.add_argument('--valid_input_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_valid')
    parser.add_argument('--valid_label_path', type=str, default='D:/workspace/Restoration/Unetpp_CAMVID_fold1_resnet50/case1/valid')
    parser.add_argument('--valid_flare_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_valid/')
    parser.add_argument('--valid_cam_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_valid_CAM/highres_cam')
    parser.add_argument('--valid_mask_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_valid_CAM/mask')
    parser.add_argument('--valid_fr_masked_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/fold1_valid_CAM/mask_combine')
    parser.add_argument('--valid_seg_path', type=str, default='C:/workspace/Datas/camvid_label_12_2fold_v2/fold1_valid/labels')

    parser.add_argument('--test_input_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/test')
    parser.add_argument('--test_label_path', type=str, default='D:/workspace/Restoration/Unetpp_CAMVID_fold1_resnet50/case1/test')
    parser.add_argument('--test_cam_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/test_CAM/highres_cam')
    parser.add_argument('--test_mask_path', type=str,default='flare_synthesized_CamVid_12label_Dataset_0527/test_CAM/mask')
    parser.add_argument('--test_fr_masked_path', type=str, default='flare_synthesized_CamVid_12label_Dataset_0527/test_CAM/mask_combine')
    parser.add_argument('--test_seg_label_path', type=str, default='camvid_label_12_2fold_v2/test/labels')

    parser.add_argument('--weights_save_path', type=str, default='D:/workspace/KD_unet_cam/fold1_final_softmax/')
    parser.add_argument('--val_results_path', type=str, default='D:/workspace/KD_unet_cam/fold1_final_softmax/val_results')
    parser.add_argument('--test_save_root', type=str, default='D:/workspace/KD_unet_cam/fold1_final_softmax/test_results')
    parser.add_argument('--save_root', type=str, default='./save_root')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--teacher_checkpoint_path', type=str, default='D:/final_model/CAMVID/fold1/T/seg/no_pos/best_model.pth')

    parser.add_argument('--resume_checkpoint', type=str, default='D:/workspace/KD_unet_cam/fold1_final_softmax/best.pth')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    train(args)
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Train flare removal')
#     parser.add_argument('--data_root', type=str, default='D:/KITTI/Datas/resize_not')
#     parser.add_argument('--database', type=str, default='KITTI')
#
#     parser.add_argument('--train_input_path', type=str, default='KITTI_flare/fold1_train')
#     parser.add_argument('--train_label_path', type=str, default='D:/workspace/Restoration/Unetpp_KITTI_fold1_resnet50v2/case1/train')
#     parser.add_argument('--train_flare_path', type=str, default='KITTI_flare/fold1_train')
#     parser.add_argument('--train_cam_path', type=str, default='KITTI_cam/fold1_train/highres_cam')
#     parser.add_argument('--train_mask_path', type=str, default='KITTI_cam/fold1_train/mask')
#     parser.add_argument('--train_fr_masked_path', type=str, default='KITTI_cam/fold1_train/mask_combine')
#     parser.add_argument('--train_seg_path', type=str, default='KITTI_clean/fold1_train/labels')
#
#     parser.add_argument('--valid_input_path', type=str, default='KITTI_flare/fold1_val')
#     parser.add_argument('--valid_label_path', type=str, default='D:/workspace/Restoration/Unetpp_KITTI_fold1_resnet50v2/case1/valid')
#     parser.add_argument('--valid_flare_path', type=str, default='KITTI_flare/fold1_val')
#     parser.add_argument('--valid_cam_path', type=str, default='KITTI_cam/fold1_val/highres_cam')
#     parser.add_argument('--valid_mask_path', type=str, default='KITTI_cam/fold1_val/mask')
#     parser.add_argument('--valid_fr_masked_path', type=str, default='KITTI_cam/fold1_val/mask_combine')
#     parser.add_argument('--valid_seg_path', type=str, default='KITTI_clean/fold1_val/labels')
#
#     parser.add_argument('--test_input_path', type=str, default='KITTI_flare/test')
#     parser.add_argument('--test_label_path', type=str, default='D:/workspace/Restoration/Unetpp_KITTI_fold1_resnet50v2/case1/test')
#     parser.add_argument('--test_cam_path', type=str, default='KITTI_cam/test/highres_cam')
#     parser.add_argument('--test_mask_path', type=str, default='KITTI_cam/test/mask')
#     parser.add_argument('--test_fr_masked_path', type=str, default='KITTI_cam/test/mask_combine')
#     parser.add_argument('--test_seg_label_path', type=str, default='KITTI_clean/test/labels')
#
#     parser.add_argument('--weights_save_path', type=str, default='D:/workspace/KD_unet_cam/KITTI/fold2_final/')
#     parser.add_argument('--val_results_path', type=str, default='D:/workspace/KD_unet_cam/KITTI/fold2_final/val_results')
#     parser.add_argument('--test_save_root', type=str, default='D:/workspace/KD_unet_cam/KITTI/fold2_final//test_results')
#     parser.add_argument('--save_root', type=str, default='./save_root')
#     parser.add_argument('--lr', type=float, default=0.0001)
#     parser.add_argument('--num_epoch', type=int, default=1500)
#     parser.add_argument('--train_batch_size', type=int, default=2)
#     parser.add_argument('--eval_batch_size', type=int, default=1)
#     parser.add_argument('--teacher_checkpoint_path', type=str, default='D:/workspace/CAM/KITTI/unetpp_CAM_fold1_no_pos3_case1_best/Degraded/checkpoint_epoch_1052.pth')
#
#     parser.add_argument('--resume_checkpoint', type=str, default='')
#     parser.add_argument('--device', type=str, default='cuda')
#
#     args = parser.parse_args()
#     train(args)