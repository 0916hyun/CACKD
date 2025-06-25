import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm
from utils.stream_metrics import StreamSegMetrics
from torchvision import transforms
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_inference(model, img, device, n_classes, target_size=(352, 480)):
    B, _, H, W = img.shape
    img_resized = torch.nn.functional.interpolate(img, size=target_size, mode='bilinear', align_corners=False).to(
        device)

    preds_resized = model(img_resized)
    if isinstance(preds_resized, tuple):
        preds_resized = preds_resized[0]

    preds = torch.nn.functional.interpolate(preds_resized, size=(H, W), mode='bilinear', align_corners=False)

    return preds

class Trainer(nn.Module):
    def __init__(self, save_root, save_model_weights_path, save_val_results_path, device, lr, num_epochs, label_color_map, n_classes, model_type, model):
        super(Trainer, self).__init__()

        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.label_color_map = label_color_map
        self.n_classes = n_classes
        self.model_type = model_type.lower()
        self.model = model
        self.model.to(self.device)

        self.save_val_results_path = save_val_results_path
        self.save_model_weights_path = save_model_weights_path
        os.makedirs(self.save_val_results_path, exist_ok=True)
        os.makedirs(self.save_model_weights_path, exist_ok=True)

        self.seg_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.seg_loss = nn.CrossEntropyLoss(ignore_index=19).to(self.device)

        self.metrics = StreamSegMetrics(n_classes)

        self.train_log_path = os.path.join(self.save_model_weights_path, "train_metrics.csv")
        self.train_class_iou_path = os.path.join(self.save_model_weights_path, "train_class_iou.csv")
        self.valid_log_path = os.path.join(self.save_model_weights_path, "valid_metrics.csv")
        self.valid_class_iou_path = os.path.join(self.save_model_weights_path, "valid_class_iou.csv")
        self.test_log_path = os.path.join(self.save_model_weights_path, "test_metrics.csv")

        self._initialize_csv(self.train_log_path, metrics=True)
        self._initialize_csv(self.train_class_iou_path, metrics=False)
        self._initialize_csv(self.valid_log_path, metrics=True)
        self._initialize_csv(self.valid_class_iou_path, metrics=False)

    def _initialize_csv(self, filepath, metrics=True):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            if metrics:
                writer.writerow(["epoch", "seg_loss", "mIoU"])
            else:
                writer.writerow(["epoch"] + [f'class_{i}_IoU' for i in range(self.n_classes)])

    def _log_to_csv(self, filepath, epoch, seg_loss=None, mIoU=None, class_iou=None):
        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            if seg_loss is not None and mIoU is not None:
                writer.writerow([epoch + 1, seg_loss, mIoU])
            if class_iou is not None:
                writer.writerow([epoch + 1] + [class_iou[c] for c in range(self.n_classes)])

    def _train_epoch(self, epoch, train_loader):
        self.model.train()
        epoch_seg_loss = 0
        num_batches = len(train_loader)

        self.metrics.reset()

        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{self.num_epochs} - Training", unit="batch") as pbar:
            for i, data in enumerate(train_loader):
                input_img = data[0].to(self.device)
                seg_label = data[1].to(self.device)

                self.seg_optimizer.zero_grad()

                final_out = self.model(input_img)

                if seg_label.dim() == 4:
                    seg_label = seg_label.argmax(dim=1)
                elif seg_label.dim() == 3 and seg_label.shape[1] == 1:
                    seg_label = seg_label.squeeze(1)

                seg_label = seg_label.long()

                seg_loss = self.seg_loss(final_out, seg_label)

                seg_loss.backward()
                self.seg_optimizer.step()

                epoch_seg_loss += seg_loss.item()

                preds = final_out.argmax(dim=1).cpu().numpy()
                labels = seg_label.cpu().numpy()

                self.metrics.update(labels, preds)

                pbar.set_postfix({'seg_loss': seg_loss.item()})
                pbar.update(1)

        avg_seg_loss = epoch_seg_loss / num_batches
        metrics = self.metrics.get_results()
        mIoU = metrics['Mean IoU']
        class_iou = metrics['Class IoU']

        self._log_to_csv(self.train_log_path, epoch, avg_seg_loss, mIoU)
        self._log_to_csv(self.train_class_iou_path, epoch, class_iou=class_iou)

        print(f"Epoch {epoch + 1} - Train Segmentation Loss: {avg_seg_loss:.4f}, mIoU: {mIoU:.4f}")
        for cls, iou in class_iou.items():
            print(f"Class {cls} IoU: {iou:.4f}")

        return avg_seg_loss, mIoU

    def _valid_epoch(self, epoch, val_loader):
        self.model.eval()
        epoch_val_seg_loss = 0
        num_batches = len(val_loader)

        self.metrics.reset()

        with torch.no_grad():
            with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{self.num_epochs} - Validation",
                      unit="batch") as pbar:
                for i, data in enumerate(val_loader):
                    input_img = data[0].to(self.device)
                    seg_label = data[1].to(self.device)

                    final_out = self.model(input_img)

                    if seg_label.dim() == 4 and seg_label.shape[1] == 1:
                        seg_label = seg_label.squeeze(1)
                    elif seg_label.dim() == 4 and seg_label.shape[1] > 1:
                        seg_label = seg_label.argmax(dim=1)

                    seg_label = seg_label.long()

                    seg_loss = self.seg_loss(final_out, seg_label)
                    epoch_val_seg_loss += seg_loss.item()

                    preds = final_out.argmax(dim=1).cpu().numpy()
                    labels = seg_label.cpu().numpy()
                    self.metrics.update(labels, preds)

                    pbar.set_postfix({'val_seg_loss': seg_loss.item()})
                    pbar.update(1)

        avg_val_seg_loss = epoch_val_seg_loss / num_batches
        metrics = self.metrics.get_results()
        mIoU = metrics['Mean IoU']
        class_iou = metrics['Class IoU']

        self._log_to_csv(self.valid_log_path, epoch, avg_val_seg_loss, mIoU)
        self._log_to_csv(self.valid_class_iou_path, epoch, class_iou=class_iou)

        print(f"Epoch {epoch + 1} - Validation Segmentation Loss: {avg_val_seg_loss:.4f}, mIoU: {mIoU:.4f}")
        for cls, iou in class_iou.items():
            print(f"Class {cls} IoU: {iou:.4f}")

        return avg_val_seg_loss, mIoU

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        save_path = os.path.join(self.save_model_weights_path, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, save_path)
        print(f"모델 체크포인트를 {save_path}에 저장했습니다.")

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.seg_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"모델 체크포인트 {checkpoint_path}로부터 로드했습니다. 이어서 {start_epoch} 에포크부터 학습을 재개합니다.")
        return start_epoch

    def train(self, train_loader, val_loader, test_loader, resume_from_checkpoint=None):
        if resume_from_checkpoint:
            start_epoch = self._load_checkpoint(resume_from_checkpoint)
        else:
            start_epoch = 0

        for epoch in range(start_epoch, self.num_epochs):
            tqdm.write(f"Epoch {epoch + 1}/{self.num_epochs} 시작...")

            avg_train_seg_loss, train_mIoU = self._train_epoch(epoch, train_loader)
            tqdm.write(f"Epoch {epoch + 1} - Train Segmentation Loss: {avg_train_seg_loss:.4f}, mIoU: {train_mIoU:.4f}")

            avg_val_seg_loss, val_mIoU = self._valid_epoch(epoch, val_loader)
            tqdm.write(f"Epoch {epoch + 1} - Validation Segmentation Loss: {avg_val_seg_loss:.4f}, mIoU: {val_mIoU:.4f}")

            self._save_checkpoint(epoch)
