import torch.optim as optim
import os
from ignite.engine import Engine
from ignite.metrics import SSIM, PSNR
from utils.inception import InceptionV3
import csv
import segmentation_models_pytorch as smp
from losses.losses import *
import torchvision
from tqdm import tqdm
from torch.nn.utils import spectral_norm
from utils.stream_metrics import StreamSegMetrics
from ptflops import get_model_complexity_info
from custom_module.CAM import *
from custom_module.corrsum import SkipScAggregator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_cxc_matrix(logits, temperature=4.0):

    B, C, H, W = logits.size()

    logits_flat = F.normalize(logits.view(B, C, -1), p=2, dim=-1)

    cxc_matrix = torch.bmm(logits_flat, logits_flat.transpose(1, 2))

    cxc_matrix_normalized = torch.softmax(cxc_matrix / temperature, dim=1)

    return cxc_matrix_normalized


def kd_loss_cxc(student_logits, teacher_logits, temperature=4.0):

    student_cxc = calculate_cxc_matrix(student_logits, temperature)
    teacher_cxc = calculate_cxc_matrix(teacher_logits, temperature)

    loss_fn = nn.L1Loss(reduction='mean')
    loss = loss_fn(student_cxc, teacher_cxc)

    return loss


def resize_inference(model, img, device, n_classes, target_size=(352, 480)):
    B, _, H, W = img.shape
    img_resized = torch.nn.functional.interpolate(img, size=target_size, mode='bilinear', align_corners=False).to(
        device)

    preds_resized = model(img_resized)
    if isinstance(preds_resized, tuple):
        preds_resized = preds_resized[0]

    preds = torch.nn.functional.interpolate(preds_resized, size=(H, W), mode='bilinear', align_corners=False)

    return preds

class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(Discriminator, self).__init__()

        self.use_sigmoid = use_sigmoid

        def apply_spectral_norm(layer):
            if use_spectral_norm:
                return spectral_norm(layer)
            else:
                return layer

        self.conv1_block = nn.Sequential(
            apply_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2_block = nn.Sequential(
            apply_spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3_block = nn.Sequential(
            apply_spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4_block = nn.Sequential(
            apply_spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5_block = nn.Sequential(
            apply_spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self._init_weight()

    def forward(self, x):
        conv1_out = self.conv1_block(x)
        conv2_out = self.conv2_block(conv1_out)
        conv3_out = self.conv3_block(conv2_out)
        conv4_out = self.conv4_block(conv3_out)
        conv5_out = self.conv5_block(conv4_out)

        output = conv5_out
        if self.use_sigmoid:
            output = torch.sigmoid(output)

        return output, [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]

    def _init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class Trainer(nn.Module):

    def __init__(self, save_root, save_model_weights_path, save_val_results_path, device, lr, num_epochs, n_classes, label_color_map, temperature=4.0, teacher_checkpoint_path=None):
        super(Trainer, self).__init__()

        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_classes = n_classes
        self.label_color_map = label_color_map
        self.save_val_results_path = save_val_results_path
        self.save_model_weights_path = save_model_weights_path
        self.temperature = temperature
        os.makedirs(self.save_val_results_path, exist_ok=True)
        os.makedirs(self.save_model_weights_path, exist_ok=True)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.incepv3 = InceptionV3([block_idx]).to(device)

        self.evaluator = Engine(self.eval_step)
        SSIM(data_range=1.0).attach(self.evaluator, 'ssim')
        PSNR(data_range=1.0).attach(self.evaluator, 'psnr')


        self.Tmodel_seg = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        ).to(device)

        if teacher_checkpoint_path:
            checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
            self.Tmodel_seg.load_state_dict(checkpoint['model_state_dict'])
            print(f"T-모델 {teacher_checkpoint_path}  로드 완료")

        self.Smodel_seg = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        ).to(device)

        self.discriminator = Discriminator(in_channels=3, use_sigmoid=True).to(device)
        self.optimizer_S = optim.Adam(self.Smodel_seg.parameters(), lr=self.lr)

        self.metrics = StreamSegMetrics(n_classes)

        self.l1_loss = nn.L1Loss().to(self.device)
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.style_loss = StyleLoss().to(self.device)
        self.edge_loss = EdgeLoss().to(self.device)
        self.bce_loss = nn.BCELoss().to(self.device)

        self.train_log_path = os.path.join(self.save_model_weights_path, "train_log.csv")
        self.valid_log_path = os.path.join(self.save_model_weights_path, "valid_log.csv")
        self.test_log_path = os.path.join(self.save_model_weights_path, "test_log.csv")

        self.skip_sc_aggregator = SkipScAggregator(skip_in_channels=[3, 64, 256, 512, 1024, 2048], base_channel=256).to(self.device)

    def _initialize_csv(self, filepath, labels=None):
        headers = [
            "epoch", "total_loss", "gen_loss", "seg_loss", "disc_loss",
            "psnr", "ssim", "kd_loss_out", "kd_loss_att", "kd_loss_skip", "fid", "mIoU"
        ] + [f"class_{i}_iou" for i in range(self.n_classes)]
        if labels is not None:
            headers.extend([f"label_{i}" for i in range(len(labels))])
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    def _log_to_csv(self, filepath, epoch, total_loss, gen_loss, seg_loss, disc_loss, psnr, ssim, kd_loss_out=None,
                    kd_loss_att=None, kd_loss_skip=None, fid=None, mIoU=None, class_iou=None, labels=None):
        row = [
            epoch + 1,
            total_loss, gen_loss, seg_loss, disc_loss, psnr, ssim,
            kd_loss_out if kd_loss_out is not None else "",
            kd_loss_att if kd_loss_att is not None else "",
            kd_loss_skip if kd_loss_skip is not None else "",
            fid if fid is not None else "",
            mIoU if mIoU is not None else ""
        ]
        if class_iou is None:
            row.extend([0.0] * self.n_classes)
        else:
            row.extend([class_iou.get(i, 0.0) for i in range(self.n_classes)])
        if labels is not None:
            row.extend(labels)
        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def initialize_logs(self):
        self._initialize_csv(self.train_log_path)
        self._initialize_csv(self.valid_log_path)
        self._initialize_csv(self.test_log_path)

    def eval_step(self, engine, batch):
        return batch

    def _train_epoch(self, epoch, train_loader):
        self.Tmodel_seg.eval()
        self.Smodel_seg.train()

        epoch_gen_loss = 0
        epoch_disc_loss = 0
        epoch_kd_loss_out = 0
        epoch_kd_loss_att = 0
        epoch_skip_kd_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_total_loss = 0
        epoch_seg_loss = 0
        num_batches = len(train_loader)
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=11).to(self.device)

        self.metrics.reset()

        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{self.num_epochs} - Training", unit="batch") as pbar:
            for i, data in enumerate(train_loader):
                input_img = data[0].to(self.device)
                clean = data[1].to(self.device)
                cam = data[3].to(self.device)
                mask = data[4].to(self.device)
                fr_masked_image = data[5].to(self.device)
                seg_label = data[6].to(self.device)

                with torch.no_grad():
                    teacher_seg_output, teacher_before_head, teacher_output, teacher_skip = self.Tmodel_seg(clean)

                self.optimizer_S.zero_grad()

                student_seg_output, student_before_head, student_output, student_skip = self.Smodel_seg(input_img)

                seg_loss = self.seg_loss(student_seg_output, seg_label)
                KD_loss_out =nn.L1Loss()(student_seg_output, teacher_seg_output)

                student_logit = student_seg_output
                teacher_logit = teacher_seg_output
                student_prob = F.softmax(student_logit, dim=1)
                teacher_prob = F.softmax(teacher_logit, dim=1)
                KD_loss_out_softmax = F.l1_loss(student_prob, teacher_prob)

                KD_loss_att = nn.L1Loss()(student_output, teacher_output)
                cxc_kd_loss = kd_loss_cxc(student_seg_output, teacher_seg_output)

                teacher_sc = self.skip_sc_aggregator(teacher_skip)
                student_sc = self.skip_sc_aggregator(student_skip)
                KD_loss_skip = nn.L1Loss()(student_sc, teacher_sc)

                KD_loss = KD_loss_out + KD_loss_att + 100*cxc_kd_loss + 10*KD_loss_skip
                total_loss = seg_loss + KD_loss

                total_loss.backward()

                self.optimizer_S.step()

                epoch_seg_loss += seg_loss.item()
                epoch_kd_loss_out += KD_loss_out.item()
                epoch_kd_loss_att += KD_loss_att.item()
                epoch_skip_kd_loss += KD_loss_skip.item()
                epoch_total_loss += total_loss.item()


                preds = student_seg_output.argmax(dim=1).cpu().numpy()
                labels = seg_label.cpu().numpy()
                self.metrics.update(labels, preds)
                pbar.set_postfix({'seg_loss': seg_loss.item(), 'kd_loss': KD_loss.item()})
                pbar.update(1)


        avg_gen_loss = epoch_gen_loss / num_batches
        avg_disc_loss = epoch_disc_loss / num_batches
        avg_seg_loss = epoch_seg_loss / num_batches
        avg_kd_loss_out = epoch_kd_loss_out / num_batches
        avg_kd_loss_att = epoch_kd_loss_att / num_batches
        avg_skip_kd_loss = epoch_skip_kd_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        avg_psnr = epoch_psnr / len(train_loader)
        avg_ssim = epoch_ssim / len(train_loader)
        metrics = self.metrics.get_results()
        mIoU = metrics.get("Mean IoU", 0.0)
        class_iou = metrics.get("Class IoU", {})

        self._log_to_csv(
            self.train_log_path, epoch,
            total_loss=avg_total_loss, gen_loss=0, seg_loss=avg_seg_loss,
            disc_loss=0, psnr=0, ssim=0,
            kd_loss_out=avg_kd_loss_out,
            kd_loss_att=avg_kd_loss_att,
            kd_loss_skip=avg_skip_kd_loss,
            fid=0, mIoU=mIoU, class_iou=class_iou
        )

        return avg_total_loss, avg_gen_loss, avg_seg_loss, avg_disc_loss, avg_kd_loss_out, avg_kd_loss_att, avg_psnr, avg_ssim, mIoU, class_iou

    def _valid_epoch(self, epoch, val_loader):
        self.Tmodel_seg.eval()
        self.Smodel_seg.eval()
        epoch_val_gen_loss = 0
        epoch_val_seg_loss = 0
        epoch_val_kd_loss_out = 0
        epoch_val_kd_loss_att = 0
        epoch_val_kd_loss_skip = 0
        epoch_val_total_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_fid = 0
        num_batches = len(val_loader)

        self.metrics.reset()

        with torch.no_grad():
            with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{self.num_epochs} - Validation",
                      unit="batch") as pbar:
                for i, data in enumerate(val_loader):
                    input_img = data[0].to(self.device)
                    clean = data[1].to(self.device)
                    cam = data[3].to(self.device)
                    mask = data[4].to(self.device)
                    fr_masked_image = data[5].to(self.device)
                    seg_label = data[6].to(self.device)

                    teacher_seg_output, teacher_before_head, teacher_output, teacher_skip = self.Tmodel_seg(clean)
                    final_image = input_img
                    student_seg_output, student_before_head, student_output, student_skip = self.Smodel_seg(final_image)

                    if seg_label.dim() == 4:
                        seg_label = seg_label.argmax(dim=1)
                    elif seg_label.dim() == 3 and seg_label.shape[1] == 1:
                        seg_label = seg_label.squeeze(1)

                    seg_label = seg_label.long()

                    seg_loss = self.seg_loss(student_seg_output, seg_label)

                    KD_loss_out = nn.L1Loss()(student_seg_output, teacher_seg_output)
                    KD_loss_att = nn.L1Loss()(student_output, teacher_output)
                    cxc_kd_loss = kd_loss_cxc(student_seg_output, teacher_seg_output)

                    teacher_sc = self.skip_sc_aggregator(teacher_skip)
                    student_sc = self.skip_sc_aggregator(student_skip)
                    KD_loss_skip = nn.L1Loss()(student_sc, teacher_sc)

                    KD_loss = KD_loss_out + KD_loss_att + 100*cxc_kd_loss + 10*KD_loss_skip
                    total_loss = seg_loss + KD_loss

                    preds = student_seg_output.argmax(dim=1).cpu().numpy()
                    labels = seg_label.cpu().numpy()
                    self.metrics.update(labels, preds)

                    epoch_val_seg_loss += seg_loss.item()
                    epoch_val_kd_loss_out += KD_loss_out.item()
                    epoch_val_kd_loss_att += KD_loss_att.item()
                    epoch_val_kd_loss_skip += KD_loss_skip.item()
                    epoch_val_total_loss += total_loss.item()

                    last_val_input_img = input_img
                    last_val_final_image = final_image
                    last_val_clean = clean

                    pbar.set_postfix({'val_total_loss': total_loss.item()})
                    pbar.update(1)

        avg_val_gen_loss = epoch_val_gen_loss / num_batches
        avg_val_seg_loss = epoch_val_seg_loss / num_batches
        avg_val_kd_loss_out = epoch_val_kd_loss_out / num_batches
        avg_val_kd_loss_att = epoch_val_kd_loss_att / num_batches
        avg_val_kd_loss_skip = epoch_val_kd_loss_skip / num_batches
        avg_val_total_loss = epoch_val_total_loss / num_batches
        avg_psnr = epoch_psnr / len(val_loader)
        avg_ssim = epoch_ssim / len(val_loader)
        avg_fid = epoch_fid / len(val_loader)

        metrics = self.metrics.get_results()
        mIoU = metrics.get("Mean IoU", 0.0)
        class_iou = metrics.get("Class IoU", {})

        self._log_to_csv(
            self.valid_log_path, epoch,
            total_loss=avg_val_total_loss, gen_loss=avg_val_gen_loss, seg_loss=avg_val_seg_loss,
            disc_loss=None, psnr=avg_psnr, ssim=avg_ssim,
            kd_loss_out=avg_val_kd_loss_out,
            kd_loss_att=avg_val_kd_loss_att,
            kd_loss_skip=avg_val_kd_loss_skip,
            fid=avg_fid, mIoU=mIoU, class_iou=class_iou
        )

        return avg_val_total_loss, avg_val_gen_loss, avg_val_seg_loss, avg_val_kd_loss_out, avg_val_kd_loss_att, avg_psnr, avg_ssim, avg_fid, mIoU, class_iou


    def _save_checkpoint(self, epoch):

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.Smodel_seg.state_dict(),
            'optimizer_S_state_dict' : self.optimizer_S.state_dict()
        }
        save_path = os.path.join(self.save_model_weights_path, f'last_epoch.pth')
        torch.save(checkpoint, save_path)
        print(f"모델 체크포인트를 {save_path}에 저장했습니다.")

    def _load_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path)
        self.Smodel_seg.load_state_dict(checkpoint['seg_model_state_dict'])
        self.optimizer_S.load_state_dict(checkpoint['optimizer_S_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"모델 체크포인트 {checkpoint_path}로부터 로드했습니다. 이어서 {start_epoch} 에포크부터 학습을 재개합니다.")
        return start_epoch


    def train(self, train_loader, val_loader, test_loader=None, resume_from_checkpoint=None):
        if resume_from_checkpoint:
            start_epoch = self._load_checkpoint(resume_from_checkpoint)
        else:
            start_epoch = 0

        best_val_loss = float('inf')
        best_val_mIoU = 0.0
        best_val_loss_epoch = 0

        for epoch in range(start_epoch, self.num_epochs):
            tqdm.write(f"Epoch {epoch + 1}/{self.num_epochs} 시작...")

            avg_total_loss, avg_gen_loss, avg_seg_loss, avg_disc_loss, \
            avg_kd_loss_out, avg_kd_loss_att, avg_psnr, avg_ssim, train_mIoU, class_iou = self._train_epoch(epoch,
                                                                                                            train_loader)
            tqdm.write(f"Epoch {epoch + 1} - Train Segmentation Loss: {avg_total_loss:.4f}, mIoU: {train_mIoU:.4f}")

            avg_val_total_loss, avg_val_gen_loss, avg_val_seg_loss, avg_val_kd_loss_out, \
            avg_val_kd_loss_att, avg_psnr, avg_ssim, avg_fid, val_mIoU, class_iou = self._valid_epoch(epoch, val_loader)
            tqdm.write(
                f"Epoch {epoch + 1} - Validation Segmentation Loss: {avg_val_total_loss:.4f}, mIoU: {val_mIoU:.4f}")

            if avg_val_seg_loss < best_val_loss:
                best_val_loss = avg_val_seg_loss
                best_val_loss_epoch = epoch + 1
                save_path = os.path.join(self.save_model_weights_path, 'val_loss_min.pth')
                torch.save({'model_state_dict': self.Smodel_seg.state_dict()}, save_path)
                tqdm.write(f"최저 Validation Loss 모델 저장: {save_path}")

            self._save_checkpoint(epoch)

        print("\n=== 학습 종료: Best Model 정보 ===")
        print(f"- 최저 Validation Loss 모델: Epoch {best_val_loss_epoch}")
        print("=================================")


