import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from ignite.engine import Engine
from ignite.metrics import SSIM, PSNR
from utils.inception import InceptionV3
from utils.metrics import calculate_fretchet
from tqdm import tqdm
import csv
import segmentation_models_pytorch as smp
from losses.losses import *
import torchvision.utils as vutils
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch.nn.utils import spectral_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_inference(model, img, device, target_size=(352, 480)):

    B, _, H, W = img.shape
    img_resized = torch.nn.functional.interpolate(img, size=target_size, mode='bilinear', align_corners=False).to(
        device)

    preds_resized = model(img_resized)

    preds = torch.nn.functional.interpolate(preds_resized, size=(H, W), mode='bilinear', align_corners=False)

    return preds

def resize_inference_for_kitti(model, img, device, target_size=(352, 1152)):

    B, _, H, W = img.shape
    img_resized = torch.nn.functional.interpolate(img, size=target_size, mode='bilinear', align_corners=False).to(
        device)

    preds_resized = model(img_resized)

    preds = torch.nn.functional.interpolate(preds_resized, size=(H, W), mode='bilinear', align_corners=False)

    return preds

class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(Discriminator, self).__init__()

        self.use_sigmoid = use_sigmoid

        def apply_spectral_norm(layer):
            if use_spectral_norm:
                return spectral_norm(layer)  # spectral_norm 적용
            else:
                return layer  # 그냥 레이어 반환

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

        # 가중치 초기화 호출
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
        """가중치 초기화 메서드"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class Trainer(nn.Module):
    def __init__(self, save_root, save_model_weights_path, save_val_results_path, device, lr, num_epochs, num_classes, label_color_map):
        super(Trainer, self).__init__()

        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.label_color_map = label_color_map
        self.save_val_results_path = save_val_results_path  # 이미지 저장 경로
        self.save_model_weights_path = save_model_weights_path  # 모델 가중치 저장 경로
        os.makedirs(self.save_val_results_path, exist_ok=True)
        os.makedirs(self.save_model_weights_path, exist_ok=True)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.incepv3 = InceptionV3([block_idx]).to(device)

        # PSNR, SSIM, FID 계산용 Engine
        self.evaluator = Engine(self.eval_step)
        SSIM(data_range=1.0).attach(self.evaluator, 'ssim')
        PSNR(data_range=1.0).attach(self.evaluator, 'psnr')

        # Unet 모델 설정
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet50",         # ResNet50 encoder 사용
            encoder_weights="imagenet",      # ImageNet으로 사전 학습된 가중치 사용
            in_channels=10,                  # 총 10개의 입력 채널 (손상 이미지, cam, mask, 합성된 이미지)
            classes=3,                       # 출력 채널: 복원된 이미지(RGB, 3채널)
        ).to(device)

        self.discriminator = Discriminator(in_channels=3, use_sigmoid=True).to(device)

        self.optimizer_G = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        # 손실 함수 설정
        self.l1_loss = nn.L1Loss().to(self.device)
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.style_loss = StyleLoss().to(self.device)
        self.edge_loss = EdgeLoss().to(self.device)
        self.bce_loss = nn.BCELoss().to(self.device)

        self.train_log_path = os.path.join(self.save_model_weights_path, "train_log.csv")
        self.valid_log_path = os.path.join(self.save_model_weights_path, "valid_log.csv")
        self.test_log_path = os.path.join(self.save_model_weights_path, "test_log.csv")

        self._initialize_csv(self.train_log_path)
        self._initialize_csv(self.valid_log_path)
        self._initialize_csv(self.test_log_path)

    def _initialize_csv(self, filepath):

        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "gen_loss", "total_loss", "psnr", "ssim", "fid"])

    def _log_to_csv(self, filepath, epoch, gen_loss, total_loss, psnr, ssim, fid=None, disc_loss=None):

        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            if disc_loss is not None:
                writer.writerow([epoch + 1, gen_loss, total_loss, psnr, ssim, fid, disc_loss])
            else:
                writer.writerow([epoch + 1, gen_loss, total_loss, psnr, ssim, fid])


    def eval_step(self, engine, batch):
        # 이미 학습에서 combined_input을 처리했으므로 간단하게 batch 반환
        return batch

    def _train_epoch(self, epoch, train_loader):
        self.model.train()
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_total_loss = 0
        num_batches = len(train_loader)

        # tqdm으로 배치 진행 상황 출력
        to_pil_image = transforms.ToPILImage()
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{self.num_epochs} - Training", unit="batch") as pbar:
            for i, data in enumerate(train_loader):
                input_img = data[0].to(self.device)
                clean = data[1].to(self.device)
                cam = data[3].to(self.device)
                mask = data[4].to(self.device)
                fr_masked_image = data[5].to(self.device)

                # images = [input_img[0], clean[0], cam[0], mask[0], fr_masked_image[0]]
                # titles = ['Input Image', 'Clean Image', 'CAM', 'Mask', 'FR Masked Image']
                # fig, axs = plt.subplots(1, 5, figsize=(15, 5))  # 1행 5열의 서브플롯
                #
                # for ax, img, title in zip(axs, images, titles):
                #     img_pil = to_pil_image(img.cpu())  # 텐서를 PIL 이미지로 변환
                #     ax.imshow(img_pil)
                #     ax.set_title(title)
                #     ax.axis('off')  # 축 숨김
                #
                # plt.show()  # 한 번에 5장의 이미지 출력
                #
                # pbar.update(1)


                input_img_clone = input_img * cam

                # 손상 이미지, cam, mask, fr_masked_image를 채널 방향으로 결합
                combined_input = torch.cat([input_img, input_img_clone, mask, fr_masked_image], dim=1)
                #combined_input = torch.cat([input_img, input_img_clone, mask], dim=1)
                # 모델 예측 (복원된 이미지 생성)
                final_out = self.model(combined_input)

                # 복원된 이미지를 손상된 원본 이미지에 더해 최종 복원된 이미지 생성
                #final_image = final_out + input_img
                final_image = final_out

                self.optimizer_D.zero_grad()
                dis_real, _ = self.discriminator(clean)
                dis_fake, _ = self.discriminator(final_out.detach())

                real_labels = torch.ones_like(dis_real)  # Discriminator 출력과 동일한 크기로 맞춤
                fake_labels = torch.zeros_like(dis_fake)



                dis_real_loss = self.bce_loss(dis_real, real_labels)
                dis_fake_loss = self.bce_loss(dis_fake, fake_labels)

                disc_loss = (dis_real_loss + dis_fake_loss) / 2
                disc_loss.backward()
                self.optimizer_D.step()



                self.optimizer_G.zero_grad()

                # 손실 계산
                perceptual_loss_ = self.perceptual_loss(final_image, clean)
                style_loss_ = self.style_loss(final_image, clean)
                edge_loss_ = self.edge_loss(final_image, clean)

                dis_fake_for_gen, _ = self.discriminator(final_out)
                gen_gan_loss = self.bce_loss(dis_fake_for_gen, real_labels)

                gen_loss = 30 * (style_loss_) + perceptual_loss_ + edge_loss_ + gen_gan_loss + tv_loss(final_out, 2e-6)

                total_loss = gen_loss

                total_loss.backward()
                self.optimizer_G.step()

                self.evaluator.run([[final_image, clean]])
                metrics = self.evaluator.state.metrics
                epoch_psnr += metrics['psnr']
                epoch_ssim += metrics['ssim']

                # 손실 기록
                epoch_gen_loss += gen_loss.item()
                epoch_total_loss += total_loss.item()
                epoch_disc_loss += disc_loss.item()

                pbar.set_postfix({'gen_loss': gen_loss.item(), 'disc_loss': disc_loss.item()})
                pbar.update(1)

        # 평균 손실 계산
        avg_gen_loss = epoch_gen_loss / num_batches
        avg_disc_loss = epoch_disc_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        avg_psnr = epoch_psnr / len(train_loader)
        avg_ssim = epoch_ssim / len(train_loader)

        self._log_to_csv(self.train_log_path, epoch, avg_gen_loss, avg_total_loss, avg_psnr, avg_ssim, disc_loss=avg_disc_loss)

        return avg_gen_loss, avg_total_loss, avg_psnr, avg_ssim, avg_disc_loss

    def _valid_epoch(self, epoch, val_loader):
        self.model.eval()
        epoch_val_gen_loss = 0
        epoch_val_total_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_fid = 0
        num_batches = len(val_loader)

        with torch.no_grad():
            with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{self.num_epochs} - Validation",
                      unit="batch") as pbar:
                for i, data in enumerate(val_loader):
                    input_img = data[0].to(self.device)
                    clean = data[1].to(self.device)
                    cam = data[3].to(self.device)
                    mask = data[4].to(self.device)
                    fr_masked_image = data[5].to(self.device)

                    input_img_clone = input_img * cam
                    combined_input = torch.cat([input_img, input_img_clone, mask, fr_masked_image], dim=1)
                    #combined_input = torch.cat([input_img, input_img_clone, mask], dim=1)

                    final_out = self.model(combined_input)
                    final_image = final_out

                    perceptual_loss_ = self.perceptual_loss(final_image, clean)
                    style_loss_ = self.style_loss(final_image, clean)
                    edge_loss_ = self.edge_loss(final_image, clean)

                    gen_loss = 30 * (style_loss_) + perceptual_loss_ + edge_loss_ + tv_loss(final_image, 2e-6)

                    self.evaluator.run([[final_image, clean]])  # `final_image` 사용
                    metrics = self.evaluator.state.metrics
                    epoch_psnr += metrics['psnr']
                    epoch_ssim += metrics['ssim']

                    # FID 계산
                    fid_value = calculate_fretchet(final_image, clean, self.incepv3)
                    epoch_fid += fid_value

                    total_loss = gen_loss

                    # 손실 기록
                    epoch_val_gen_loss += gen_loss.item()
                    epoch_val_total_loss += total_loss.item()

                    last_val_input_img = input_img
                    last_val_final_image = final_image
                    last_val_clean = clean

                    # tqdm 진행 바 업데이트
                    pbar.set_postfix({'val_gen_loss': gen_loss.item(), 'val_total_loss': total_loss.item()})
                    pbar.update(1)

        # 평균 손실 계산
        avg_val_gen_loss = epoch_val_gen_loss / num_batches
        avg_val_total_loss = epoch_val_total_loss / num_batches
        avg_psnr = epoch_psnr / len(val_loader)
        avg_ssim = epoch_ssim / len(val_loader)
        avg_fid = epoch_fid / len(val_loader)

        self._save_last_image(epoch, last_val_input_img, last_val_final_image, last_val_clean)

        self._log_to_csv(self.valid_log_path, epoch, avg_val_gen_loss, avg_val_total_loss, avg_psnr, avg_ssim, avg_fid)

        return avg_val_gen_loss, avg_val_total_loss, avg_psnr, avg_ssim, avg_fid

    def _save_last_image(self, epoch, input_img, final_image, clean):
        """에포크 마지막 배치의 복원된 이미지를 저장"""
        result_dir = self.save_val_results_path  # 저장 경로
        os.makedirs(result_dir, exist_ok=True)

        # 이미지 저장 경로 설정
        save_path = os.path.join(result_dir, f'epoch_{epoch + 1}_final.png')

        # 동일한 인덱스의 이미지를 사용하기 위해 첫 번째 이미지를 선택
        input_img_0 = input_img[0].unsqueeze(0)  # 첫 번째 입력 이미지
        final_image_0 = final_image[0].unsqueeze(0)  # 첫 번째 복원된 이미지
        clean_0 = clean[0].unsqueeze(0)  # 첫 번째 깨끗한 이미지

        # 첫 번째 이미지의 입력, 출력, 정답 이미지를 함께 저장
        torchvision.utils.save_image(
            torch.cat([input_img_0, final_image_0, clean_0], dim=0), save_path, nrow=3, normalize=False
        )

        print(f"복원된 이미지를 {save_path}에 저장했습니다.")

    def _save_checkpoint(self, epoch):
        """Generator와 Discriminator의 모델 가중치와 옵티마이저 상태를 체크포인트로 저장"""
        checkpoint = {
            'epoch': epoch + 1,  # 현재 에포크
            'gen_model_state_dict': self.model.state_dict(),  # Generator 가중치
            'dis_model_state_dict': self.discriminator.state_dict(),  # Discriminator 가중치
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),  # Generator 옵티마이저 상태
            'optimizer_D_state_dict': self.optimizer_D.state_dict()  # Discriminator 옵티마이저 상태
        }
        save_path = os.path.join(self.save_model_weights_path, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, save_path)
        print(f"모델 체크포인트를 {save_path}에 저장했습니다.")

    def _load_checkpoint(self, checkpoint_path):
        """저장된 체크포인트로부터 Generator와 Discriminator, 옵티마이저 상태 복원"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['gen_model_state_dict'])  # Generator 가중치 복원
        self.discriminator.load_state_dict(checkpoint['dis_model_state_dict'])  # Discriminator 가중치 복원
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])  # Generator 옵티마이저 복원
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])  # Discriminator 옵티마이저 복원
        start_epoch = checkpoint['epoch']
        print(f"모델 체크포인트 {checkpoint_path}로부터 로드했습니다. 이어서 {start_epoch} 에포크부터 학습을 재개합니다.")
        return start_epoch

    def test(self, epoch, test_loader):
        # 테스트 기록 초기화
        epoch_test_psnr = 0
        epoch_test_ssim = 0
        epoch_test_fid = 0

        print(f"Epoch {epoch + 1} - Test 시작...")

        # 모델을 평가 모드로 설정
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                test_input, test_clean, test_cam, test_mask, test_fr_masked_image, test_seg, file_name = data
                test_input = test_input.to(self.device)
                test_clean = test_clean.to(self.device)
                test_cam = test_cam.to(self.device)
                test_mask = test_mask.to(self.device)
                test_fr_masked_image = test_fr_masked_image.to(self.device)

                combined_input = torch.cat([test_input, test_input * test_cam, test_mask, test_fr_masked_image], dim=1)
                test_out = resize_inference_for_kitti(self.model, combined_input, device, target_size=(352, 1152)) #camvid 480 kitti 1152
                #test_out = self.model(combined_input)

                self.evaluator.run([[test_out, test_clean]])
                metrics = self.evaluator.state.metrics
                epoch_test_psnr += metrics['psnr']
                epoch_test_ssim += metrics['ssim']

                test_fid = calculate_fretchet(test_out, test_clean, self.incepv3)
                epoch_test_fid += test_fid

                #file_name = file_name[0] if isinstance(file_name, tuple) else file_name
                #torchvision.utils.save_image(test_out, os.path.join(self.save_val_results_path, file_name + '.png'))

        avg_test_psnr = epoch_test_psnr / len(test_loader)
        avg_test_ssim = epoch_test_ssim / len(test_loader)
        avg_test_fid = epoch_test_fid / len(test_loader)

        self._log_to_csv(self.test_log_path, epoch, None, None, avg_test_psnr, avg_test_ssim, avg_test_fid)

        print(f"Epoch {epoch + 1} - Test PSNR: {avg_test_psnr:.4f}, SSIM: {avg_test_ssim:.4f}, FID: {avg_test_fid:.4f}")

    def train(self, train_loader, val_loader, test_loader, resume_from_checkpoint=None):

        if resume_from_checkpoint:
            start_epoch = self._load_checkpoint(resume_from_checkpoint)
        else:
            start_epoch = 0

        # Generator와 Discriminator의 학습 로그를 기록하기 위한 메트릭 설정
        train_metrics = {"gen_loss": [], "total_loss": [], "psnr": [], "ssim": [], "disc_loss": []}
        valid_metrics = {"gen_loss": [], "total_loss": [], "psnr": [], "ssim": [], "fid": [], "disc_loss": []}

        for epoch in range(start_epoch, self.num_epochs):
            tqdm.write(f"Epoch {epoch + 1}/{self.num_epochs} 시작...")  # tqdm.write()로 에포크 시작 알림

            # Training step
            avg_train_gen_loss, avg_train_total_loss, avg_psnr, avg_ssim, avg_disc_loss = self._train_epoch(epoch,
                                                                                                            train_loader)
            # tqdm.write()로 에포크별 훈련 결과 출력
            tqdm.write(
                f"Epoch {epoch + 1} - Train Total Loss: {avg_train_total_loss:.4f}, Gen Loss: {avg_train_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

            # Validation step
            avg_val_gen_loss, avg_val_total_loss, avg_psnr, avg_ssim, avg_fid = self._valid_epoch(epoch, val_loader)
            # tqdm.write()로 에포크별 검증 결과 출력
            tqdm.write(
                f"Epoch {epoch + 1} - Validation Total Loss: {avg_val_total_loss:.4f}, Gen Loss: {avg_val_gen_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, FID: {avg_fid:.4f}")

            # Train metrics 로그 업데이트
            train_metrics["gen_loss"].append(avg_train_gen_loss)
            train_metrics["total_loss"].append(avg_train_total_loss)
            train_metrics["psnr"].append(avg_psnr)
            train_metrics["ssim"].append(avg_ssim)
            train_metrics["disc_loss"].append(avg_disc_loss)  # Discriminator의 훈련 손실 추가

            # Valid metrics 로그 업데이트에서 Discriminator 손실 제거
            valid_metrics["gen_loss"].append(avg_val_gen_loss)
            valid_metrics["total_loss"].append(avg_val_total_loss)
            valid_metrics["psnr"].append(avg_psnr)
            valid_metrics["ssim"].append(avg_ssim)
            valid_metrics["fid"].append(avg_fid)

            # Test 단계 추가
            if epoch + 1 >= 1000:
                self.test(epoch, test_loader)

            # 체크포인트 저장
            self._save_checkpoint(epoch)




# from thop import profile
# import torch
# import segmentation_models_pytorch as smp
#
# def calculate_flops_params(model, input_tensor):
#     # thop으로 FLOPs와 파라미터 수 계산
#     macs, params = profile(model, inputs=(input_tensor,))  # 입력을 튜플로 전달
#     flops = macs * 2  # FLOPs는 MACs의 2배
#
#     # Giga(MACs, FLOPs)와 Mega(Params) 단위로 변환하여 출력
#     macs_in_giga = macs / 1e9
#     flops_in_giga = flops / 1e9
#     params_in_mega = params / 1e6
#
#     print(f"MACs: {macs_in_giga:.2f} GMac")
#     print(f"FLOPs: {flops_in_giga:.2f} GFlops")
#     print(f"Parameters: {params_in_mega:.2f} M")
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# model = smp.UnetPlusPlus(
#     encoder_name="resnet50",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=12
# ).to(device)
#
# input_tensor = torch.randn(1, 3, 352, 480).to(device)
#
# from torchinfo import summary
#
# summary(model, input_size=(1, 3, 352, 480))
#
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Conv2d):
#         print(f"Layer: {name}")
#         print(f"Stride: {module.stride}, Padding: {module.padding}")
# #
# # calculate_flops_params(model, input_tensor)
