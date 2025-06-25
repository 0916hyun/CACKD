import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import cv2

img_dir = 'flare dir'
weight_path = 'classifier weight dir'

base_save_dir_combined = 'save dir'
os.makedirs(base_save_dir_combined, exist_ok=True)

mask_save_dir = 'save mask dir'
os.makedirs(mask_save_dir, exist_ok=True)

masking_image_dir = 'save mask_combine dir'
os.makedirs(masking_image_dir, exist_ok=True)

model = models.vgg16(weights='IMAGENET1K_V1')

model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 2)
)

model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, original_size, target_class=None):
        self.model.zero_grad()
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        target = output[0, target_class]
        target.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, original_size)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def visualize_cam(img_path, cam, output_path, concat_output_path, gray_output_path):
    image = cv2.imread(img_path)
    original_size = (image.shape[1], image.shape[0])
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_image = heatmap + np.float32(image) / 255
    cam_image = cam_image / np.max(cam_image)

    cv2.imwrite(output_path, np.uint8(255 * cam_image))

    concat_image = np.hstack((image, np.uint8(255 * cam_image)))
    cv2.imwrite(concat_output_path, concat_image)

    gray_cam = np.uint8(255 * cam)
    cv2.imwrite(gray_output_path, gray_cam)

target_layers = [2, 7, 14, 21, 28]

def combine_cams(cams):
    combined_cam = np.ones_like(cams[0])
    for cam in cams:
        combined_cam = np.minimum(combined_cam, cam)  # Combine using element-wise minimum
    combined_cam = combined_cam - np.min(combined_cam)
    combined_cam = combined_cam / np.max(combined_cam)
    return combined_cam

for img_file in os.listdir(img_dir):
    if img_file.endswith('.png'):
        img_path = os.path.join(img_dir, img_file)

        original_image = cv2.imread(img_path)
        original_size = (original_image.shape[1], original_image.shape[0])
        input_image = preprocess_image(img_path).to(device)

        cams = []
        for layer_num, layer in enumerate(model.features):
            if layer_num in target_layers:
                grad_cam = GradCAM(model, layer)
                cam = grad_cam.generate_cam(input_image, original_size)
                cams.append(cam)

        combined_cam = combine_cams(cams)

        mask = combined_cam.copy()
        mask[mask >= 0.18] = 1
        mask[mask < 0.18] = 0

        mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

        masked_image = original_image * (1 - mask_resized[:, :, np.newaxis]) + mask_resized[:, :, np.newaxis] * 255

        combined_output_path = os.path.join(base_save_dir_combined, img_file)
        combined_concat_output_path = os.path.join(base_save_dir_combined, 'concat', img_file)
        combined_gray_output_path = os.path.join(base_save_dir_combined, 'highres_cam', img_file)
        os.makedirs(os.path.dirname(combined_concat_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(combined_gray_output_path), exist_ok=True)
        visualize_cam(img_path, combined_cam, combined_output_path, combined_concat_output_path, combined_gray_output_path)

        mask_output_path = os.path.join(mask_save_dir, img_file)
        cv2.imwrite(mask_output_path, np.uint8(255 * mask))

        masked_image_output_path = os.path.join(masking_image_dir, img_file)
        cv2.imwrite(masked_image_output_path, masked_image)
