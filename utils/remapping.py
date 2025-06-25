import cv2
import numpy as np
import os

# 기존 HP_COLOR_MAP
HP_COLOR_MAP = {
    0:  [128, 128, 128],  # Sky
    1:  [128, 0, 0],      # Building
    2:  [192, 192, 128],  # Pole
    3:  [128, 64, 128],   # Road
    4:  [0, 0, 192],      # Sidewalk
    5:  [128, 128, 0],    # Tree
    6:  [192, 128, 128],  # SignSymbol
    7:  [64, 64, 128],    # Fence
    8:  [64, 0, 128],     # Car
    9:  [64, 64, 0],      # Pedestrian
    10: [0, 128, 192],    # Bicyclist
    11: [0, 0, 0]         # Void
}

# 새로운 색상 맵
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

id_to_label = {
    0: "Sky", 1: "Building", 2: "Pole", 3: "Road", 4: "Sidewalk",
    5: "Vegetation", 6: "SignSymbol", 7: "Fence", 8: "Car",
    9: "Pedestrian", 10: "Bicyclist", 11: "Void"
}

remap_dict = {
    tuple(HP_COLOR_MAP[class_id]): label_color_map[id_to_label[class_id]]
    for class_id in HP_COLOR_MAP
}

def remap_segmentation_image(image_path):
    image = cv2.imread(image_path)  # BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_image = np.zeros_like(image_rgb)

    for old_rgb, new_rgb in remap_dict.items():
        mask = np.all(image_rgb == old_rgb, axis=-1)
        output_image[mask] = new_rgb

    output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    return output_bgr

def remap_all_images_in_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            result_image = remap_segmentation_image(input_path)
            cv2.imwrite(output_path, result_image)
            print(f"✅ {filename} → 변환 완료")

# 사용 예시
input_folder = "원본_세그_이미지_폴더경로"
output_folder = "재매핑_결과_저장_폴더경로"

remap_all_images_in_folder(input_folder, output_folder)
