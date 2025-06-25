import os
from PIL import Image

# 최상위 입력 및 출력 디렉토리 설정
input_dir = "C:/workspace/Datas/syn-flare_KITTI_Dataset"
output_dir = "D:/KITTI/Datas/resize/KITTI_flare"
start_coord = (0, 0)  # 크롭 시작 좌표 (x, y)
crop_size = (1152, 352)  # 크롭 크기 (width, height)

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

def crop_and_save_images(input_dir, output_dir, start_coord, crop_size):
    for root, _, files in os.walk(input_dir):  # 모든 하위 폴더 탐색
        for file in files:
            if file.endswith(".png"):  # PNG 파일만 처리
                # 원본 파일 경로와 저장할 파일 경로 설정
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)  # 입력 폴더 구조를 유지
                save_dir = os.path.join(output_dir, relative_path)
                os.makedirs(save_dir, exist_ok=True)  # 출력 폴더 생성
                output_path = os.path.join(save_dir, file)

                # 이미지 크롭
                with Image.open(input_path) as img:
                    width, height = img.size
                    x_start, y_start = start_coord
                    crop_width, crop_height = crop_size

                    # 크롭 범위 계산
                    if x_start + crop_width > width or y_start + crop_height > height:
                        print(f"Skipping {file}: crop size exceeds image dimensions.")
                        continue

                    cropped_img = img.crop((
                        x_start, y_start,
                        x_start + crop_width,
                        y_start + crop_height
                    ))

                    # 크롭된 이미지 저장
                    cropped_img.save(output_path)

crop_and_save_images(input_dir, output_dir, start_coord, crop_size)
