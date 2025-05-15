import os
import json
import shutil
import argparse
from tqdm import tqdm
from PIL import Image
import random

def prepare_train_pbr(train_pbr_path, output_path, obj_id):
    """
    Prepares the train_pbr dataset for YOLO, filtering by object ID and
    splitting into training and validation sets (80:20 ratio).
    """

    cameras = ["rgb_cam1", "rgb_cam2", "rgb_cam3"]
    camera_gt_map = {
        "rgb_cam1": "scene_gt_cam1.json",
        "rgb_cam2": "scene_gt_cam2.json",
        "rgb_cam3": "scene_gt_cam3.json"
    }
    camera_gt_info_map = {
        "rgb_cam1": "scene_gt_info_cam1.json",
        "rgb_cam2": "scene_gt_info_cam2.json",
        "rgb_cam3": "scene_gt_info_cam3.json"
    }

    # 1. Prepare directory structure
    train_images_dir = os.path.join(output_path, "train", "images")
    train_labels_dir = os.path.join(output_path, "train", "labels")
    val_images_dir = os.path.join(output_path, "val", "images")
    val_labels_dir = os.path.join(output_path, "val", "labels")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # 2. Collect all image and label data with full paths
    all_data = []
    scene_folders = sorted([
        d for d in os.listdir(train_pbr_path)
        if os.path.isdir(os.path.join(train_pbr_path, d)) and not d.startswith(".")
    ])

    for scene_folder in tqdm(scene_folders, desc="Scanning scenes"):
        scene_path = os.path.join(train_pbr_path, scene_folder)

        for cam in cameras:
            rgb_path = os.path.join(scene_path, cam)
            scene_gt_file = os.path.join(scene_path, camera_gt_map[cam])
            scene_gt_info_file = os.path.join(scene_path, camera_gt_info_map[cam])

            if not (os.path.exists(rgb_path) and os.path.exists(scene_gt_file) and os.path.exists(scene_gt_info_file)):
                continue

            with open(scene_gt_file, "r") as f:
                scene_gt_data = json.load(f)
            with open(scene_gt_info_file, "r") as f:
                scene_gt_info_data = json.load(f)

            num_imgs = len(scene_gt_data)
            for img_id in range(num_imgs):
                img_key = str(img_id)

                # Determine image file path (jpg or png)
                img_file_jpg = os.path.join(rgb_path, f"{img_id:06d}.jpg")
                img_file_png = os.path.join(rgb_path, f"{img_id:06d}.png")
                img_file = img_file_jpg if os.path.exists(img_file_jpg) else img_file_png if os.path.exists(img_file_png) else None

                if img_file is None or img_key not in scene_gt_data or img_key not in scene_gt_info_data:
                    continue

                # Extract valid bounding boxes for the specified object ID
                valid_bboxes = []
                for bbox_info, gt_info in zip(scene_gt_info_data[img_key], scene_gt_data[img_key]):
                    if gt_info["obj_id"] == obj_id and bbox_info["visib_fract"] > 0:
                        valid_bboxes.append(bbox_info["bbox_obj"])

                if not valid_bboxes:
                    continue

                # Generate YOLO label content
                with Image.open(img_file) as img:
                    img_width, img_height = img.size
                
                label_lines = []
                for (x, y, w, h) in valid_bboxes:
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width = w / img_width
                    height = h / img_height
                    label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                # Store all data: image path, label lines, and output base filename
                base_filename = f"{scene_folder}_{cam}_{img_id:06d}"
                all_data.append({
                    "image_path": img_file,
                    "label_lines": label_lines,
                    "base_filename": base_filename
                })

    # 3. Split the data into training and validation sets
    random.shuffle(all_data)
    split_index = int(len(all_data) * 0.8)
    train_data = all_data[:split_index]
    val_data = all_data[split_index:]

    print(f"[INFO] Training data size: {len(train_data)}, Validation data size: {len(val_data)}")

    # 4. Function to write data to the specified directory
    def write_data(data, images_dir, labels_dir, desc):
        for item in tqdm(data, desc=desc):
            image_path = item["image_path"]
            label_lines = item["label_lines"]
            base_filename = item["base_filename"]
            
            # Determine the image extension (.jpg or .png)
            img_extension = os.path.splitext(image_path)[1]
            
            # Construct output file paths
            out_image_path = os.path.join(images_dir, base_filename + img_extension)
            out_label_path = os.path.join(labels_dir, base_filename + ".txt")

            # Copy image and write label file
            shutil.copy(image_path, out_image_path)
            with open(out_label_path, "w") as f:
                f.writelines(label_lines)

    # 5. Write training and validation data
    write_data(train_data, train_images_dir, train_labels_dir, "Writing training data")
    write_data(val_data, val_images_dir, val_labels_dir, "Writing validation data")


def generate_yaml(output_path, obj_id):
    """
    Generates a YOLO .yaml file pointing to the training and validation datasets.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_configs_dir = os.path.join(script_dir, "configs")
    os.makedirs(yolo_configs_dir, exist_ok=True)

    train_path = os.path.abspath(os.path.join(output_path, "train", "images"))
    val_path = os.path.abspath(os.path.join(output_path, "val", "images"))
    yaml_path = os.path.join(yolo_configs_dir, f"data_obj_{obj_id}.yaml")

    yaml_content = {
        "train": train_path,
        "val": val_path,
        "nc": 1,
        "names": [f"object_{obj_id}"]
    }

    with open(yaml_path, "w") as f:
        for key, value in yaml_content.items():
            f.write(f"{key}: {value}\n")

    print(f"[INFO] YAML file generated at: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare the train_pbr dataset for YOLO training.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the train_pbr dataset.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for YOLO dataset.")
    parser.add_argument("--obj_id", type=int, required=True,
                        help="Object ID to filter for.")

    args = parser.parse_args()

    prepare_train_pbr(args.dataset_path, args.output_path, args.obj_id)
    generate_yaml(args.output_path, args.obj_id)

    print("[INFO] Dataset preparation complete!")


if __name__ == "__main__":
    main()
