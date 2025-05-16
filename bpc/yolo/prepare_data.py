import os
import json
import shutil
import argparse
from types import SimpleNamespace
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import cv2
import sys
import yaml
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
import multiprocessing
from multiprocessing import cpu_count

# Global constant for depth processing
MAX_GRAD_ABS_VALUE = 5000.0  # mm

def transform_depth_image(depth_image, depth_image_scale, max_depth_mm, background_factor=1.1):
    depth_image = depth_image.copy()
    if len(depth_image.shape) and depth_image.shape[-1] == 3:
        depth_image = depth_image[:,:,0] # get only one channel, they are all the same

    depth_image = depth_image.astype(np.float32) * depth_image_scale

    # In the PBR training data, many times background depth is 0.0, but we want to map it to be behind the target objects.
    # That's why we are mapping anything below 10 mm or above 5 meters to the maximum original depth value plus some margin.
    depth_image[(depth_image < 10) & (depth_image > max_depth_mm)] = 0 # anything below 10 mm or above 5 meters gets mapped to zero
    depth_image_max_distance = np.max(depth_image)
    depth_image[depth_image == 0] = depth_image_max_distance * background_factor
    
    return depth_image



def hillshade(elevation, azimuth=135, altitude=45, cellsize=10):
    azimuth_rad = np.radians(360 - azimuth + 90)
    altitude_rad = np.radians(altitude)

    # Compute the gradient
    dy, dx = np.gradient(elevation, cellsize, cellsize)

    # Slope and aspect
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    aspect = np.where(aspect < 0, 2 * np.pi + aspect, aspect)

    # Hillshade calculation
    shaded = (np.sin(altitude_rad) * np.cos(slope) +
              np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))

    shaded = np.clip(shaded, 0, 1)
    return (shaded * 255).astype(np.uint8)


def apply_noise_sim2real(depth_image):
    # Apply Gaussian noise to the depth image:
    # depth_image is expected to be single-channel2400x2400 float32, with values in the range [0.0, 65535.0].

    depth_image = depth_image.copy()
    if len(depth_image.shape) and depth_image.shape[-1] == 3:
        depth_image = depth_image[:,:,0] # get only one channel, they are all the same

    #depth_image = depth_image.astype(np.float32)
    #depth_image = (255.0 * (depth_image / 255.0)).astype(np.uint8)

    #depth_image = cv2.medianBlur(depth_image, 3)
    depth_image = depth_image.astype(np.float32)
    # OpenCV cv2.medianBlur only supports uint8 images.
    # So we need to convert the depth image to uint8, apply the median blur, and then convert back to float32.
    #depth_image = depth_image.astype(np.uint8)
    #depth_image = cv2.medianBlur(depth_image, 7)
   # depth_image = depth_image.astype(np.float32)

    # Emulate quantization noise:
    L = 10
    depth_image = np.floor(depth_image / L) * L

    # Apply a Gaussian blur to the noise to make it more realistic (introduce spatial correlation)
    noise1 = np.random.normal(0, 10, depth_image.shape).astype(np.float32)
    #noise1 = cv2.GaussianBlur(noise1, (5, 5), 0)

    noise2 = np.random.normal(0, 70, depth_image.shape).astype(np.float32)
    noise2 = cv2.GaussianBlur(noise2, (31, 31), 0)

    # Add the noise to the depth image:
    depth_image = depth_image + noise2


    # Apply salt and pepper noise:
    if True:
        salt_noise_probability = 0.02
        pepper_noise_probability = 0.02
        random_image = np.random.random(depth_image.shape).astype(np.float32) # uniform random noise in the range [0, 1]
        pepper_mask = random_image < pepper_noise_probability
        depth_image[pepper_mask] = 0        
        salt_mask = random_image > (1 - salt_noise_probability)
        depth_image[salt_mask] = 65535
    
    return depth_image


def hillshade_depth_image(depth_image, new_width=1280, azimuth=135, is_synthetic=False):
    """
    depth_image is expected to be 2400x2400 uint16, with values in the range [0, 65535].
    """
    elevation = depth_image.copy()
    if len(elevation.shape) and elevation.shape[-1] == 3:
        elevation = elevation[:,:,0] # get only one channel, they are all the same

    elevation = np.array(elevation).astype(np.float32)
    
    if is_synthetic:
        elevation = apply_noise_sim2real(elevation)
    else:     
        pass
        #elevation = cv2.bilateralFilter(elevation, 7, 255, 100) # these values were tuned empirically for 2400x2400 images

    
    hillshade_img = hillshade(elevation, azimuth=azimuth)
        
    # resize depth_image to 1280xR, where R is the aspect ratio of the original image:
    h, w = hillshade_img.shape
    aspect_ratio = w / h
    new_height = int(new_width / aspect_ratio)
    hillshade_img = cv2.resize(hillshade_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    if is_synthetic:
        hillshade_img = cv2.medianBlur(hillshade_img, 9)
        # apply sharpening filter:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        hillshade_img = cv2.filter2D(hillshade_img, -1, kernel)
        hillshade_img = cv2.filter2D(hillshade_img, -1, kernel)
    
    return hillshade_img


def compose_grey_plus_hillshade_depth_image(img_gray_np, img_depth_np_raw_pixel_values, new_width=1280):

    hillshade_img_0 = hillshade_depth_image(img_depth_np_raw_pixel_values, new_width=new_width, azimuth=0, is_synthetic=True)
    hillshade_img_135 = hillshade_depth_image(img_depth_np_raw_pixel_values, new_width=new_width, azimuth=135, is_synthetic=True)

    w,h = hillshade_img_0.shape
    img_gray_np = cv2.resize(img_gray_np, (w,h), interpolation=cv2.INTER_LINEAR)

    composed_img_np = np.stack((img_gray_np, hillshade_img_0, hillshade_img_135), axis=-1)
    return composed_img_np




def compose_grey_lograd_depth_image(img_gray_np, img_depth_np_raw_pixel_values, depth_scale_pixel_to_mm=0.1, max_depth_mm=5000.0):
    """
    Concatenates greyscale image with gradients of depth image into a 3-channel image.
    lograd stands for log-gradient operation.
    Processes depth image with Sobel gradients and compresses them to 0-255 range for 0-500 mm range,
    using log-compression with clipping so that they can be written to disk and fed to YOLO model as 8-bit PNG images.
    The log-compression is done with floor(41.0 * log(x + 1)) to map 0-500 mm to 0-255 range.

    Args:
        img_gray_np: Grayscale image as numpy array
        img_depth_np_raw_pixel_values: Raw depth image as numpy array
        depth_scale_pixel_to_mm: Scale factor to convert depth pixels to mm
        max_depth_mm: Maximum depth value in mm
        
    Returns:
        3-channel numpy array with (Grayscale, f(SobelX), f(SobelY)) channels. where f(x) = floor(41.0 * log(x + 1))
    """
    # Transform depth image
    img_depth_np = transform_depth_image(img_depth_np_raw_pixel_values, depth_image_scale=depth_scale_pixel_to_mm, max_depth_mm=max_depth_mm)

    # Calculate Sobel gradients on depth image
    sobel_x = cv2.Sobel(img_depth_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_depth_np, cv2.CV_64F, 0, 1, ksize=3)

    # Compress Sobel gradients to 0-255 range for 0-500 mm range, with clipping
    sobel_x_compressed = np.floor(41.0 * np.log(sobel_x + 1))
    sobel_y_compressed = np.floor(41.0 * np.log(sobel_y + 1))
    sobel_x_clipped = np.clip(sobel_x_compressed, 0, 255).astype(np.uint8)                
    sobel_y_clipped = np.clip(sobel_y_compressed, 0, 255).astype(np.uint8)

    # Stack to create 3-channel image: (Grayscale, SobelX, SobelY)
    return np.stack((img_gray_np, sobel_x_clipped, sobel_y_clipped), axis=-1)


def _process_and_save_item(image_task_meta, images_dir, labels_dir):
    """
    Loads, processes, and saves a single image and its corresponding label file.
    """
    rgb_img_file = image_task_meta["rgb_img_file"]
    depth_img_file = image_task_meta["depth_img_file"]
    depth_scale_pixel_to_mm = image_task_meta["depth_scale_pixel_to_mm"]
    raw_annotations = image_task_meta["raw_annotations"]
    base_filename = image_task_meta["base_filename"]
    # scene_folder_name = image_task_meta["scene_folder_name"] # For more specific error messages if needed
    # cam_rgb_folder_name = image_task_meta["cam_rgb_folder_name"] # For more specific error messages if needed
    # img_id_str = image_task_meta["img_id_str"] # For more specific error messages if needed


    try:
        # Load RGB and depth images, convert RGB to grayscale
        img_gray_np = cv2.imread(rgb_img_file, flags=cv2.IMREAD_UNCHANGED)
        img_height, img_width = img_gray_np.shape[:2]

        if len(img_gray_np.shape) == 3:
            img_gray_np = img_gray_np[..., 0] # Convert to single channel

        # Load depth image as uint16 and take first channel if multi-channel
        img_depth_np = cv2.imread(depth_img_file, flags=cv2.IMREAD_UNCHANGED)
        if len(img_depth_np.shape) == 3:
            img_depth_np = img_depth_np[..., 0] # Convert to single channel
        
        if img_depth_np.shape != (img_height, img_width):
            print(f"Failed to process {base_filename}: Depth image dimensions ({img_depth_np.shape}) mismatch RGB ({img_height, img_width}).", file=sys.stderr)
            return False # Item processing failure

    except Exception as e:
        print(f"Error processing image files for {base_filename}: {e}", file=sys.stderr)
        return False # Item processing failure

    composed_img_np = compose_grey_plus_hillshade_depth_image(img_gray_np, img_depth_np, new_width=1280)


    # Generate label lines
    label_lines = []
    for ann in raw_annotations:
        obj_id = ann["obj_id"]
        x, y, w, h = ann["bbox_obj"]
        # Normalized center coordinates and dimensions are calculated here
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        label_lines.append(f"{obj_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

    out_image_path = os.path.join(images_dir, base_filename + ".png")
    out_label_path = os.path.join(labels_dir, base_filename + ".txt")

    # Save composed image
    try:
        # Reorder channels for OpenCV: (Gray, SobelX, SobelY) -> (SobelY, SobelX, Gray)
        # Gray is composed_img_np[..., 0]
        # SobelX is composed_img_np[..., 1]
        # SobelY is composed_img_np[..., 2]
        # OpenCV expects BGR, so B=SobelY, G=SobelX, R=Gray
        composed_img_bgr = composed_img_np[..., ::-1] # Reverses the last dimension (channels)
        cv2.imwrite(out_image_path, composed_img_bgr)
    except Exception as e:
        print(f"Error saving image {out_image_path}: {e}", file=sys.stderr)
        return False # Item processing failure

    # Write label file
    with open(out_label_path, "w") as f:
        f.writelines(label_lines)
    
    return True


def _process_item_wrapper(args_tuple):
    """Helper function to unpack arguments for _process_and_save_item for use with imap."""
    return _process_and_save_item(*args_tuple)


def _process_tasks_and_write_files(tasks_list, images_dir, labels_dir, desc_prefix):
    """
    Iterates through a list of image processing tasks, processes each in parallel,
    and writes the output files.
    Returns True if all tasks are processed successfully, False otherwise.
    """
    if not tasks_list:
        print(f"No tasks to process for {desc_prefix}.", file=sys.stderr)
        return True # No tasks, so technically successful or no work to do.

    num_workers = cpu_count()
    print(f"Using {num_workers} CPU cores for {desc_prefix} processing.")

    # Prepare arguments for processing: list of (task_meta, images_dir, labels_dir)
    args_for_processing = [(task_meta, images_dir, labels_dir) for task_meta in tasks_list]

    results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use tqdm with pool.imap_unordered for better progress bar updates
        with tqdm(total=len(tasks_list), desc=f"Processing and writing {desc_prefix} data") as pbar:
            for result in pool.imap_unordered(_process_item_wrapper, args_for_processing):
                results.append(result)
                pbar.update(1) # Manually update progress bar for each completed task

    if not all(results):
        failed_count = results.count(False)
        print(f"ERROR: {failed_count} item(s) failed during {desc_prefix} processing. Check logs for details.", file=sys.stderr)
        return False # Propagate failure

    return True


def prepare_yolo_dataset(train_pbr_path, output_path, camera_id_to_process, yolo_data_yaml_path):
    """
    Prepares the train_pbr dataset for YOLO training with a single 10-class model.
    - Combines all object IDs into one dataset.
    - Creates 3-channel images:
        - Channel 0: Grayscale from RGB image
        - Channel 1: Sobel X gradient of depth image
        - Channel 2: Sobel Y gradient of depth image
    - Splits data into training and validation sets (80:20 ratio).
    - Generates a single data.yaml file for all 10 classes.
    - Returns True if successful, False otherwise.
    """
    selected_camera_rgb_name = f"rgb_cam{camera_id_to_process}"
    cameras = [selected_camera_rgb_name]
    camera_gt_map = {
        selected_camera_rgb_name: f"scene_gt_cam{camera_id_to_process}.json"
    }
    camera_gt_info_map = {
        selected_camera_rgb_name: f"scene_gt_info_cam{camera_id_to_process}.json"
    }
    camera_intrinsics_map = {
        selected_camera_rgb_name: f"scene_camera_cam{camera_id_to_process}.json"
    }

    # 1. Prepare directory structure
    train_images_dir = os.path.join(output_path, "images", "train")
    train_labels_dir = os.path.join(output_path, "labels", "train")
    val_images_dir = os.path.join(output_path, "images", "val")
    val_labels_dir = os.path.join(output_path, "labels", "val")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    all_image_tasks = []
    scene_folders = sorted([
        d for d in os.listdir(train_pbr_path)
        if os.path.isdir(os.path.join(train_pbr_path, d)) and not d.startswith(".")
    ])

    print(f"Scanning {len(scene_folders)} scene folders in {train_pbr_path}...")
    for scene_folder_name in tqdm(scene_folders, desc="Scanning scenes"):
        scene_path = os.path.join(train_pbr_path, scene_folder_name)
        for cam_rgb_folder_name in cameras:
            rgb_image_folder_path = os.path.join(scene_path, cam_rgb_folder_name)
            cam_depth_folder_name = cam_rgb_folder_name.replace("rgb", "depth")
            depth_image_folder_path = os.path.join(scene_path, cam_depth_folder_name)

            scene_gt_file = os.path.join(scene_path, camera_gt_map[cam_rgb_folder_name])
            scene_gt_info_file = os.path.join(scene_path, camera_gt_info_map[cam_rgb_folder_name])
            scene_camera_file = os.path.join(scene_path, camera_intrinsics_map[cam_rgb_folder_name]) # Path for scene_camera

            if not (os.path.exists(rgb_image_folder_path) and \
                      os.path.exists(depth_image_folder_path) and \
                      os.path.exists(scene_gt_file) and \
                      os.path.exists(scene_gt_info_file) and \
                      os.path.exists(scene_camera_file)): # Check for scene_camera_file
                print(f"Skipping {scene_folder_name}/{cam_rgb_folder_name} due to missing files/folders.", file=sys.stderr)
                return False # failure

            with open(scene_camera_file, "r") as f: # Load scene_camera_data
                scene_camera_data = json.load(f)
            with open(scene_gt_file, "r") as f:
                scene_gt_data = json.load(f)
            with open(scene_gt_info_file, "r") as f:
                scene_gt_info_data = json.load(f)

            # For {cam_id} in {scene_folder_name}, we have:
            # - scene_camera_{cam_id}.json: contains the camera intrinsics and extrinsics calibration data for each image. Key is image ID. Also contains depth_scale.
            # - scene_gt_{cam_id}.json: contains object IDs and their pose relative to the camera {cam_id}. First key is image ID, second key is bounding box ID.
            # - scene_gt_info_{cam_id}.json: contains the bounding box annotations and respective pixel count for each object in the image. First key is image ID, second key is bounding box ID.

            # Check if scene_camera_data, scene_gt_data and scene_gt_info_data have exactly the same keys (image IDs)
            if set(scene_camera_data.keys()) != set(scene_gt_data.keys()) != set(scene_gt_info_data.keys()):
                print(f"Failed on {scene_folder_name}/{cam_rgb_folder_name} due to mismatch in image IDs.", file=sys.stderr)
                return False # failure
            
            # We will iterate based on common image IDs present in both.
            image_ids = scene_camera_data.keys()

            if not image_ids:
                print(f"Failed on {scene_folder_name}/{cam_rgb_folder_name} due to no image IDs.", file=sys.stderr)
                return False # failure

            # Iterate through all image IDs
            for img_id_str in image_ids:
                # annotations_gt_this_image is a dict of bbox_instance_ids to their details
                annotations_gt_this_image = scene_gt_data[img_id_str]
                # annotations_gt_info_this_image is a dict of bbox_instance_ids to their details
                annotations_gt_info_this_image = scene_gt_info_data[img_id_str] 
                camera_params_this_image = scene_camera_data[img_id_str]

                # Extract depth_scale for the current image from scene_camera_data
                depth_scale_pixel_to_mm = camera_params_this_image.get("depth_scale")
                if depth_scale_pixel_to_mm is None:
                    print(f"'depth_scale' value not found for image {img_id_str} in '{scene_camera_file}' (scene {scene_folder_name}, camera {cam_rgb_folder_name}).", file=sys.stderr)
                    return False # failure
                
                # Determine image file paths (jpg or png)
                img_numeric_id = int(img_id_str)
                
                rgb_img_file_jpg = os.path.join(rgb_image_folder_path, f"{img_numeric_id:06d}.jpg")
                rgb_img_file_png = os.path.join(rgb_image_folder_path, f"{img_numeric_id:06d}.png")
                rgb_img_file = rgb_img_file_jpg if os.path.exists(rgb_img_file_jpg) else rgb_img_file_png if os.path.exists(rgb_img_file_png) else None

                depth_img_file_jpg = os.path.join(depth_image_folder_path, f"{img_numeric_id:06d}.jpg")
                depth_img_file_png = os.path.join(depth_image_folder_path, f"{img_numeric_id:06d}.png")
                depth_img_file = depth_img_file_jpg if os.path.exists(depth_img_file_jpg) else depth_img_file_png if os.path.exists(depth_img_file_png) else None

                if rgb_img_file is None or depth_img_file is None:
                    print(f"ERROR: Missing RGB or Depth image for {img_id_str} in {scene_folder_name}/{cam_rgb_folder_name}. Cannot proceed.", file=sys.stderr)
                    return False # failure

                # Ensure bounding box IDs are the same between scene_gt_data and scene_gt_info_data
                # Both scene_gt_data[img_id_str] and annotations_gt_info_this_image are lists of dicts, so we use len() to check if they have same entries
                if len(scene_gt_data[img_id_str]) != len(annotations_gt_info_this_image):
                    print(f"Bounding box IDs mismatch between scene_gt_data and scene_gt_info_data for image {img_id_str}. Scene: {scene_folder_name}, Cam: {cam_rgb_folder_name}", file=sys.stderr)
                    return False # failure

                # Collect raw annotations for this image task
                raw_annotations_for_task = []
                bbox_instance_ids = range(0, len(annotations_gt_this_image)) # Or len(scene_gt_info_data[img_id_str]), they are the same

                for bbox_instance_id in bbox_instance_ids:
                    bbox_details_gt = scene_gt_data[img_id_str][bbox_instance_id]
                    bbox_details_info = annotations_gt_info_this_image[bbox_instance_id]

                    # Combine details
                    # For this dataset structure, obj_id comes from scene_gt_data, bbox_obj from scene_gt_info_data
                    # visib_fract from scene_gt_info_data

                    if bbox_details_info.get("visib_fract", 0) <= 0: # Check visibility from scene_gt_info
                        continue # Not a failure, just skip this annotation.

                    obj_id = bbox_details_gt.get("obj_id") # obj_id from scene_gt
                    if obj_id is None:
                        print(f"Invalid obj_id (is None) found in scene_gt for image {img_id_str}, bbox_instance_id {bbox_instance_id}. Scene: {scene_folder_name}, Cam: {cam_rgb_folder_name}", file=sys.stderr)
                        return False # fatal failure

                    bbox_obj = bbox_details_info.get("bbox_obj") # bbox_obj from scene_gt_info
                    if bbox_obj is None:
                         print(f"Invalid bbox_obj (is None) found in scene_gt_info for image {img_id_str}, bbox_instance_id {bbox_instance_id}. Scene: {scene_folder_name}, Cam: {cam_rgb_folder_name}", file=sys.stderr)
                         return False # fatal failure
                    
                    x, y, w, h = bbox_obj
                    if w <= 0 or h <= 0: # Invalid bbox dimensions
                        print(f"ERROR: Invalid bbox dimensions {w}x{h} for bbox instance {bbox_instance_id} in image {img_id_str}. Scene: {scene_folder_name}, Cam: {cam_rgb_folder_name}. Cannot proceed.", file=sys.stderr)
                        # This is now a fatal error for the whole dataset.
                        return False

                    raw_annotations_for_task.append({
                        "obj_id": obj_id,
                        "bbox_obj": [x, y, w, h] # Store raw bbox
                    })

                # Unique base filename for output
                base_filename = f"{scene_folder_name}_{cam_rgb_folder_name.replace('rgb_', '')}_{img_numeric_id:06d}"
                
                # Add task metadata to the list
                all_image_tasks.append({
                    "rgb_img_file": rgb_img_file,
                    "depth_img_file": depth_img_file,
                    "depth_scale_pixel_to_mm": depth_scale_pixel_to_mm,
                    "raw_annotations": raw_annotations_for_task, # List of dicts
                    "base_filename": base_filename,
                    # Optional: for more detailed error messages later if _process_and_save_item needs them
                    # "scene_folder_name": scene_folder_name,
                    # "cam_rgb_folder_name": cam_rgb_folder_name,
                    # "img_id_str": img_id_str 
                })

    if not all_image_tasks:
        print("No image tasks found to process. Please check dataset paths, content, or potential errors during metadata collection.", file=sys.stderr)
        return False # failure

    # 3. Shuffle and split the task list
    random.shuffle(all_image_tasks)
    split_index = int(len(all_image_tasks) * 0.8)
    train_tasks = all_image_tasks[:split_index]
    val_tasks = all_image_tasks[split_index:]

    print(f"[INFO] Total image tasks collected: {len(all_image_tasks)}")
    print(f"[INFO] Training tasks: {len(train_tasks)}, Validation tasks: {len(val_tasks)}")

    # 4. Process and write training data
    if not _process_tasks_and_write_files(train_tasks, train_images_dir, train_labels_dir, "training"):
        print("[ERROR] Failed during training data processing and writing.", file=sys.stderr)
        return False # Propagate failure

    # 5. Process and write validation data
    if not _process_tasks_and_write_files(val_tasks, val_images_dir, val_labels_dir, "validation"):
        print("[ERROR] Failed during validation data processing and writing.", file=sys.stderr)
        return False # Propagate failure
    
    # 6. Generate YAML file for YOLO training
    generate_yolo_yaml(output_path, num_classes=10, yolo_data_yaml_path=yolo_data_yaml_path)

    return True # success


def generate_yolo_yaml(output_path, num_classes, yolo_data_yaml_path):
    """
    Generates a YOLO .yaml file pointing to the training and validation datasets.
    The YAML file will be named 'data_all_objects.yaml' and placed in a 'configs'
    subdirectory relative to this script's location.
    """

    # Paths in YAML should be absolute or relative to where YOLO `train` command is run.
    # For simplicity, using absolute paths.
    train_img_path = os.path.abspath(os.path.join(output_path, "images", "train"))
    val_img_path = os.path.abspath(os.path.join(output_path, "images", "val"))
    
    # Ensure the directory for the yaml_filepath exists
    os.makedirs(os.path.dirname(yolo_data_yaml_path), exist_ok=True)

    class_names = [f"object_{i}" for i in range(num_classes)]

    yaml_content = {
        "train": train_img_path,
        "val": val_img_path,
        "nc": num_classes,
        "names": class_names
    }

    with open(yolo_data_yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False) # Use yaml.dump for robust YAML generation

    print(f"[INFO] YOLO YAML file generated at: {yolo_data_yaml_path}")
    return yolo_data_yaml_path


def main():
    args = None
    
    DEBUG = True
    if DEBUG:
        args = SimpleNamespace()
        args.dataset_path = "/mnt/061A31701A315E3D/ipd-dataset/bpc_baseline/datasets/phase2/train_pbr"
        args.output_path = "/mnt/061A31701A315E3D/ipd-dataset/phase2-dataset-yolo11-all-objects"
        args.camera_id = 1 # Default for DEBUG mode, can be changed
        args.yolo_data_yaml_path = "bpc/yolo/configs/data_all_objects_debug.yaml" # Default for DEBUG
    else:
        parser = argparse.ArgumentParser(description="Prepare dataset for 10-class YOLO training with 3-channel (Gray, DepthSobelX, DepthSobelY) images.")
        parser.add_argument("--dataset_path", type=str, required=True,
                            help="Path to the root train_pbr dataset (e.g., /workspace/bpc_phase2/train_pbr).")
        parser.add_argument("--output_path", type=str, required=True,
                            help="Output directory for the processed YOLO dataset (e.g., /workspace/datasets/yolo11/all_objects_processed).")
        parser.add_argument("--camera_id", type=int, choices=[1, 2, 3], required=True,
                            help="ID of the camera to process (1, 2, or 3).")
        parser.add_argument("--yolo_data_yaml_path", type=str, required=True,
                            help="Path to save the generated YOLO data YAML file (e.g., bpc/yolo/configs/data_all_objects.yaml).")
        args = parser.parse_args()

    if prepare_yolo_dataset(args.dataset_path, args.output_path, args.camera_id, args.yolo_data_yaml_path):
        print("[INFO] Dataset preparation complete!")
        sys.exit(0) # Exit with success code
    else:
        print("[ERROR] Dataset preparation failed!")
        sys.exit(1) # Exit with error code

if __name__ == "__main__":
    main()
