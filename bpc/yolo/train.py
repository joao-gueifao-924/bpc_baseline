import os
from ultralytics import YOLO
import torch
import argparse
import sys
import shutil
from datetime import datetime

def format_experiment_base_name(model_variant, task="detection"):
    """Formats the experiment base name, e.g., train_detection_medium."""
    return f"train_{task}_{model_variant}"

def find_latest_checkpoint(project_dir, model_variant, task="detection"):
    """
    Finds the path to the latest 'last.pt' checkpoint for a given model variant and task.
    """
    experiment_base_name = format_experiment_base_name(model_variant, task)
    candidate_checkpoint_paths = []

    if not os.path.exists(project_dir):
        print(f"Project directory '{project_dir}' not found for checkpoints.")
        return None

    all_subdirs = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]

    for subdir_name in all_subdirs:
        # Check if the subdir_name is the base name itself or starts with base_name + number suffix
        if subdir_name == experiment_base_name or \
           (subdir_name.startswith(experiment_base_name) and \
            subdir_name[len(experiment_base_name):].isdigit() and \
            len(subdir_name) > len(experiment_base_name)):
            potential_ckpt_path = os.path.join(project_dir, subdir_name, "weights", "last.pt")
            if os.path.exists(potential_ckpt_path):
                candidate_checkpoint_paths.append((subdir_name, potential_ckpt_path))

    if not candidate_checkpoint_paths:
        print(f"No candidate checkpoints found for base name '{experiment_base_name}' in '{project_dir}'.")
        return None

    # Sort by full directory name to find the latest (Ultralytics appends numbers like '2', '3')
    candidate_checkpoint_paths.sort(key=lambda x: x[0])
    latest_checkpoint_path = candidate_checkpoint_paths[-1][1]

    print(f"Found latest checkpoint for model variant '{model_variant}' (task: {task}) at: {latest_checkpoint_path} (from dir: {candidate_checkpoint_paths[-1][0]})")
    return latest_checkpoint_path

def train_yolo_model(
    data_yaml_path,
    model_variant="medium", # e.g., "nano", "small", "medium", "large", "xlarge"
    epochs=100,
    imgsz=1280,
    batch_size=16,
    task="detection",
    project_dir="runs/detect",
    continue_training=False,
    workers=8
):
    """
    Trains a YOLO model with specified configurations, disabling all augmentations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Determine pretrained model filename based on variant
    # Assuming "YOLO11" naming convention like yolo11n.pt, yolo11m.pt
    model_short_variant = model_variant[0] # n, s, m, l, x
    base_pretrained_model_filename = f"yolo11{model_short_variant}.pt"
    if task == "segmentation": # segmentation models have different suffix
         base_pretrained_model_filename = f"yolo11{model_short_variant}-seg.pt"

    model_to_load = base_pretrained_model_filename
    current_experiment_name = format_experiment_base_name(model_variant, task)

    if not os.path.exists(data_yaml_path):
        print(f"Error: Dataset YAML file not found at {data_yaml_path}")
        return False # failure

    if continue_training:
        latest_ckpt_path = find_latest_checkpoint(project_dir, model_variant, task)
        if latest_ckpt_path:
            print(f"Resuming training: Loading latest checkpoint from {latest_ckpt_path}.")
            model_to_load = latest_ckpt_path
        else:
            print(f"Continue_training=True, but no checkpoint found for '{current_experiment_name}'. "
                  f"Starting new training from base pretrained model: {base_pretrained_model_filename}.")
    else:
        print(f"Starting new training from base pretrained model: {base_pretrained_model_filename}.")

    print(f"Initializing YOLO model with: {model_to_load}")
    model = YOLO(model_to_load)

    print(f"Starting training: {epochs} epochs, ImgSz: {imgsz}, Batch: {batch_size}")
    print(f"Task: {task}, Model Variant: {model_variant}")
    print(f"Project directory for outputs: {project_dir}")
    print(f"Experiment name: {current_experiment_name}")
    print("Augmentations: ALL DISABLED")

    # Train the model with augmentations explicitly disabled
    # amp=True is default for YOLO recent versions for mixed precision
    training_results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=workers,
        project=project_dir,
        name=current_experiment_name, # Ultralytics will make this unique if it exists (e.g., by appending '2')
        exist_ok=False,  # Recommended to ensure new run unless resuming
        save=True,       # Save model checkpoints
        cache=True,      # Cache images for faster training
        deterministic=False, # Allow for non-deterministic training features
        amp=True,           # Mixed precision training (faster)
        plots=True,         # Save plots
        
        # Disable ALL augmentations
        augment=False,      # Master switch for augmentations
        hsv_h=0.0,          # Hue augmentation (0 means no change)
        hsv_s=0.0,          # Saturation augmentation (0 means no change)
        hsv_v=0.0,          # Value augmentation (0 means no change)
        degrees=0.0,        # Rotation (degrees)
        translate=0.0,      # Translation (fraction of width/height)
        scale=0.0,          # Scale jitter (0 means no scaling from 1.0)
        shear=0.0,          # Shear (degrees)
        perspective=0.0,    # Perspective distortion
        flipud=0.0,         # Vertical flip (probability)
        fliplr=0.0,         # Horizontal flip (probability)
        bgr=0.0,            # BGR augmentation (probability)
        mosaic=0.0,         # Mosaic augmentation (probability, 1.0 for always on)
        mixup=0.0,          # Mixup augmentation (probability)
        copy_paste=0.0,     # Copy-paste augmentation (probability)
        erasing=0.0,        # Erasing augmentation (probability)
    )

    print(f"YOLO training artifacts for this run are in: {training_results.save_dir}")
    
    # Save the final model to a more structured custom path
    # format timestamp as YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    custom_model_save_dir = os.path.join("bpc", "yolo", "models", task, model_variant, timestamp)
    os.makedirs(custom_model_save_dir, exist_ok=False)
    
    final_model_filename = f"yolo11-{task}-{model_variant}-final-{timestamp}.pt"
    final_model_path = os.path.join(custom_model_save_dir, final_model_filename)
    
    # Best model is often saved as best.pt by YOLO, but we can also save the last state
    # model.export(format="pytorch", path=final_model_path) # Preferred way to save final model
    # For simplicity or if export needs more args, model.save() works for .pt
    shutil.copy2(os.path.join(training_results.save_dir, 'weights', 'best.pt'), final_model_path)
    print(f"Training complete. Final 'best.pt' model copied to: {final_model_path}")
    
    # Also save last.pt if needed
    # last_model_path = os.path.join(custom_model_save_dir, f"yolo11-{task}-{model_variant}-last.pt")
    # shutil.copy2(os.path.join(training_results.save_dir, 'weights', 'last.pt'), last_model_path)
    # print(f"Last model 'last.pt' copied to: {last_model_path}")

    return True # success


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model with all augmentations disabled.")
    parser.add_argument("--data_yaml_path", type=str, required=True,
                        help="Path to the dataset YAML file (e.g., bpc/yolo/configs/data_all_objects.yaml).")
    parser.add_argument("--model_variant", type=str, default="medium",
                        choices=["nano", "small", "medium", "large", "xlarge"],
                        help="YOLO model variant to train (default: medium).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs for this run.")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Input image size (square).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--task", type=str, choices=["detection", "segmentation"], default="detection",
                        help="Task type (detection or segmentation).")
    parser.add_argument("--project_dir", type=str, default="runs/detect",
                        help="Base project directory for saving YOLO training runs.")
    parser.add_argument("--continue_training", action="store_true",
                        help="If set, attempts to load the latest 'last.pt' checkpoint to continue training.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of worker threads for data loading.")
    
    args = parser.parse_args()

    if train_yolo_model(
        data_yaml_path=args.data_yaml_path,
        model_variant=args.model_variant,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        task=args.task,
        project_dir=args.project_dir,
        continue_training=args.continue_training,
        workers=args.workers
    ):
        print("Training completed successfully.")
        sys.exit(0) # success
    else:
        print("Training failed.")
        sys.exit(1) # failure

if __name__ == "__main__":
    main()
