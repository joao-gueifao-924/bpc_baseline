import os
from ultralytics import YOLO
import torch
import argparse
import glob # For pattern matching

def format_experiment_base_name(obj_id):
    """Formats the experiment base name with 6 leading zeros and '_object' suffix."""
    return f"train_{obj_id:06d}_object"

def find_latest_checkpoint(project_dir, obj_id):
    """
    Finds the path to the latest 'last.pt' checkpoint for a given object ID
    using the new naming convention (e.g., 'train_000000_object').

    Args:
        project_dir (str): The base directory where YOLO runs are saved.
        obj_id (int): The object ID.

    Returns:
        str or None: Path to the latest 'last.pt' if found, otherwise None.
    """
    experiment_base_name = format_experiment_base_name(obj_id)
    candidate_checkpoint_paths = []

    if not os.path.exists(project_dir):
        print(f"Project directory '{project_dir}' not found.")
        return None

    all_subdirs = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]

    for subdir_name in all_subdirs:
        # Check if the subdir_name is the base name itself (e.g., train_000000_object)
        # OR if it starts with the base name and the suffix is purely numeric
        # (e.g., train_000000_object2, train_000000_object10)
        if subdir_name == experiment_base_name or \
           (subdir_name.startswith(experiment_base_name) and \
            subdir_name[len(experiment_base_name):].isdigit() and \
            len(subdir_name) > len(experiment_base_name)): # Ensure there's actually a suffix

            potential_ckpt_path = os.path.join(project_dir, subdir_name, "weights", "last.pt")
            if os.path.exists(potential_ckpt_path):
                candidate_checkpoint_paths.append((subdir_name, potential_ckpt_path))

    if not candidate_checkpoint_paths:
        print(f"No candidate checkpoints found for base name '{experiment_base_name}' in '{project_dir}'.")
        return None

    # Sort by directory name to find the latest.
    # Example: 'train_000000_object', 'train_000000_object2', 'train_000000_object10'
    # Standard string sort will handle this correctly due to the numeric suffix.
    candidate_checkpoint_paths.sort(key=lambda x: x[0])

    latest_checkpoint_path = candidate_checkpoint_paths[-1][1]
    print(f"Found latest checkpoint for obj_id {obj_id} at: {latest_checkpoint_path} (from dir: {candidate_checkpoint_paths[-1][0]})")
    return latest_checkpoint_path


def train_yolo11(task, data_path, obj_id, epochs_for_this_run, imgsz, batch, load_checkpoint_if_exists=False, project="runs/detect"):
    """
    Train YOLO11 using the new experiment naming convention.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    task_suffix = ""
    base_pretrained_model_filename = ""
    if task == "detection":
        base_pretrained_model_filename = "yolo11n.pt"
        task_suffix = "detection"
    elif task == "segmentation":
        base_pretrained_model_filename = "yolo11n-seg.pt"
        task_suffix = "segmentation"
    else:
        print(f"Invalid task: {task}. Must be 'detection' or 'segmentation'.")
        return None

    if not os.path.exists(data_path):
        print(f"Error: Dataset YAML file not found at {data_path}")
        return None

    model_to_load = base_pretrained_model_filename
    current_experiment_base_name = format_experiment_base_name(obj_id) # Use the new formatting

    if load_checkpoint_if_exists:
        latest_ckpt_path = find_latest_checkpoint(project, obj_id)
        if latest_ckpt_path:
            print(f"Latest checkpoint identified at {latest_ckpt_path}. Loading it to continue training further.")
            model_to_load = latest_ckpt_path
        else:
            print(f"No existing checkpoint found for obj_id {obj_id} with base name '{current_experiment_base_name}' in project '{project}'. "
                  f"Starting new training from base model: {base_pretrained_model_filename}.")
    else:
        print(f"Not attempting to load a checkpoint. "
              f"Starting new training from base model: {base_pretrained_model_filename}.")

    print(f"Initializing YOLO model with: {model_to_load}")
    model = YOLO(model_to_load)

    print(f"Starting a training session for {epochs_for_this_run} epochs.")
    print(f"  Task: {task_suffix}, Object ID: {obj_id}")
    print(f"  Project directory for outputs: {project}")
    print(f"  Base experiment name for this object: {current_experiment_base_name}")

    training_results = model.train(
        data=data_path,
        epochs=epochs_for_this_run,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=8,
        save=True,
        project=project,
        name=current_experiment_base_name, # Ultralytics will make this unique (e.g., by adding a numeric suffix like '2')
        exist_ok=False
    )

    print(f"YOLO training artifacts for this run are in: {training_results.save_dir}")

    custom_save_dir = os.path.join("bpc", "yolo", "models", task_suffix, f"obj_{obj_id:06d}") # Also format obj_id here for consistency if desired
    os.makedirs(custom_save_dir, exist_ok=True)
    # Use the new naming scheme for the final saved model as well, for consistency
    final_model_filename = f"yolo11-{task_suffix}-obj_{obj_id:06d}.pt"
    final_model_path = os.path.join(custom_save_dir, final_model_filename)
    
    model.save(final_model_path)

    print(f"Training complete. Final model from this run saved to custom path: {final_model_path}")
    return final_model_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model with updated naming convention.")
    parser.add_argument("--obj_id", type=int, required=True, help="Object ID for training.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for THIS training run.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--task", type=str, choices=["detection", "segmentation"], default="detection", help="Task type.")
    parser.add_argument("--continue_training", action="store_true",
                        help="If set, loads the LATEST 'last.pt' for this obj_id to continue training further.")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Base project directory for saving YOLO training runs.")

    args = parser.parse_args()

    train_yolo11(
        task=args.task,
        data_path=args.data_path,
        obj_id=args.obj_id,
        epochs_for_this_run=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        load_checkpoint_if_exists=args.continue_training,
        project=args.project
    )

if __name__ == "__main__":
    main()
