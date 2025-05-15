#!/bin/bash

# Script to prepare data and train a single 10-class YOLO model
#
# This script automates the process of:
# 1. Preparing a dataset for YOLO object detection training. This involves
#    running a Python script (`prepare_data.py`) that processes a raw dataset
#    (e.g., images and annotations) and organizes it into the format expected
#    by YOLO, including generating a `data.yaml` file.
# 2. Training a YOLO model using the prepared dataset. This involves running
#    another Python script (`train.py`) with specified training parameters
#    like model variant, number of epochs, batch size, etc.
#
# The script allows overriding default paths and training parameters via
# command-line arguments. It also logs its execution and the output of
# the underlying Python scripts.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Script Configuration ---
# These variables define default paths, script locations, and training parameters.
# They can be overridden by command-line arguments.

# Default path to the raw (unprocessed) dataset.
DEFAULT_RAW_DATASET_PATH="/workspace/bpc_phase2/train_pbr"

# Processed dataset path (output of prepare_data.py, input to YOLO)
# This is where the `prepare_data.py` script will place the YOLO-formatted dataset.
DEFAULT_PROCESSED_DATASET_OUTPUT_PATH="/workspace/datasets/yolo11/all_objects_processed"

# Python scripts location (assuming they are in bpc/yolo/ relative to this script)
PREPARE_DATA_SCRIPT="bpc/yolo/prepare_data.py" # Script for data preparation
TRAIN_SCRIPT="bpc/yolo/train.py"               # Script for model training

# Default training parameters (can be overridden by command-line arguments)
DEFAULT_EPOCHS=100                             # Number of training epochs
DEFAULT_MODEL_VARIANT="medium"                 # YOLO model size (e.g., nano, small, medium, large, xlarge)
DEFAULT_PROJECT_DIR="runs/detect_all_objects"  # YOLO's output directory for storing training runs
DEFAULT_BATCH_SIZE=16                          # Batch size for training
DEFAULT_IMG_SIZE=1280                          # Image size for training (input resolution)
DEFAULT_WORKERS=8                              # Number of worker threads for data loading
DEFAULT_CAMERA_ID=1                            # Default camera ID to process from the raw dataset
DEFAULT_CONTINUE_TRAINING_FLAG=false           # Flag to indicate if training should resume from a checkpoint
DEFAULT_SKIP_DATA_PREP_FLAG=false              # Flag to skip the data preparation step
DEFAULT_LOG_DIR="/workspace/yolo_pipeline_logs" # Directory to store pipeline log files
DEFAULT_YOLO_DATA_YAML_PATH="bpc/yolo/configs/data_all_objects.yaml" # Path to the generated YOLO data configuration file

# Initialize runtime parameters with default values.
# These will be updated if corresponding command-line arguments are provided.
EPOCHS_FOR_RUN="${DEFAULT_EPOCHS}"
MODEL_VARIANT="${DEFAULT_MODEL_VARIANT}"
PROJECT_DIR_RUN="${DEFAULT_PROJECT_DIR}"
BATCH_SIZE_RUN="${DEFAULT_BATCH_SIZE}"
IMG_SIZE_RUN="${DEFAULT_IMG_SIZE}"
WORKERS_RUN="${DEFAULT_WORKERS}"
CAMERA_ID_RUN="${DEFAULT_CAMERA_ID}"
CONTINUE_TRAINING_FLAG="${DEFAULT_CONTINUE_TRAINING_FLAG}"
SKIP_DATA_PREP_FLAG="${DEFAULT_SKIP_DATA_PREP_FLAG}"
YOLO_DATA_YAML_PATH_RUN="${DEFAULT_YOLO_DATA_YAML_PATH}"
RAW_DATASET_PATH_RUN="${DEFAULT_RAW_DATASET_PATH}"
PROCESSED_DATASET_OUTPUT_PATH_RUN="${DEFAULT_PROCESSED_DATASET_OUTPUT_PATH}"
LOG_DIR_RUN="${DEFAULT_LOG_DIR}"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --epochs) EPOCHS_FOR_RUN="$2"; shift ;;
    --model_variant) MODEL_VARIANT="$2"; shift ;;
    --project_dir) PROJECT_DIR_RUN="$2"; shift ;;
    --batch_size) BATCH_SIZE_RUN="$2"; shift ;;
    --img_size) IMG_SIZE_RUN="$2"; shift ;;
    --workers) WORKERS_RUN="$2"; shift ;;
    --camera_id) CAMERA_ID_RUN="$2"; shift ;;
    --yolo_data_yaml_path) YOLO_DATA_YAML_PATH_RUN="$2"; shift ;;
    --raw_dataset_path) RAW_DATASET_PATH_RUN="$2"; shift ;;
    --processed_dataset_output_path) PROCESSED_DATASET_OUTPUT_PATH_RUN="$2"; shift ;;
    --log_dir) LOG_DIR_RUN="$2"; shift ;;
    --continue_training) CONTINUE_TRAINING_FLAG=true ;;
    --skip_data_prep) SKIP_DATA_PREP_FLAG=true ;;
    *) echo "Unknown parameter passed to train-yolo.sh: $1"; exit 1 ;;
  esac
  shift
done

mkdir -p "${LOG_DIR_RUN}"
PIPELINE_LOG_FILE="${LOG_DIR_RUN}/train_pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "YOLO Training Pipeline Started: $(date)"
echo "--------------------------------------------------"

# Redirect all stdout and stderr of this script to the log file and also print to console.
exec > >(tee -i "${PIPELINE_LOG_FILE}") 2>&1

# Print the effective runtime parameters being used for this run.
echo "Runtime Parameters:"
echo "Epochs: ${EPOCHS_FOR_RUN}"
echo "Model Variant: ${MODEL_VARIANT}"
echo "Project Directory: ${PROJECT_DIR_RUN}"
echo "Batch Size: ${BATCH_SIZE_RUN}"
echo "Image Size: ${IMG_SIZE_RUN}"
echo "Workers: ${WORKERS_RUN}"
echo "Camera ID: ${CAMERA_ID_RUN}"
echo "Continue Training: ${CONTINUE_TRAINING_FLAG}"
echo "Skip Data Preparation: ${SKIP_DATA_PREP_FLAG}"
echo "YOLO Data YAML Path: ${YOLO_DATA_YAML_PATH_RUN}"
echo "Raw Dataset Path: ${RAW_DATASET_PATH_RUN}"
echo "Processed Dataset Output Path: ${PROCESSED_DATASET_OUTPUT_PATH_RUN}"
echo "Log Directory: ${LOG_DIR_RUN}"
echo "--------------------------------------------------"


# --- Step 1: Data Preparation ---
# This step processes the raw dataset into a format suitable for YOLO training
# using the `prepare_data.py` script. It can be skipped using the
# `--skip_data_prep` flag.
if [ "$SKIP_DATA_PREP_FLAG" = false ]; then
  echo "Running Data Preparation..."
  echo "Output will be in ${PROCESSED_DATASET_OUTPUT_PATH_RUN}"
  
  mkdir -p "${PROCESSED_DATASET_OUTPUT_PATH_RUN}"
  
  # Execute the data preparation script.
  # The -u flag for python3 ensures unbuffered output, which is good for logging.
  python3 -u "${PREPARE_DATA_SCRIPT}" \
    --dataset_path "${RAW_DATASET_PATH_RUN}" \
    --output_path "${PROCESSED_DATASET_OUTPUT_PATH_RUN}" \
    --camera_id "${CAMERA_ID_RUN}" \
    --yolo_data_yaml_path "${YOLO_DATA_YAML_PATH_RUN}"
  
  echo "Data Preparation complete."
else
  echo "Skipping Data Preparation as per --skip_data_prep flag."
fi

# Sanity check: Ensure the YOLO data YAML file exists.
# This file is crucial for the training step and should be generated by
# `prepare_data.py` or provided correctly if data preparation is skipped.
if [ ! -f "${YOLO_DATA_YAML_PATH_RUN}" ]; then
    echo "ERROR: YOLO data YAML file not found at ${YOLO_DATA_YAML_PATH_RUN} after data preparation step (or if skipped)."
    echo "Please ensure prepare_data.py runs successfully and generates this file, or that the path is correct."
    exit 1
fi
echo "Using YOLO data YAML: ${YOLO_DATA_YAML_PATH_RUN}"
echo "--------------------------------------------------"


# --- Step 2: Training ---
# This step trains the YOLO model using the `train.py` script and the
# prepared dataset.
echo "Running YOLO Model Training..."

# Construct training command arguments for train.py
TRAIN_CMD_ARGS=(
  --data_yaml_path "${YOLO_DATA_YAML_PATH_RUN}"
  --model_variant "${MODEL_VARIANT}"
  --epochs "${EPOCHS_FOR_RUN}"
  --imgsz "${IMG_SIZE_RUN}"
  --batch_size "${BATCH_SIZE_RUN}"
  --task "detection" # Specifies that this is an object detection task
  --project_dir "${PROJECT_DIR_RUN}"
  --workers "${WORKERS_RUN}"
)

# If the continue_training flag is set, add the corresponding argument
# to the training command. The `train.py` script is expected to handle
# finding the latest checkpoint in the project directory.
if [ "$CONTINUE_TRAINING_FLAG" = true ]; then
  echo "Attempting to continue training (train.py will find the latest checkpoint)."
  TRAIN_CMD_ARGS+=(--continue_training)
fi

# Execute the training command.
# The -u flag for python3 ensures unbuffered output.
# "${TRAIN_CMD_ARGS[@]}" expands the array into separate arguments.
python3 -u "${TRAIN_SCRIPT}" "${TRAIN_CMD_ARGS[@]}"

echo "YOLO Model Training complete."
echo "--------------------------------------------------"
echo "YOLO Training Pipeline Finished: $(date)"
echo "Log file for this run: ${PIPELINE_LOG_FILE}"
