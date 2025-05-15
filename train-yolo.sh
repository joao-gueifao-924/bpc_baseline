#!/bin/bash

# Script to iterate from 0 to 9 and run YOLO data preparation and training.
# Logs output for each object ID to a separate, sequentially numbered file.
# Allows continuing training from the latest checkpoint if specified.

# Exit immediately if a command exits with a non-zero status.
set -e

# Create a directory for log files if it doesn't already exist
LOG_DIR="/workspace/yolo_processing_logs"
mkdir -p "${LOG_DIR}"

# --- Script Configuration ---
# Default epochs for each training run. Can be overridden by a command-line argument.
DEFAULT_EPOCHS=20
# Default project directory for YOLO runs
PROJECT_DIR="runs/detect"
# --- End Script Configuration ---

# --- Parse Command-Line Arguments ---
# Initialize flags/variables
CONTINUE_TRAINING_FLAG=false
EPOCHS_FOR_RUN="${DEFAULT_EPOCHS}"

# Process arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --continue_training) CONTINUE_TRAINING_FLAG=true; echo "Flag --continue_training is set. Will attempt to load latest checkpoints."; shift ;;
        --epochs) EPOCHS_FOR_RUN="$2"; echo "Number of epochs for this run set to: ${EPOCHS_FOR_RUN}"; shift; shift ;;
        --project_dir) PROJECT_DIR="$2"; echo "YOLO project directory set to: ${PROJECT_DIR}"; shift; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done
# --- End Parse Command-Line Arguments ---

echo "Starting YOLO data preparation and training loop..."
echo "Log files will be stored in the '${LOG_DIR}' directory."
echo "YOLO project directory for runs: '${PROJECT_DIR}'"
echo "Epochs for each training session in this script run: ${EPOCHS_FOR_RUN}"
if [ "$CONTINUE_TRAINING_FLAG" = true ]; then
  echo "Script will attempt to continue training from latest checkpoints."
else
  echo "Script will start new training sessions (or from base models if no checkpoints specified to load)."
fi
echo "=================================================="

# Loop through integers from 0 to 9 (inclusive)
# You can adjust this range if you have more than 10 objects.
# For example, for 100 objects: for N in $(seq 0 99)
for N in {0..9}
do
  # Format object ID with leading zeros for consistency with prepare_data.py if needed
  # If prepare_data.py expects simple numbers like 0, 1, 2, keep ${N}
  # If it also needs formatted IDs, adjust accordingly.
  # For this example, we assume prepare_data.py uses plain ${N}.
  FORMATTED_N_LOG=$(printf "%06d" ${N}) # For log file naming consistency

  echo "---"
  echo "Processing for Object ID (N): ${N} (Formatted for logs: ${FORMATTED_N_LOG})"
  echo "------------------------------------"

  # Define output and data paths using the current value of N
  # Ensure these paths are correctly formatted if your other scripts expect leading zeros
  OUTPUT_PATH_DATA_PREP="/workspace/datasets/yolo11/train_obj_${N}" # Assuming this script uses plain N
  DATA_PATH_TRAIN_CONFIG="/workspace/bpc_baseline/bpc/yolo/configs/data_obj_${N}.yaml" # Assuming this uses plain N

  # --- Logic for sequentially numbered log files ---
  # Using formatted N for log file names
  BASE_LOG_NAME="${LOG_DIR}/obj_${FORMATTED_N_LOG}_processing"
  LOG_EXT=".log"
  ACTUAL_LOG_FILE="${BASE_LOG_NAME}${LOG_EXT}"
  COUNTER=1
  while [ -f "${ACTUAL_LOG_FILE}" ]; do
    ACTUAL_LOG_FILE="${BASE_LOG_NAME}_${COUNTER}${LOG_EXT}"
    COUNTER=$((COUNTER + 1))
  done
  # --- End of logic for sequentially numbered log files ---

  echo "Starting processing for Object ID ${N} at $(date)" > "${ACTUAL_LOG_FILE}"
  echo "Python script outputs will be logged to: ${ACTUAL_LOG_FILE}"
  echo "" >> "${ACTUAL_LOG_FILE}"


  # Command 1: Data Preparation
  echo "Step 1: Running data preparation for Object ID ${N}..."
  python3 -u bpc/yolo/prepare_data.py --dataset_path "/workspace/bpc_phase2/train_pbr" \
          --output_path "${OUTPUT_PATH_DATA_PREP}" --obj_id ${N} &>> "${ACTUAL_LOG_FILE}"
  echo "Data preparation for Object ID ${N} logged."
  echo ""


  # Command 2: Training
  echo "Step 2: Running training for Object ID ${N}..."
  # Base arguments for train.py
  TRAIN_CMD_ARGS=(
      --obj_id "${N}" # train.py expects the plain integer ID
      --data_path "${DATA_PATH_TRAIN_CONFIG}"
      --epochs "${EPOCHS_FOR_RUN}" # Use the epochs value determined at the start
      --imgsz 1280
      --batch 16
      --task detection
      --project "${PROJECT_DIR}" # Pass the determined project directory
  )

  # Conditionally add the --continue_training flag
  if [ "$CONTINUE_TRAINING_FLAG" = true ]; then
    echo "Attempting to continue training for Object ID ${N} (train.py will find the latest checkpoint)."
    TRAIN_CMD_ARGS+=(--continue_training)
  else
    echo "Starting new training session (or from base model) for Object ID ${N}."
  fi

  # Execute the training command
  python3 -u bpc/yolo/train.py "${TRAIN_CMD_ARGS[@]}" &>> "${ACTUAL_LOG_FILE}"

  echo "Training for Object ID ${N} logged."
  echo "=================================================="
done


echo ""
echo "All iterations completed."
echo "Check the '${LOG_DIR}' directory for individual log files."
