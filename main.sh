METHOD="DRP"
DATASET="Books"  # Options: Books, CDs_and_Vinyl
MODEL_DEEPSEEK="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_QWEN="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="./output"
BATCH_SIZE=8
MAX_LENGTH=2048

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Method: $METHOD"
echo "Dataset: $DATASET"
echo "=========================================="

# Step 0: Prepare differential inputs
echo "Step 0: Preparing differential inputs..."
python 0_prepare_diff_inputs.py \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR

# Step 1a: Generate DeepSeek outputs
echo "Step 1a: Running DeepSeek model..."
python 1_get_diff_outputs_deepseek.py \
    --model_name $MODEL_DEEPSEEK \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH

# Step 1b: Generate Qwen outputs
echo "Step 1b: Running Qwen model..."
python 1_get_diff_outputs_qwen.py \
    --model_name $MODEL_QWEN \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH

# Step 2: Process final outputs
echo "Step 2: Processing final outputs..."
python 2_get_final_outputs.py \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR

# Step 3: Evaluate results
echo "Step 3: Evaluating results..."
python 3_eval_basic.py \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR

echo "=========================================="
echo "Experiment completed!"
echo "=========================================="