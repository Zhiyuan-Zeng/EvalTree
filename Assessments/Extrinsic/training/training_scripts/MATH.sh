run_name=$1
dataset_path="Assessments/Extrinsic/training/data/MATH/${run_name}.json"
model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
epoch=3
output_dir="../MATH_checkpoints/${run_name}_[epoch=${epoch}]"

gpus="0"
NUM_GPUS=$(( $(grep -o "," <<< "$gpus" | wc -l) + 1 ))

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${gpus}
torchrun --nproc-per-node=${NUM_GPUS} --master_port=$(expr $RANDOM + 1000) Assessments/Extrinsic/training/train.py \
  --model_name_or_path "${model_name_or_path}" \
  --fp16 False \
  --bf16 True \
  --seed 42 \
  --output_dir "${output_dir}" \
  --dataset_path "${dataset_path}" \
  --num_train_epochs "${epoch}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --eval_steps 1000000000 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 1E-4 \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.1 \
  --weight_decay 0.0 \
  --eval_strategy "steps" \
  --logging_steps 10 \
  --run_name "${run_name}" \
  --tf32 True \
  --ddp_timeout 1800
