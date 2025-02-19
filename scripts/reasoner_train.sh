# sh scripts/reasoner_train.sh
# LLAMA instruction tuning ##

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=6
# export WANDB_PROJECT="TableReasoning"
dataset_name='Question_then_Pinpoint/data/InsTaSumm/input_output_format/train_data/Reasoner_joint_instruction_tuning_set.json'
base_model_name="meta-llama/Llama-2-7b-hf"
model_last_name="meta-llama-2-7b-hf"
model_path_to_be_saved="./Question_then_Pinpoint/qlora/checkpoints/meta-llama-2-7b-hf/QtP_Reasoner_joint_trained"
# export WANDB_NAME=$model_path_to_be_saved

python Question_then_Pinpoint/qlora/qlora.py \
--output_dir ${model_path_to_be_saved} \
--model_name_or_path $base_model_name \
--use_auth \
--logging_steps 10 \
--save_strategy steps \
--data_seed 999 \
--save_steps 100 \
--save_total_limit 10 \
--max_new_tokens 2048 \
--dataloader_num_workers 1 \
--group_by_length \
--logging_strategy steps \
--remove_unused_columns False \
--max_steps 8000 \
--do_train \
--lora_r 64 \
--lora_alpha 16 \
--lora_modules all \
--double_quant \
--quant_type nf4 \
--bf16 \
--bits 4 \
--warmup_ratio 0.03 \
--lr_scheduler_type constant \
--gradient_checkpointing \
--dataset $dataset_name \
--source_max_len 2048 \
--target_max_len 2048 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--learning_rate 2e-4 \
--adam_beta2 0.999 \
--max_grad_norm 0.3 \
--lora_dropout 0.1 \
--weight_decay 0.0 \
--seed 0 \
--dataset_format input-output \
--train_on_source False \
--do_predict False \
# --report_to wandb \

# merge model
python Question_then_Pinpoint/qlora/merge.py \
--base_model_name_or_path meta-llama/Llama-2-7b-hf \
--peft_model_path Question_then_Pinpoint/qlora/checkpoints/meta-llama-2-7b-hf/QtP_Reasoner_joint_trained/checkpoint-8000