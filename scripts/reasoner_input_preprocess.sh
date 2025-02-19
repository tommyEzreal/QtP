### Reasoner training data process ###

# (train) training data
python Question_then_Pinpoint/src/input_preprocess.py \
--data_args_path Question_then_Pinpoint/data/InsTaSumm/train/D_prime.json \
--prompt_path Question_then_Pinpoint/prompt/Aspect_focused_QG_prompt.txt \
--task_type Aspect_focused_QG \
--save_path Question_then_Pinpoint/data/InsTaSumm/input_output_format/train_data/Reasoner_aspect_focused_QG_train.json


# ### IG task training data process ###
python Question_then_Pinpoint/src/input_preprocess.py \
--data_args_path Question_then_Pinpoint/data/InsTaSumm/train/D_prime.json \
--prompt_path Question_then_Pinpoint/prompt/Evidence_focused_IG_prompt.txt \
--task_type Evidence_focused_IG \
--save_path Question_then_Pinpoint/data/InsTaSumm/input_output_format/train_data/Reasoner_evidence_focused_IG_train.json


# ### merge and shuffle two instruction tuning set ###
python Question_then_Pinpoint/src/merge_dataset.py \
--QG_data_path Question_then_Pinpoint/data/InsTaSumm/input_output_format/train_data/Reasoner_aspect_focused_QG_train.json \
--QA_data_path Question_then_Pinpoint/data/InsTaSumm/input_output_format/train_data/Reasoner_evidence_focused_IG_train.json \
--save_path Question_then_Pinpoint/data/InsTaSumm/input_output_format/train_data/Reasoner_joint_instruction_tuning_set.json