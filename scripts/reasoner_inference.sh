# Reasoner inference #

# Aspect & Qusetion Inference ### 
python Question_then_Pinpoint/src/input_preprocess.py \
--data_args_path Question_then_Pinpoint/data/InsTaSumm/test/instasumm_test.json \
--prompt_path Question_then_Pinpoint/prompt/Aspect_focused_QG_prompt.txt \
--task_type Aspect_focused_QG \
--save_path Question_then_Pinpoint/data/InsTaSumm/input_output_format/test_data/Reasoner_aspect_focused_QG_inference.json

# Question inference
python Question_then_Pinpoint/src/vllm_inference.py \
--test_data_path Question_then_Pinpoint/data/InsTaSumm/input_output_format/test_data/Reasoner_aspect_focused_QG_inference.json \
--model_path Question_then_Pinpoint/qlora/checkpoints/meta-llama-2-7b-hf/QtP_Reasoner_joint_trained/checkpoint-8000-merged \
--type Question_then_Pinpoint \
--save_path Question_then_Pinpoint/data/InsTaSumm/output/Reasoner_QG_output.json


## Evidence & Insight Inference ### 
# bring Aspect, Qusetion pair from step one output
python Question_then_Pinpoint/src/input_preprocess.py \
--data_args_path Question_then_Pinpoint/data/InsTaSumm/test/instasumm_test.json \
--step_one_output_path Question_then_Pinpoint/data/InsTaSumm/output/Reasoner_QG_output.json \
--prompt_path Question_then_Pinpoint/prompt/Evidence_focused_IG_prompt.txt \
--task_type Evidence_focused_IG_inference \
--save_path Question_then_Pinpoint/data/InsTaSumm/input_output_format/test_data/Reasoner_evidence_focused_IG_inference.json

#inference
python Question_then_Pinpoint/src/vllm_inference.py \
--test_data_path Question_then_Pinpoint/data/InsTaSumm/input_output_format/test_data/Reasoner_evidence_focused_IG_inference.json \
--model_path Question_then_Pinpoint/qlora/checkpoints/meta-llama-2-7b-hf/QtP_Reasoner_joint_trained/checkpoint-8000-merged \
--type insight_generation \
--save_path Question_then_Pinpoint/data/InsTaSumm/output/Reasoner_IG_output.json
