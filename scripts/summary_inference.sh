##### Summarization ###### 

# #T, I -> S input preprocess 
python Question_then_Pinpoint/src/input_preprocess.py \
--data_args_path Question_then_Pinpoint/data/InsTaSumm/test/instasumm_test.json \
--step_one_output_path Question_then_Pinpoint/data/InsTaSumm/output/Reasoner_IG_output.json \
--prompt_path Question_then_Pinpoint/prompt/SG_all_0shot.txt \
--task_type SG_w_gen_I \
--save_path Question_then_Pinpoint/data/InsTaSumm/input_output_format/test_data/Summary_inference_w_QtP.json


# zeroshot summarziation (Chat GPT)
python Question_then_Pinpoint/src/gpt_inference.py \
--data_args_path Question_then_Pinpoint/data/InsTaSumm/input_output_format/test_data/Summary_inference_w_QtP.json \
--type zero_shot_summarization \
--save_path Question_then_Pinpoint/data/InsTaSumm/output/Summarizer_output_ChatGPT_w_QtP.json

#auto eval 
python Question_then_Pinpoint/src/auto_evaluation.py \
--raw_test_json_path Question_then_Pinpoint/data/InsTaSumm/test/instasumm_test.json \
--model_output_path Question_then_Pinpoint/data/InsTaSumm/output/Summarizer_output_ChatGPT_w_QtP.json \
--save_path Question_then_Pinpoint/data/InsTaSumm/output/Evaluation_Summarizer_output_ChatGPT_w_QtP.json