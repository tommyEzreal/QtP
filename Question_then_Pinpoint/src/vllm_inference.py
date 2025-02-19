# vllm_inference.py
from vllm import LLM, SamplingParams
import json
import argparse
from transformers import AutoTokenizer
# import wandb
import re

from table_linearize import IndexedRowTableLinearize
parser = argparse.ArgumentParser()
parser.add_argument("--test_data_path", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--type", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)

args = parser.parse_args()

# wandb.init(project= "TableReasoning")


def dict_sort_by_id(item_list):
    sorted_list = sorted((item for item in item_list if item['id'].isdigit()), key=lambda x: int(x['id']))
    return sorted_list


with open(args.test_data_path, "r") as f:
    test = json.load(f)
    test = test['data']




if __name__ == "__main__":



    merged_model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, verbose=False)
    
    max_tokens = 2048 
    top_p = 1
    temperature = 1
    sampling_params = SamplingParams(n = 1, max_tokens = max_tokens, top_p = top_p, temperature = temperature)
    llm = LLM(merged_model_path)

    all_input_prompts = [example['input'] for example in test]
    print("all_input_prompts:", len(all_input_prompts))

    model_outputs = llm.generate(all_input_prompts, sampling_params)

    assert len(all_input_prompts) == len(model_outputs), "size mismtach between input-output"


    if args.type == "insight_generation":
        for output, test_example in zip(model_outputs, test):
            output_text = output.outputs[0].text
            test_example['raw_response'] = output_text
        
            answer_match = re.search(r'\(Answer\):\s*(.*)', output_text)
            if answer_match:
                answer = answer_match.group(1)
            else:
                answer = "."
                print("Answer not found.")

            
            test_example['response'] = answer

    else:
        for output, test_example in zip(model_outputs, test):
            output_text = output.outputs[0].text
            test_example['response'] = output_text


    if args.type == "insight_generation":
        unique_ids = []
        aggregation = {}
        for item in test:
            if item['id'] not in unique_ids:
                unique_ids.append(item['id'])
                fact_instance = item['response']
                item['response'] = []
                item['response'].append(fact_instance)
                

                # raw_responses (with cell evidence)
                raw_answer = item['raw_response']
                item['raw_response'] = []
                item['raw_response'].append(raw_answer)

                aggregation[item['id']] = item

            else:
                fact_instance = item['response']
                raw_answer = item['raw_response']
                aggregation[item['id']]['response'].append(fact_instance)
                aggregation[item['id']]['raw_response'].append(raw_answer)
                
        parsed_collected_predictions = []
        for idx in list(aggregation.keys()):
            parsed_collected_predictions.append(aggregation[idx])
        
        test = parsed_collected_predictions

        

    else: # QG  # SG # baseline #ours
        pass


    with open(args.save_path, "w") as f:
        json.dump(test, f, indent=4)
