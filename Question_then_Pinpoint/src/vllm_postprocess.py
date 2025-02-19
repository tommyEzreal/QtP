# vllm_inference.py
from vllm import LLM, SamplingParams
import json
import argparse
from transformers import AutoTokenizer
import wandb
parser = argparse.ArgumentParser()
parser.add_argument("--vllm_output_path", type=str, required=True)
parser.add_argument("--parsed_output_path", type=str, required=True)
args = parser.parse_args()

wandb.init(project= "TableReasoning")

with open(args.vllm_output_path, "r") as f:
    vllm_output = json.load(f)


if __name__ == "__main__":
    
    for item in vllm_output:
        whole_response = item['response']

        start_index = whole_response.find("(Paragraph-form Summary):") + len("(Paragraph-form Summary):")
        summary = whole_response[start_index:].strip()
        item['response'] = summary
    
        
    with open(args.parsed_output_path, "w") as f:
        json.dump(vllm_output, f, indent=4)
