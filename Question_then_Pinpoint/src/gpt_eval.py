import numpy as np
from tqdm import tqdm
from langchain.llms import OpenAIChat, OpenAI
from langchain.chat_models import ChatOpenAI, openai
import json
import argparse
import asyncio
from tqdm.asyncio import tqdm_asyncio
import random
import re
import os

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from pathlib import Path

from table_linearize import IndexedRowTableLinearize


random.seed(8989)

parser = argparse.ArgumentParser()
parser.add_argument("--raw_test_json_path", type = str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--data_args_path", type=str, required=True)
parser.add_argument("--comparing_data_args_path", type=str)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--max_token", type=int, default=1000)
parser.add_argument("--type", type=str)
args = parser.parse_args()


with open(args.raw_test_json_path, "r") as f:
    raw_test_json = json.load(f)

with open(args.data_args_path, "r") as f:
    data_args = json.load(f)

if args.comparing_data_args_path:
    with open(args.comparing_data_args_path, "r") as f:
        comparing_data_args = json.load(f)

assert len(raw_test_json) == len(data_args), "size mismatch: Check if data args are sampled !"

with open(args.prompt, "r") as f:
    prompt = f.read()

def dict_sort_by_id(item_list):
    sorted_list = sorted((item for item in item_list if item['id'].isdigit()), key=lambda x: int(x['id']))
    return sorted_list

def extract_number(input_string):
    match = re.search(r'\d+', input_string)
    if match:
        return match.group(0)
    return None

# GPT input instance cache
all_model_inputs = []

if (args.type == 'table_coverage_eval') | (args.type == 'faithfulness_eval') | (args.type == 'analytic_depth'):
    model_name ='gpt-4-turbo-2024-04-09'
    print(f"{args.type}...{model_name}")
    print("instance_num_sampled:", len(data_args))


    linearize_func = IndexedRowTableLinearize()
    
    i = 0
    for raw, item in zip(raw_test_json, data_args):
        i += 1
        item['id'] = str(i)
        linearized_table = linearize_func.process_table(raw['table'])
    
        all_model_inputs.append(prompt.format(table = linearized_table,
                                              summary = item['response']))



elif (args.type == 'comprehensiveness') | (args.type == 'informativeness') | (args.type == 'naturalness'):
    import random
    model_name ='gpt-4-turbo-2024-04-09'
    print(f"{args.type}...{model_name}")
    print("instance_num_sampled:", len(data_args))


    list_length = len(data_args)

    random_indices = random.sample(range(list_length), 100)

    data_args = [data_args[i] for i in random_indices]
    comparing_data_args = [comparing_data_args[i] for i in random_indices]
    raw_test_json = [raw_test_json[i] for i in random_indices]

    linearize_func = IndexedRowTableLinearize()
    
    i = 0
    for raw, ours, comp in zip(raw_test_json, data_args, comparing_data_args):
        i += 1
        ours['id'] = str(i)
        ours['compare_to'] = args.comparing_data_args_path
        linearized_table = linearize_func.process_table(raw['table'])

        all_model_inputs.append(prompt.format(table = linearized_table,
                                              summary_A = ours['response'],
                                              summary_B = comp['response']))




# do not use
else:
    print("Something Went Wrong :( ")



print("len(all_model_inputs):",len(all_model_inputs))


collected_predictions = []

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

total_cost = 0
lock = asyncio.Lock()

async def async_generate(llm, model_input, i):
    global total_cost
    global collected_predictions
    system_message = SystemMessage(
        content="")
    while True:
        try:
            response = await llm.agenerate([[system_message, HumanMessage(content=model_input)]])

            prompt_tokens = response.llm_output['token_usage']['prompt_tokens']
            completion_tokens = response.llm_output['token_usage']['completion_tokens']
            if model_name == 'gpt-4-turbo-2024-04-09':
                prompt_tokens_cost = prompt_tokens / 1000 * 0.010   # gpt-4-turbo-2024-04-09
                completion_tokens_cost = completion_tokens / 1000 * 0.030
            # elif model_name == "gpt-3.5-turbo-0125":
            #     prompt_tokens_cost = prompt_tokens / 1000 * 0.0005    # turbo-0125
            #     completion_tokens_cost = completion_tokens / 1000 * 0.015

            print("completion_tokens:", completion_tokens)
            print("prompt_tokens_cost:", prompt_tokens_cost)
            print("completion_tokens_cost:", completion_tokens_cost)
            
            total_cost += (prompt_tokens_cost + completion_tokens_cost)

            print("total_cost:",total_cost)
            break
            
        except Exception as e:
            print(f"Exception occurred: {e}")
            response = None

    async with lock:

        cur_data = data_args[i]

        cur_data['prompt'] = model_input
        cur_data['eval'] = response.generations[0][0].text
        collected_predictions.append(cur_data)

        if len(collected_predictions) % 30 == 0:
            print(f"Expected Cost: {round(total_cost / len(collected_predictions) * len(data_args))}")



async def generate_concurrently(all_model_input, model_name):
    llm = ChatOpenAI(temperature=0, model_name=model_name)
    tasks = [async_generate(llm, model_input, i) for i, model_input in enumerate(all_model_input)]
    await tqdm_asyncio.gather(*tasks)


async def main():
    await generate_concurrently(all_model_inputs, model_name)


if __name__ == "__main__":

    asyncio.run(main())
    print("collected_predictions:",len(collected_predictions))
    
            
    if args.type == "faithfulness_eval":
        collected_predictions_sorted = dict_sort_by_id(collected_predictions)

        whole_true_cnt = 0
        whole_cnt = 0
        exception_cnt =0 
        for item in collected_predictions_sorted:
            string = item['eval']
            verification_results = re.findall(r'\(Verification\): (True|False)', string)
            
            cnt = 0
            for res in verification_results:
                if res == "True":
                    cnt += 1
                    whole_true_cnt += 1

            whole_cnt += len(verification_results)
            try:
                score = cnt / len(verification_results)
                item['score'] = score
            except:
                exception_cnt += 1
                continue
            

        item['whole_acc'] = whole_true_cnt / whole_cnt
        print("faithfulness acc:", whole_true_cnt / whole_cnt)
        print("parsing_error:", exception_cnt)


        with open(args.save_path, "w") as f:
            json.dump(collected_predictions_sorted, f, indent=4)


    elif args.type == "analytic_depth":
        print("[analytic_depth]")
        collected_predictions_sorted = dict_sort_by_id(collected_predictions)

        score = 0
        for item in collected_predictions_sorted:
            item['eval'] = extract_number(item['eval'])
            score += eval(extract_number(item['eval']))

        avg = score / len(collected_predictions_sorted)
        print("avg:", avg)
        collected_predictions_sorted.append({"avg_score": avg})

        with open(args.save_path, "w") as f:
            json.dump(collected_predictions_sorted, f, indent=4)

    elif (args.type == "comprehensiveness") | (args.type == 'informativeness') | (args.type == 'naturalness'):
        print("[comprehensiveness]")
        collected_predictions_sorted = dict_sort_by_id(collected_predictions)



        def parse_better_summary_index(text):
            match = re.search(r'Better Summary Index: \[?([A-B])\]?', text)
            if match:
                return match.group(1)
            return None
        
        count = 0
        for item in collected_predictions_sorted:
            item['result'] = parse_better_summary_index(item['eval'])
            if item['result'] == "A":
                count += 1

        collected_predictions_sorted.append({"win_percent": count / len(collected_predictions_sorted) * 100})

        with open(args.save_path, "w") as f:
            json.dump(collected_predictions_sorted, f, indent=4)




    else:
        collected_predictions_sorted = dict_sort_by_id(collected_predictions)
        with open(args.save_path, "w") as f:
            json.dump(collected_predictions_sorted, f, indent=4)