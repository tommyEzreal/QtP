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
parser.add_argument("--prompt", type=str)
parser.add_argument("--data_args_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--max_token", type=int, default=512)
parser.add_argument("--type", type=str)
args = parser.parse_args()

with open(args.data_args_path, "r") as f:
    data_args = json.load(f)
    data_args = data_args['data']

def dict_sort_by_id(item_list):
    sorted_list = sorted((item for item in item_list if item['id'].isdigit()), key=lambda x: int(x['id']))
    return sorted_list


# GPT input instance cache
all_model_inputs = []


if (args.type == 'zero_shot_summarization'):
    print(f"{args.type}...")
    model_name = 'gpt-3.5-turbo-0125'
    print("instance_num:", len(data_args))
    
    for i, item in enumerate(data_args):
        item['id'] = str(i+1)
        all_model_inputs.append(item['input'])
       

# do not use
else:
    print("Something Went Wrong :( ")



print("len(all_model_inputs):",len(all_model_inputs))


collected_predictions = []
os.environ["OPENAI_API_KEY"] = "YOUR API KEY"

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
            
            prompt_tokens_cost = prompt_tokens / 1000 * 0.001   
            completion_tokens_cost = completion_tokens / 1000 * 0.002

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
        cur_data['response'] = response.generations[0][0].text
        collected_predictions.append(cur_data)

        if len(collected_predictions) % 30 == 0:
            print(f"Expected Cost: {round(total_cost / len(collected_predictions) * len(data_args))}")



async def generate_concurrently(all_model_input, model_name):
    llm = ChatOpenAI(temperature=0, model_name=model_name,max_tokens=512)
    tasks = [async_generate(llm, model_input, i) for i, model_input in enumerate(all_model_input)]
    await tqdm_asyncio.gather(*tasks)


async def main():
    await generate_concurrently(all_model_inputs, model_name)


if __name__ == "__main__":

    asyncio.run(main())
    print("collected_predictions:",len(collected_predictions))

    if (args.type == "zero_shot_summarization"):
        collected_predictions_sorted = dict_sort_by_id(collected_predictions)
            
        with open(args.save_path, "w") as f:
            json.dump(collected_predictions_sorted, f, indent=4)


    else:
        print("something went wrong :(")