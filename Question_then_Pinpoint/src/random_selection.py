import random
import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--step_one_output_path", type=str)
parser.add_argument("--num_sample", type=int)
parser.add_argument("--save_path", type=str)

args = parser.parse_args()

with open(args.step_one_output_path, "r") as f:
    step_one_output = json.load(f)


def random_selection(step_one_output,num_sample,save_path):
    exception_count = 0

    for item in step_one_output:
        try:
            item['response'] = random.sample(item['response'], num_sample)
        except:
            print("exception: smaller than the sample size")
            exception_count += 1
            pass
    
    print("exception_count:", exception_count)
    with open(save_path, "w") as f:
        json.dump(step_one_output, f, indent=4)

if __name__ == "__main__":
    random_selection(step_one_output, args.num_sample, args.save_path)
