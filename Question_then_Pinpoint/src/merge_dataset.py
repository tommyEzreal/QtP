import random
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--QG_data_path", type=str, required=True)
parser.add_argument("--QA_data_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
args = parser.parse_args()


with open(args.QG_data_path, "r") as f:
    QG_data = json.load(f)['data']

with open(args.QA_data_path, "r") as f:
    QA_data = json.load(f)['data']


print(QG_data[100]['input'])
print('---------------------')
print(QG_data[100]['output'])
print('---------------------')
print('---------------------')
print('---------------------')
print(QA_data[100]['input'])
print('---------------------')
print(QA_data[100]['output'])


merged_list = QG_data + QA_data
random.shuffle(merged_list)

merged_data= {}
merged_data['data'] = merged_list

with open(args.save_path, "w") as f:
    json.dump(merged_data, f, indent=4)