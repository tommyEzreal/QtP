import json
import argparse
import os
import evaluate
from autoacu import A3CU
from typing import List
from nltk import word_tokenize
from tapas_acc import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--raw_test_json_path", type = str, required=True)
parser.add_argument("--model_output_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--experiment_name", type=str)



import wandb
wandb.init(project="EVALUATION")

args = parser.parse_args()

def get_bleu_1234_scores(predictions, references):
    bleu = evaluate.load("bleu")
    all_results = []
    for i in range(1,5):
        results = bleu.compute(predictions = predictions, references = references, max_order=i)
        results[f'bleu{i}'] = round(results['bleu'], 4)
        all_results.append({f"bleu{i}" : results[f'bleu{i}']})        
    return all_results

def get_sacrebleu_scores(predictions, references):
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(predictions=predictions, references=[[r] for r in references])
    return results["score"]

def get_rougel_scores(predictions, references):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return results["rougeL"]

def get_meteor_scores(predictions, references):
    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=predictions, references=references)
    return results["meteor"]

def get_bert_scores(predictions, references):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    # print(len(results["f1"]))
    return sum(results["f1"]) / len(results["f1"])

def get_tapas_scores(raw_test_json, predictions):
    tapas = TapasTest("google/tapas-large-finetuned-tabfact")
    
    data = MyData(raw_test_json, predictions, tapas.tokenizer)
    test_dataloader = DataLoader(data, batch_size=8, shuffle=False, num_workers=4)
    results = tapas.test(test_dataloader)

    
    return results["acc"], results['num_correct'], results['num_all']

    
    
def get_autoacu_scores(predictions, references):
    a3cu = A3CU(device=0)
    recall_scores, prec_scores, f1_scores = a3cu.score(
        references=references,
        candidates=predictions,
        batch_size=32,
        output_path=None,
    )
    return sum(f1_scores) / len(f1_scores)

def get_prediction_lengths(predictions):
    total_length = 0
    for prediction in predictions:
        total_length += len(word_tokenize(prediction))
    return total_length / len(predictions)
    
    
def get_all_scores(raw_test_json, predictions, references, experiment_name = args.model_output_path, verbose = True):
    all_scores = {}
    all_scores['experiment_name'] = f"{experiment_name}"
    
    # all_scores['BLEU1234'] = get_bleu_1234_scores(predictions, references)
    # if verbose==True:
    #     for i in range(1,5):
    #         print(f"BLEU{i}:" ,all_scores['BLEU1234'][i-1])

    all_scores["sacreBLEU"] = get_sacrebleu_scores(predictions, references)
    print("sacreBLEU score: ", all_scores["sacreBLEU"])
    
    all_scores["Rouge-L"] = get_rougel_scores(predictions, references)
    if verbose==True:
        print("Rouge-L score: ", all_scores["Rouge-L"])
    all_scores["METEOR"] = get_meteor_scores(predictions, references)
    if verbose==True:
        print("METEOR score: ", all_scores["METEOR"])
    all_scores["BERTScore"] = get_bert_scores(predictions, references)
    if verbose==True:
        print("BERTScore score: ", all_scores["BERTScore"])
    
    tapas_score, num_correct, num_all = get_tapas_scores(raw_test_json, predictions)
    all_scores["TAPAS"] = (tapas_score, num_correct, num_all) 
    if verbose==True:
        print("TAPAS score: ", tapas_score)
        print("num_correct:", num_correct)
        print("num_all:", num_all)

    
    all_scores["AutoACU"] = get_autoacu_scores(predictions, references)
    if verbose==True:
        print("AutoACU score: ", all_scores["AutoACU"])
    all_scores["Prediction Length"] = get_prediction_lengths(predictions)
    if verbose==True:
        print("Prediction Length: ", all_scores["Prediction Length"])
    return all_scores


def evaluate_prediction(raw_test_json_path, prediction_json_path):

    with open(prediction_json_path, "r") as f:
        output_json = json.load(f)
    with open(raw_test_json_path, 'r') as f:
            raw_test = json.load(f)
    
    predictions, references = [], []
    for item, raw in zip(output_json, raw_test):
        # reference = item['output']
        reference = raw['reference_summary']
        prediction = item['response']
        references.append(reference)
        predictions.append(prediction)

    assert len(predictions) == len(references), "pred-refer size mismatch"

    all_scores = get_all_scores(raw_test, predictions, references)
    
    return all_scores

def evaluate_per_instance(raw_test_json_path, prediction_json_path):
    with open(prediction_json_path, "r") as f:
        output_json = json.load(f)
    with open(raw_test_json_path, 'r') as f:
            raw_test = json.load(f)

    for item, raw in zip(tqdm(output_json), raw_test):
        # reference = [item['output']]
        reference = [item['reference_summary']]
        prediction = [item['response']]
        raw = [raw]
        
        instance_score = get_all_scores(raw, prediction, reference, verbose=False)
        item['score'] = instance_score

    return output_json


if __name__ == "__main__":

    all_scores = evaluate_prediction(args.raw_test_json_path, args.model_output_path)
    # model_output = evaluate_per_instance(args.raw_test_json_path, args.model_output_path)
    
    with open(args.save_path, "w") as f:
        json.dump(all_scores, f, indent=4)

    # with open(args.model_output_path, "w") as f:
    #     json.dump(model_output, f, indent=4)

    print("Finished evaluating {}.".format(args.model_output_path))
    print(all_scores)