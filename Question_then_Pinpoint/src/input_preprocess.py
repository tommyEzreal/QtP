from typing import Dict, List, List
import json
import argparse
import re
from table_linearize import IndexedRowTableLinearize

parser = argparse.ArgumentParser()
parser.add_argument("--data_args_path", type=str, required=True)
parser.add_argument("--step_one_output_path", type=str)
parser.add_argument("--prompt_path", type=str)
parser.add_argument("--task_type", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
args = parser.parse_args()


def format_prompt(datapoint, prompt, task_type):
    
    if (task_type == "Aspect_focused_QG") :
        model_input = prompt.format(table = datapoint['linearized_table'])  
    
    elif (task_type == "SG_w_gen_I"):
        model_input = prompt.format(table = datapoint['linearized_table'],
                                    insights = datapoint['gen_insights'])
    
    return model_input


def process_step_one_output(step_one_output_path, data_args):
    
    with open(step_one_output_path, "r") as f:
        step_one_output = json.load(f)
    
    assert len(step_one_output) == len(data_args)

    for one, raw in zip(step_one_output, data_args):
        step_one_response = one['response']
        
        # summary inference with gen insight 
        if (args.task_type == "SG_w_gen_I"):
            insight_string = ""
            for res in step_one_response:
                insight_string += res
                insight_string += ". "

            raw['gen_insights'] = insight_string
        
    
    print("inject STEP1 output to summarize module prompt..")

    return data_args



def verbalization(response):
    output_string = ""

    for pair in response:
        aspect = pair['aspect']
        output_string += "(Coarse-level Aspect): "
        output_string += aspect
        output_string += "\n"
        triples = pair['triples']
        
        k = 0
        for key in list(triples.keys()):
            k += 1
            question = triples[key]['question']
    
            output_string += f"(Fine-level Question {str(k)}): "
            output_string += question
        
            output_string += "\n"
        output_string += "\n"

    return output_string


 
def preprocess(raw_train):
    linearize_func = IndexedRowTableLinearize()
    data_dict = {'data' : []}
    for j, item in enumerate(raw_train):
        
        item['linearized_table'] = linearize_func.process_table(item['table'])
        

        # formulate input-output
        instance = {} 
        instance['input'] = format_prompt(item, 
                                        prompt, 
                                        task_type = args.task_type)
        
        if args.task_type == "Aspect_focused_QG":
            try:  # train data
                output_string = verbalization(item['implicit_knowledge'])  # aspect & questions & relevants 
                instance['output'] = output_string

            except: # test data
                instance['output'] = ""
        
        # Summary Generation
        elif args.task_type == "SG_w_gen_I":
            instance['output'] = item['reference_summary']
        
        data_dict['data'].append(instance)
  
    # save
    with open(args.save_path, "w") as f:
        json.dump(data_dict, f, indent=4)

 
 
def IG_task_train_preprocess(raw_train): 
    linearize_func = IndexedRowTableLinearize()

    data_dict = {'data' : []}
    for k, item in enumerate(raw_train):
        
        item['linearized_table'] = linearize_func.process_table(item['table'])

        for rationale in item['implicit_knowledge']:
            
            triples = rationale['triples']

            for key in list(triples.keys()):
                instance = {}
                instance['id'] = str(k+1)

                
                question = triples[key]['question']
                if triples[key]['cell_evidence'] == None:
                    cell_evidence = "N/A"
                else:
                    cell_evidence = triples[key]['cell_evidence']
                insight = triples[key]['insight']

                single_instance = prompt.format(table = item['linearized_table'],
                                                question = question)
                
                instance['input'] = single_instance
                
                # output verbalization
                output_text = ""
                output_text += f"The relevant columns and rows for the Question is {cell_evidence}. Therefore, the answer is \n"
                output_text += f"(Answer): {insight}"
                instance['output'] = output_text
                
                data_dict['data'].append(instance)
        

    # save
    with open(args.save_path, "w") as f:
        json.dump(data_dict, f, indent=4)


def question_response_parsing(input_string):
    # Split the input string into segments based on the coarse-level aspect
    segments = input_string.strip().split("(Coarse-level Aspect):")

    # Remove empty strings
    segments = [segment.strip() for segment in segments if segment.strip()]

    # Create a dictionary to store the parsed data
    parsed_data = []

    # Iterate through segments to parse coarse-level aspects and fine-level questions
    for segment in segments:
        new_dic = {}
        lines = segment.split("\n")
        aspect = lines[0].strip()

        questions = []
        
        Q_Rel_pairs = lines[1:]
       
        for Q_Rel in Q_Rel_pairs:
            Q_index = Q_Rel.find( "(Fine-level Question" )
            
            
            Q = Q_Rel[Q_index + len("(Fine-level Question") + 4: ].strip()
            questions.append(Q)
    
        new_dic['aspect'] = aspect
        new_dic['questions'] = questions
        parsed_data.append(new_dic)


    return parsed_data


def Evidence_focused_IG_inference_preprocess(step_one_output, raw_test):

    assert len(step_one_output) == len(raw_test), "output - raw mismtach"

    linearize_func = IndexedRowTableLinearize()

    data_dict = {'data' : []}
    
    for item, raw in zip(step_one_output, raw_test):
        
        dump_output = item['response']

        
        parsed_response = question_response_parsing(dump_output)
        linearized_table = linearize_func.process_table(raw['table'])
        

        for pair in parsed_response: 
            
            for question in pair['questions']:

                post_processed_dic = {}
                post_processed_dic['id'] = raw['id']
                post_processed_dic['table'] = raw['table']
                post_processed_dic['reference_summary'] = raw['reference_summary']
                post_processed_dic['parsed_response'] = parsed_response

                single_instance = prompt.format(table = linearized_table,
                                                question = question)
                single_output = ""
                post_processed_dic['input'] = single_instance
                post_processed_dic['output'] = single_output
                data_dict['data'].append(post_processed_dic)

    # save
    with open(args.save_path, "w") as f:
        json.dump(data_dict, f, indent=4)


# load
with open(args.data_args_path, "r") as f:
    data_args = json.load(f)

with open(args.prompt_path, "r") as f:
    prompt = f.read()

if (args.task_type == "Evidence_focused_IG_inference"):
    with open(args.step_one_output_path, "r") as f:
        step_one_output = json.load(f)


if __name__ == "__main__":

    if (args.task_type == "SG_w_gen_I"):
        print("bring output..")
        data_args = process_step_one_output(args.step_one_output_path, data_args)
        
    else:
        pass
    
    if (args.task_type == "Evidence_focused_IG"):
        IG_task_train_preprocess(data_args)

    elif (args.task_type == "Evidence_focused_IG_inference"):
        Evidence_focused_IG_inference_preprocess(step_one_output, data_args)

    else: # Aspect-focused QG
        preprocess(data_args)
    


