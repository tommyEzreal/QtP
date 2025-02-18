import argparse
from transformers import TapasForSequenceClassification, TapasTokenizer
import torch, json, tqdm, os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import load_dataset


class MyData(Dataset):
    '''
    Dataset for loading table-text data
    '''
    def __init__(self, raw_test, predictions, tokenizer):
        self.tokenizer = tokenizer           
        self.Data = self.load_data(raw_test, predictions)
        self.len = len(self.Data)

    def load_data(self, raw_test, predictions): 

        assert len(raw_test) == len(predictions), "size mismatch"

        new_data = []
        for example, prediction in zip(raw_test, predictions):
            header = example['table']['header']
            rows = example['table']['rows']
            new_data.append({ 
                "prediction" : prediction,
                "header": header,
                "rows": rows
            })

        return new_data


    def read_data(self, data : dict):
        '''
        the input is a sample stored as dict
        return: a pandas table and the statement
        '''
        sent = data['prediction']
        header = data['header']
        rows = data['rows']
        table = pd.DataFrame(rows, columns=header)
        table = table.astype(str)
        return table, sent

    def encode(self, table, sent):
        return self.tokenizer(table=table, queries=sent,
                truncation=True,
                padding='max_length',
                max_length=800)

    def __getitem__(self, index):
        table, sent = self.read_data(self.Data[index])
        try:
            d = self.encode(table, sent)
        except:
            print("max_len limit")
            d = self.encode(table, "None")
        for key, value in d.items():
            d[key] = torch.LongTensor(value)
        return d

    def __len__(self):
        return self.len




class TapasTest:

    def __init__(self, model_name):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = TapasForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model.to(self.device)

    def test(self, test_dataloader):
        num_correct = 0
        num_all = 0
        result = {}
        # for batch in tqdm.tqdm(test_dataloader):
        for batch in test_dataloader:
            # get the inputs
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)

            # forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            model_predictions = outputs.logits.argmax(-1)
            # print(torch.nn.functional.softmax(outputs.logits, dim=1))
            num_correct += model_predictions.sum()
            num_all += model_predictions.size(0)
            result['num_correct'] = int(num_correct)
            result['num_all'] = int(num_all)
            result['acc'] = float(num_correct / num_all)

        return result










