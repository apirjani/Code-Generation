import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch
import pandas as pd

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        # additional_tokens = ['SELECT', 'DISTINCT', 'FROM', 'WHERE', 'NOT', 'AND', 'IN', 'BETWEEN', 'IS NOT NULL', 'IS NULL', 'JOIN', 'GROUP BY', 'ORDER BY', '=', ',', '*', '(', ')', '<', '>', '<=', '>=', 'count']
        # self.tokenizer.add_tokens(additional_tokens)
        self.start_token_id = self.tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
        self.data = self.load_data(data_folder, split)

    def load_data(self, data_folder, split):
        task_prefix = "Translate English to SQL: "
        nl_path = f"{data_folder}/{split}.nl"        

        with open(nl_path, 'r') as f:
            nl_lines = [f"{task_prefix}{line.strip()}" for line in f.readlines()]

        if split == "test":
            data = [(self.tokenizer(line, truncation=True, max_length=256, return_tensors="pt").input_ids.squeeze(0),
                     self.tokenizer(line, truncation=True, max_length=256, return_tensors="pt").attention_mask.squeeze(0),
                     torch.tensor([self.start_token_id], dtype=torch.long)) for line in nl_lines]
            return data

        sql_path = f"{data_folder}/{split}.sql"
        with open(sql_path, 'r') as f:
            sql_lines = [line.strip() for line in f.readlines()]

        data = []
        for nl_line, sql_line in zip(nl_lines, sql_lines):
            input_encoding = self.tokenizer(nl_line, truncation=True, max_length=256, return_tensors="pt")
            encoder_ids = input_encoding.input_ids.squeeze(0)
            encoder_mask = input_encoding.attention_mask.squeeze(0)
            target_encoding = self.tokenizer(sql_line, truncation=True, max_length=256, return_tensors="pt")
            decoder_target_ids = target_encoding.input_ids.squeeze(0)
            # Create decoder_input_ids by excluding the last token from the target inputs
            decoder_input_ids = torch.cat([
                torch.full((target_encoding.input_ids.size(0), 1), self.start_token_id, dtype=torch.long),
                target_encoding.input_ids[:, :-1]
            ], dim=1).squeeze(0)
            initial_decoder_inputs = torch.tensor([self.start_token_id], dtype=torch.long)
            data.append((encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_inputs))

        return data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    
    encoder_ids, encoder_mask, decoder_input_ids, target_ids, initial_decoder_inputs = zip(*batch)
    encoder_ids = pad_sequence(list(encoder_ids), batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(list(encoder_mask), batch_first=True, padding_value=PAD_IDX)
    decoder_input_ids = pad_sequence(list(decoder_input_ids), batch_first=True, padding_value=PAD_IDX)
    target_ids = pad_sequence(list(target_ids), batch_first=True, padding_value=-100)

    initial_decoder_inputs = torch.stack(initial_decoder_inputs)

    return encoder_ids, encoder_mask, decoder_input_ids, target_ids, initial_decoder_inputs
    

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids, encoder_mask, initial_decoder_inputs = zip(*batch)
    encoder_ids = pad_sequence(list(encoder_ids), batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(list(encoder_mask), batch_first=True, padding_value=PAD_IDX)

    initial_decoder_inputs = torch.stack(initial_decoder_inputs)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    # tokenizer = dset.tokenizer
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_nl_path = f"{data_folder}/train.nl"
    train_sql_path = f"{data_folder}/train.sql"
    dev_nl_path = f"{data_folder}/dev.nl"
    dev_sql_path = f"{data_folder}/dev.sql"
    test_nl_path = f"{data_folder}/test.nl"


    train_x = load_lines(train_nl_path)
    train_y = load_lines(train_sql_path)
    dev_x = load_lines(dev_nl_path)
    dev_y = load_lines(dev_sql_path)
    test_x = load_lines(test_nl_path)

    return train_x, train_y, dev_x, dev_y, test_x