import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from typing import List
import math

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

class TranslateTokenizer():
    '''
    Class for tokenizer and logic for this
    '''
    def __init__(self, max_length):
        self.max_length = max_length
        self.is_fitted = False

    def load(self, path):
        if os.path.exists(path):
            self.engine = Tokenizer.from_file(path)

            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=self.engine,
                unk_token="[UNK]",
                pad_token="[PAD]"
            )
            self.vocab_size = self.engine.get_vocab_size()
            self.is_fitted = True
        else:
            raise Exception("There is not saved tokenizer, fit it")
        
    def fit(
            self, 
            texts, 
            path, 
            vocab_size: int=5000, 
            min_frequency: int=2, 
            special_tokens: List=["[PAD]", "[BOS]", "[EOS]"]
            ) -> None:
        self.engine = Tokenizer(models.BPE())
        self.engine.pre_tokenizer = pre_tokenizers.Whitespace()

        tokenizer_trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens
        )

        self.engine.train_from_iterator(texts, trainer=tokenizer_trainer)
        self.vocab_size = self.engine.get_vocab_size()

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.engine.save(path)
        self.is_fitted = True

        self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=self.engine,
                unk_token="[UNK]",
                pad_token="[PAD]"
            )

    def encode(self, texts):
        '''
        Encode list of texts into batch of tokens
        '''
        assert self.is_fitted, 'Tokenizer need to be fited'
        out = self.tokenizer(
                texts, 
                padding=True, 
                return_tensors='pt', 
                max_length=self.max_length, 
                truncation=True
            )
        return out

    def decode(self, tokens):
        '''
        Decode batch of tokens into list of texts
        '''
        assert self.is_fitted, 'Tokenizer need to be fited'
        out = self.tokenizer.batch_decode(tokens)
        return out