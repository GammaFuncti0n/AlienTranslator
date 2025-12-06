import yaml
import json
import os
from typing import Dict
import numpy as np
import torch
import torch.nn as nn

from alien_translator.model import TranslateTokenizer, TransformerModule
from alien_translator.data import Dataloader

# from clearml import Task, Logger
# from configs.config import clearml_config

# clearml_config.task = Task.init(
#     project_name='AlienTranslator', 
#     task_name='transformet_training', 
#     tags=['transformer','machine_translation'])

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    module = TranslatorModule(config)
    module.prepare_data()
    module.init_tokenizers()

    train_dataloader, val_dataloader = module.init_dataloaders()

    model = TransformerModule(config)
    model.fit(train_dataloader, val_dataloader)

    ## TO DO ##
    # evaluate model
    # save test

class TranslatorModule():
    def __init__(self, config: Dict) -> None:
        '''
        init method
        '''
        self.config = config
    
    def prepare_data(self):
        '''
        Method for load data and save in lists
        '''
        train_data = self._load_data(self.config['paths']['train_data_dir'])
        val_data = self._load_data(self.config['paths']['val_data_dir'])
        test_data = self._load_data(self.config['paths']['test_data_dir'])

        self.src_train_texts = [line['src'] for line in train_data]
        self.dst_train_texts = [line['dst'] for line in train_data]
        
        self.src_val_texts = [line['src'] for line in val_data]
        self.dst_val_texts = [line['dst'] for line in val_data]
    
    def _load_data(self, path):
        """
        Load dataset from path
        """
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        return data
    
    def init_tokenizers(self):
        '''
        Initialize tokenizer, load it or fit
        '''
        self.src_tokenizer = TranslateTokenizer(self.config['src_tokenizer']['max_length'])
        self.dst_tokenizer = TranslateTokenizer(self.config['dst_tokenizer']['max_length'])
        src_tokenizer_path = os.path.join(self.config['paths']['checkpoints'], self.config['src_tokenizer']['file_name'])
        dst_tokenizer_path = os.path.join(self.config['paths']['checkpoints'], self.config['dst_tokenizer']['file_name'])
        try:
            self.src_tokenizer.load(src_tokenizer_path)
            self.dst_tokenizer.load(dst_tokenizer_path)
        except:
            self.src_tokenizer.fit(
                self.src_train_texts, 
                src_tokenizer_path,
                vocab_size=self.config['src_tokenizer']['vocab_size'],
                min_frequency=self.config['src_tokenizer']['min_frequency'],
                special_tokens=self.config['src_tokenizer']['special_tokens']
                )
            self.dst_tokenizer.fit(
                self.dst_train_texts, 
                dst_tokenizer_path,
                vocab_size=self.config['dst_tokenizer']['vocab_size'],
                min_frequency=self.config['dst_tokenizer']['min_frequency'],
                special_tokens=self.config['dst_tokenizer']['special_tokens']
                )
    
    def init_dataloaders(self):
        '''
        Initialize train and val dataloaders
        '''
        train_dataloader = Dataloader(
            self.src_train_texts, 
            self.dst_train_texts, 
            self.src_tokenizer, 
            self.dst_tokenizer, 
            batch_size=self.config['training_params']['batch_size'], 
            shuffle=True
            )
        val_dataloader = Dataloader(
            self.src_val_texts, 
            self.dst_val_texts, 
            self.src_tokenizer, 
            self.dst_tokenizer, 
            batch_size=self.config['training_params']['batch_size'], 
            shuffle=False
            )
        return train_dataloader, val_dataloader

if __name__ == "__main__":
    main()