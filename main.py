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
    
    # Init some stuff
    module = TranslatorModule(config)
    module.prepare_data()
    module.init_tokenizers()

    train_dataloader, val_dataloader = module.init_dataloaders()

    # Load or fit the model
    model = TransformerModule(config)
    try:
        last_checkpoint = torch.load(os.path.join(config['paths']['checkpoints'], 'transformer.pt'), weights_only=False)
        model._model.load_state_dict(last_checkpoint['model_state_dict'])
    except:
        model.fit(train_dataloader, val_dataloader)

    # evaluate on test data
    test_dst = translate(
        model._model, 
        module.src_tokenizer, 
        module.dst_tokenizer, 
        module.src_test_texts, 
        config['training_params']['batch_size'], 
        config['env']['device'], 
        config['dst_tokenizer']['max_length']
        )

    # save submission
    with open('data/submission', 'w', encoding='utf-8') as f:
        for i in range(len(test_dst)):
            item = {'src': module.src_test_texts[i], 'dst': test_dst[i]}
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

class TranslatorModule():
    def __init__(self, config: Dict) -> None:
        '''
        init method
        '''
        self.config = config
        self.device = config['env']['device']
        self.max_length = config['dst_tokenizer']['max_length']
    
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

        self.src_test_texts = [line['src'] for line in test_data]
    
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

def translate(model, src_tokenizer, dst_tokenizer, test_src, batch_size, device='cpu', max_length=100):
    test_dst = []
    for i in range(len(test_src)//batch_size+int(len(test_src)%batch_size!=0)):
        texts = test_src[i*batch_size:(i+1)*batch_size]

        src_input = src_tokenizer.encode(texts)
        src_input_ids = src_input['input_ids'].to(device)
        src_attention_mask=src_input['attention_mask'].to(device)

        dst_input = dst_tokenizer.encode(len(src_input_ids)*['[BOS]'])['input_ids'].to(device)

        for i in range(max_length):
            with torch.no_grad():
                out = model(
                    src_input_ids=src_input_ids, 
                    dst_input_ids=dst_input, 
                    src_attention_mask=src_attention_mask,
                    dst_attention_mask=(dst_input!=0)
                )
            out_tokens = out[:,-1,:].argmax(-1).reshape(-1,1)
            dst_input = torch.concat((dst_input, out_tokens), -1)
        dst_output = dst_tokenizer.decode(dst_input)
        final_res = []
        for s in dst_output:
            final_s = ''
            for i in range(6, len(s)):
                if(s[i:i+5]=='[EOS]'):
                    break
                else:
                    final_s += s[i]
            final_res.append(final_s)
        test_dst = test_dst + final_res
    return test_dst

if __name__ == "__main__":
    main()