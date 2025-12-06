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

class TranslateTransformer(nn.Module):
    '''
    Transformer for machine translation
    '''
    def __init__(
                self, 
                src_vocab_size, 
                dst_vocab_size, 
                max_length,
                d_model=32, 
                nhead=2, 
                num_encoder_layers=3, 
                num_decoder_layers=2, 
                dim_feedforward=64,
                dropout=0.1,
                activation='relu',
                **kwargs,
                ):
        super(TranslateTransformer, self).__init__()
        self.src_vocab_size=src_vocab_size
        self.dst_vocab_size=dst_vocab_size
        self.max_length=max_length
        self.d_model=d_model
        self.nhead=nhead
        self.num_encoder_layers=num_encoder_layers
        self.num_decoder_layers=num_decoder_layers
        self.dim_feedforward=dim_feedforward
        self.dropout=dropout
        self.activation=activation

        self.__positional_encoding()

        self.src_emdedding = nn.Embedding(
            num_embeddings=self.src_vocab_size,
            embedding_dim=self.d_model,
        )

        self.dst_emdedding = nn.Embedding(
            num_embeddings=self.dst_vocab_size,
            embedding_dim=self.d_model,
        )

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )

        self.classifier = nn.Linear(self.d_model, self.dst_vocab_size)
    
    def forward(self, src_input_ids, dst_input_ids, src_attention_mask, dst_attention_mask):
        src_emb = self.src_emdedding(src_input_ids) * math.sqrt(self.d_model) + self.pos_embedding[:,:src_input_ids.shape[1],:].to(src_input_ids.device)
        dst_emb = self.dst_emdedding(dst_input_ids) * math.sqrt(self.d_model) + self.pos_embedding[:,:dst_input_ids.shape[1],:].to(dst_input_ids.device)

        out = self.transformer(
            src=src_emb,
            tgt=dst_emb,
            tgt_mask=self.transformer.generate_square_subsequent_mask(dst_emb.shape[1]).to(src_input_ids.device),
            src_key_padding_mask=~src_attention_mask.bool(),
            tgt_key_padding_mask=~dst_attention_mask.bool(), 
            memory_key_padding_mask=~src_attention_mask.bool()
        )

        scores = self.classifier(out)
        return scores

    def __positional_encoding(self):
        self.pos_embedding = torch.zeros(self.max_length, self.d_model)
        
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        self.pos_embedding[:, 0::2] = torch.sin(position * div_term)
        self.pos_embedding[:, 1::2] = torch.cos(position * div_term)
        
        self.pos_embedding = self.pos_embedding.unsqueeze(0)

class TransformerModule():
    '''
    Class for fitting model
    '''
    def __init__(self, config):
        '''
        Init method
        '''
        self.config = config
        self.model_parameters = self.config['model_params']
        self.train_parameters = self.config['training_params']
        self.device = self.config['env']['device']
        self.__init_model()

        ignore_index = self.train_parameters['ignore_index']
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        lr = float(self.train_parameters['lr'])
        weight_decay = float(self.train_parameters['weight_decay'])
        gamma = float(self.train_parameters['gamma'])
        self.optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        self.num_epochs = self.train_parameters['num_epochs']
        self.clip_grad = self.train_parameters['gradient_clip']

    def __init_model(self):
        '''
        Initialize model with params from config
        '''
        self._model = TranslateTransformer(
            self.config['src_tokenizer']['vocab_size'],
            self.config['dst_tokenizer']['vocab_size'],
            max(self.config['src_tokenizer']['max_length'], self.config['dst_tokenizer']['max_length']),
            **self.model_parameters
        ).to(self.device)
    
    def fit(self, train_dataloader, val_dataloader):
        '''
        Fit model
        '''
        best_loss = np.inf
        for self.epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.val_epoch(val_dataloader)
            self.scheduler.step()
            print(f"{self.epoch+1}/{self.num_epochs}: Train loss = {train_loss:.4f}. Val loss = {val_loss:.4f}, lr = {self.scheduler.get_last_lr()[0]:.6f}")

            # Save best model
            if(val_loss < best_loss):
                best_loss = val_loss

                directory = self.config['paths']['checkpoints']
                os.makedirs(directory, exist_ok=True)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'model_state_dict': self._model.state_dict(),
                        'loss': best_loss,
                        'model_params': self.model_parameters,
                        'training_params': self.train_parameters,
                    }, 
                    os.path.join(directory, f'transformer.pt')
                    )
    
    def train_epoch(self, train_dataloader):
        train_loss_list = []
        self._model.train()

        pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {self.epoch+1}/{self.num_epochs}', postfix={'loss': '?'}) 
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            out = self._model(
                batch['src_input_ids'].to(self.device), 
                batch['dst_input_ids'][:,:-1].to(self.device), 
                batch['src_attention_mask'].to(self.device), 
                batch['dst_attention_mask'][:,:-1].to(self.device)
            )
            loss = self.criterion(out.reshape(-1, out.size(-1)), batch['dst_input_ids'][:,1:].reshape(-1).to(self.device))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clip_grad)

            train_loss_list.append(loss.item())
            self.optimizer.step()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            pbar.update(1)
        pbar.close()

        return np.mean(train_loss_list)
    
    def val_epoch(self, val_dataloader):
        val_loss_list = []
        self._model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                self.optimizer.zero_grad()
                out = self._model(
                    batch['src_input_ids'].to(self.device), 
                    batch['dst_input_ids'][:,:-1].to(self.device), 
                    batch['src_attention_mask'].to(self.device), 
                    batch['dst_attention_mask'][:,:-1].to(self.device)
                )
                loss = self.criterion(out.reshape(-1, out.size(-1)), batch['dst_input_ids'][:,1:].reshape(-1).to(self.device))
                val_loss_list.append(loss.item())
        return np.mean(val_loss_list)