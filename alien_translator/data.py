import numpy as np

class Dataloader():
    '''
    Class for generate batches with tokens for src and dst texts
    '''
    def __init__(
            self, 
            src_texts,
            dst_texts, 
            src_tokenizer, 
            dst_tokenizer, 
            batch_size: int=64, 
            shuffle: bool=False
            ) -> None:
        '''
        Init method
        '''
        # Texts and check correctness
        self.src_texts = src_texts
        self.dst_texts = dst_texts
        assert len(self.dst_texts) == len(self.src_texts)
        self.data_length = len(self.dst_texts)
        assert self.data_length > 0

        # Batch size and check correctness
        self.batch_size = batch_size
        assert self.batch_size > 0
        assert isinstance(self.batch_size, int)

        self.shuffle = shuffle
        self.src_tokenizer = src_tokenizer
        self.dst_tokenizer = dst_tokenizer

    def __len__(self):
        return self.data_length // self.batch_size + (self.data_length % self.batch_size != 0)
        
    def __iter__(self):
        '''
        Iterator for generate batches
        '''
        # Make indeces and shuffle them
        indeces = np.arange(0, self.data_length)
        if self.shuffle:
            indeces = np.random.permutation(indeces)
        
        # Cycle fot yield batches
        for i in range(self.data_length // self.batch_size + (self.data_length % self.batch_size != 0)):
            # Choose texts for batch
            src_texts = [self.src_texts[j] for j in indeces[(i)*self.batch_size:(i+1)*self.batch_size]]
            dst_texts = ['[BOS]'+self.dst_texts[j]+'[EOS]' for j in indeces[(i)*self.batch_size:(i+1)*self.batch_size]]
            # Encode texts
            src_batch = self.src_tokenizer.encode(src_texts)
            dst_batch = self.dst_tokenizer.encode(dst_texts)
            
            yield {
                'src_input_ids': src_batch['input_ids'],
                'src_attention_mask': src_batch['attention_mask'],
                'dst_input_ids': dst_batch['input_ids'],
                'dst_attention_mask': dst_batch['attention_mask'],
            }