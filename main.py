import yaml
import json

# from clearml import Task, Logger
# from configs.config import clearml_config

# clearml_config.task = Task.init(
#     project_name='AlienTranslator', 
#     task_name='transformet_training', 
#     tags=['transformer','machine_translation'])

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    ## TO DO ##

    # load data
    src_train_texts, dst_train_texts = load_data(config['paths']['train_data_dir'])
    src_val_texts, dst_val_texts = load_data(config['paths']['val_data_dir'])
    
    # prepare tokenizer
    # fit model
    # evaluate model
    # save test

def load_data(path: str):
    '''
    Function for load data and split on src/dst
    '''
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    src_data = [line['src'] for line in data]
    dst_data = [line['dst'] for line in data]

    return src_data, dst_data

if __name__ == "__main__":
    main()