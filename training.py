import tensorflow as tf
import pandas as pd
import torch
from datasets import load_dataset
import os
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import argparse

def preprocess(sentence):
    #s = sentence["text"].lower()
    s = sentence["text"]
    # removes info within [], such as dates and redacted info (names, page numbers, etc)
    ret = ''
    skip1c = 0
    skip2c = 0
    for i in s:
        if i == '[':
            skip1c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i

    # remove _______
    ret = ret.replace("_", "")

    # remove large gaps and newline charcters
    ret = " ".join(ret.split())

    sentence["text"] = ret.lower()

    return sentence

def make_dataset(file_loc = "TBI_notes_unsorted.csv", time_max = 24, comb_strat = 'random_sent', len = 512, strip_num = False):
    """
    Accepts 
    file_loc: the location of the csv file with the notes
    time_max: the maximum time in hours to include in the dataset
    comb_strat: the strategy to combine notes for a patient
        includes: 'concat', 'concat_reverse', 'random_note', 'random_sent', 'note_equal'
    """
    device = torch.device(args.device)
    df = pd.read_csv(file_loc)
    # set index to HADM_ID
    df = df.set_index('HADM_ID')
    # set DIFF to timedelta
    df['DIFF'] = pd.to_timedelta(df['DIFF'])
    # drop notes with diff < 0 
    df = df[df['DIFF'] > pd.Timedelta('0 days')]
    # drop notes with diff > time_max
    df = df[df['DIFF'] < pd.Timedelta(f'{time_max} hours')]
    # strip text of numbers
    if strip_num:
        num_text = set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion"])
        count_text = set(["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth", "hundredth", "thousandth", "millionth", "billionth", "trillionth"])
        # efficiently strip numbers, num_text, and count_text
        df['TEXT'] = df['TEXT'].str.replace(r'\d+', '')
        # remove num_text and count_text words
        df['TEXT'] = df['TEXT'].str.replace(r'\b(' + '|'.join(num_text) + r')\b', '')
        df['TEXT'] = df['TEXT'].str.replace(r'\b(' + '|'.join(count_text) + r')\b', '')
        
    # combine notes for each patient by comb_strat
    # create merger strategies, EXPIER_FLAG=first, and LOS=first, dropping DIFF
    merger = {'EXPIRE_FLAG': 'first', 'LOS': 'first', 'DIFF': 'first'}
    if comb_strat == 'concat':
        df = df.sort_values(by=['DIFF'])
        df = df.groupby('HADM_ID').agg({'TEXT': ' '.join, **merger})

    if comb_strat == 'concat_split':
        split_len = 4016*5
        df = df.sort_values(by=['DIFF'])
        df = df.groupby('HADM_ID').agg({'TEXT': ' '.join, **merger})
        df['TEXT'] = df['TEXT'].str.split(' ', split_len)
        df = df.explode('TEXT')
    elif comb_strat == 'concat_reverse':
        df = df.sort_values(by=['DIFF'], ascending=False)
        df = df.groupby('HADM_ID').agg({'TEXT': ' '.join, **merger})

    elif comb_strat == 'random_note':
        df = df.sort_values(by=['DIFF'])
        df = df.groupby('HADM_ID').agg({'TEXT': lambda x: x.sample(n=1, random_state=1).iloc[0], **merger})
    
    # elif comb_strat == 'random_sent':
    #     df = df.sort_values(by=['DIFF'])
    #     # concat all notes for each patient
    #     df = df.groupby('HADM_ID').agg({'TEXT': ' '.join, **merger})
    #     # split into sentences
    #     from nltk.tokenize import sent_tokenize
    #     # apply sent tokenize to each TEXT
    #     df['TEXT'] = df['TEXT'].apply(lambda x: sent_tokenize(x))
    #     # explode TEXT to sentences
    #     df = df.explode('TEXT')
    #     # drop rows with text length < 10
    #     df = df.loc[df['TEXT'].str.len() > 10]
    #     # randomly shuffle the dataframe order 
    #     df = df.sample(frac=1, random_state=1)
    #      # join sentences by patient id 
    #     df = df.groupby('HADM_ID').agg({'TEXT': ' '.join, **merger})

    elif comb_strat == 'rand_sent_neighbors':
        # use the random sent strategy, but include the sentences before and after the randomly selected sentence
        df = df.sort_values(by=['DIFF'])
        # concat all notes for each patient
        df = df.groupby('HADM_ID').agg({'TEXT': ' '.join, **merger})
        # split into sentences
        from nltk.tokenize import sent_tokenize
        # apply sent tokenize to each TEXT
        df['TEXT'] = df['TEXT'].apply(lambda x: sent_tokenize(x))
        # concat sentences into sets of 3
        print(df['TEXT'].iloc[0])
        df['TEXT'] = df['TEXT'].apply(lambda x: [x[i:i+3] for i in range(0, len(x), 3)])
        # explode TEXT to sentences
        df = df.explode('TEXT')
        # drop rows with text length < 10
        df = df.loc[df['TEXT'].str.len() > 10]
        # randomly shuffle the dataframe order
        df = df.sample(frac=1, random_state=1)
        # join sentences by patient id
        df = df.groupby('HADM_ID').agg({'TEXT': ' '.join, **merger})

    elif comb_strat == 'note_500': 
        # use the first 500 characers of each note
        df = df.sort_values(by=['DIFF'])
        df = df.groupby('HADM_ID').agg({'TEXT': lambda x: x.str[:500], **merger})
        df = df.explode('TEXT')
        df = df.groupby('HADM_ID').agg({'TEXT': ' '.join, **merger})

    elif comb_strat == 'rand_note_shortened':
        df = df.sort_values(by=['DIFF'])
        df = df.groupby('HADM_ID').agg({'TEXT': lambda x: x.sample(n=1, random_state=1).iloc[0], **merger})
        # split into sentences
        from nltk.tokenize import sent_tokenize
        # apply sent tokenize to each TEXT
        df['TEXT'] = df['TEXT'].apply(lambda x: sent_tokenize(x))
        # explode TEXT to sentences
        df = df.explode('TEXT')
        # drop rows with text length < 10
        df = df.loc[df['TEXT'].str.len() > 10]
        
        # take out words that are common to 90%+ of the notes
        # compute the most common words, splitting notes by space
        df_copy = df.copy()
        df_copy['note_id'] = range(df_copy.shape[0])
        # create a dataframe with word from sentences with index of note_id
        word_df = df_copy['TEXT'].str.split(expand=True).stack().reset_index(level=1, drop=True).to_frame('word').join(df_copy[['note_id']], how='left')
        # aggragate the word counts by note_id into a set
        word_df = word_df.groupby('note_id').agg({'word': set})
        # get the words that are common to 90% of the notes
        common_words = {}
        for word_set in word_df['word']:
            for word in word_set:
                if word in common_words:
                    common_words[word] += 1
                else:
                    common_words[word] = 1
        # remove the 100 most common words
        # find the 100 most common words
        min_count = sorted(common_words.values())[-100]
        common_words = set([word for word, count in common_words.items() if count > min_count])
        print(common_words)
        
        # remove the common words
        df['TEXT'] = df['TEXT'].str.replace(r'\b(' + '|'.join(common_words) + r')\b', '')
        # shorten all rows to 100 characters
        df['TEXT'] = df['TEXT'].str[:100]
        # randomly shuffle the dataframe order 
        df = df.sample(frac=1, random_state=1)
         # join sentences by patient id 
        df = df.groupby('HADM_ID').agg({'TEXT': ' '.join, **merger})
        
    elif comb_strat == 'note_equal':
        # take an equal number of notes for each patient based on the total length = len
        df = df.sort_values(by=['DIFF'])
        # create a column with number of notes for each patient
        df['NUM_NOTES'] = df.groupby('HADM_ID')['TEXT'].transform('count')
        # shorten each note to len/NUM_NOTES
        df['TEXT'] = df['TEXT'].apply(lambda x: x[:len//df['NUM_NOTES'].iloc[0]])
        # drop NUM_NOTES
        df = df.drop(columns=['NUM_NOTES'])
        # concat all notes for each patient
        df = df.groupby('HADM_ID').agg({'TEXT': ' '.join, **merger})

    elif comb_strat == 'note_equal_reverse':
        # take an equal number of notes for each patient based on the total length = len
        df = df.sort_values(by=['DIFF'], ascending=False)
        # create a column with number of notes for each patient
        df['NUM_NOTES'] = df.groupby('HADM_ID')['TEXT'].transform('count')
        # shorten each note to len/NUM_NOTES
        df['TEXT'] = df['TEXT'].apply(lambda x: x[:len//df['NUM_NOTES'].iloc[0]])
        # drop NUM_NOTES
        df = df.drop(columns=['NUM_NOTES'])
        # concat all notes for each patient
        df = df.groupby('HADM_ID').agg({'TEXT': ' '.join, **merger})
        
    else:
        print('comb_strat not valid')
        raise ValueError('comb_strat not valid')

    # reset index and drop everything but TEXT and EXPIRE_FLAG
    df = df.reset_index()
    df = df[['TEXT', 'EXPIRE_FLAG']]

    # save df as temp_csv.csv
    df.to_csv('temp_csv.csv')
    
    # load temp_csv.csv in as a dataset
    dataset = load_dataset('csv', data_files='temp_csv.csv')
    dataset = dataset.rename_column("TEXT", "text")
    dataset = dataset.rename_column("EXPIRE_FLAG", "label")
    preprocessed_dataset = dataset.map(preprocess)
    # delete temp_csv.csv
    os.remove('temp_csv.csv')
    return preprocessed_dataset

def main(args):
    # create tokenizer by name: Bio_ClinicalBERT, microsoft/DialogRPT-updown, roberta-base, distilbert-base-uncased, Clinical-Longformer
    if args.tokenizer == 'Bio_ClinicalBERT':
        from transformers import AutoTokenizer
        max_len = 128
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    elif args.tokenizer == 'microsoft/DialogRPT-updown':
        from transformers import AutoTokenizer
        max_len = 128
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialogRPT-updown")
    elif args.tokenizer == 'roberta-base':
        from transformers import RobertaTokenizerFast
        max_len = 128
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif args.tokenizer == 'distilbert-base-uncased':
        from transformers import AutoTokenizer
        max_len = 512
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    elif args.tokenizer == 'Clinical-Longformer':
        from transformers import LongformerTokenizerFast
        max_len = 4096
        tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
    else:
        print('tokenizer not valid')
        raise ValueError('tokenizer not valid')

    # choose model, options include:
    if args.model == 'Bio_ClinicalBERT':
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)
    elif args.model == 'microsoft/DialogRPT-updown':
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/DialogRPT-updown", num_labels=2)
    elif args.model == 'roberta-base':
        from transformers import RobertaForSequenceClassification
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    elif args.model == 'distilbert-base-uncased':
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    elif args.model == 'Clinical-Longformer':
        from transformers import LongformerForSequenceClassification
        model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels = 2)
    else:
        print('model not valid')
        raise ValueError('model not valid')


    # load data using make_dataset, calling all arguments
    preprocessed_dataset = make_dataset(file_loc = args.file_loc, time_max = args.time_max, 
        comb_strat = args.comb_strat, len = max_len, strip_num = args.strip_num)
    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length = max_len, pad_to_max_length = True, truncation=True)
    
    tokenized_datasets = preprocessed_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets['train']
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
    

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    

    lr = args.lr 
    num_epochs = args.num_epochs 
    batch_size = args.batch_size 

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", learning_rate = lr, 
        num_train_epochs = num_epochs, per_device_train_batch_size = batch_size, per_device_eval_batch_size = batch_size)

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = small_train_dataset,
        eval_dataset = small_eval_dataset,
        compute_metrics = compute_metrics,
    )

    trainer.train() 

    print('/n/n/n Training Complete, Now Evaluating /n/n/n')
    trainer.evaluate() 

    roc_auc_score = evaluate.load("roc_auc")
    refs = small_eval_dataset['label']
    
    predictions = trainer.predict(small_eval_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    results = roc_auc_score.compute(references=predictions.label_ids, prediction_scores=preds)
    print("/n/n/n ROC AUC Score: ", round(results['roc_auc'], 2), "/n/n/n")
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_loc', type=str, default='TBI_notes_unsorted.csv')
    parser.add_argument('--time_max', type=int, default=24)
    parser.add_argument('--comb_strat', type=str, default='random_sent')
    parser.add_argument('--strip_num', action='store_true')
    parser.add_argument('--tokenizer', type=str, default='Bio_ClinicalBERT')
    parser.add_argument('--model', type=str, default='Bio_ClinicalBERT')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)

