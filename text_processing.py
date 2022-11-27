import tensorflow as tf
import pandas as pd

import nltk
nltk.download('punkt')

from nltk import word_tokenize

# url='https://raw.githubusercontent.com/jacob-hansen/NLP_in_EHR_2022/910d9f0fcfeab083dff53ea2e2969c175cc816a0/train.csv'
# train_df = pd.read_csv(url)


# then run the following in the terminal:
# pip install spacy
# python -m spacy download en_core_web_sm
import spacy
#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words


def custom_tokenizer(sentence):
    # lower case
    sentence = sentence.lower()
    # split by the label/sentence separator
    sents = sentence.split('. val is ')
    # seperate out the label and the next sentence
    sents = [sents[0]] + [i for i in sents[1].split('. ', 1)]
    # remove any trailing whitespace
    sents = [i.strip() for i in sents]
    # remove stop words in every 1,3,5... sentence
    # and apply tokenization
    for i in range(0, len(sents), 2):
        # remove stop words
        sents[i] = [word for word in word_tokenize(sents[i]) if word not in sw_spacy]
    # flatten sents 
    return_sents = []
    for i in range(len(sents)):
        if i % 2 == 0:
            return_sents.extend(sents[i])
        else:
            return_sents.append(sents[i])
    # repeat the label 3 times (loc = 1,3,5..) for every sentence 
    final_sents = []
    for i in range(0, len(return_sents)):
        if i % 2 == 0:
            final_sents.append(return_sents[i])
        else:
            final_sents.append(return_sents[i])
            final_sents.append(return_sents[i])
            final_sents.append(return_sents[i])

    return return_sents

def custom_tokenizer_bow(sentence):
    # lower case
    sentence = sentence.lower()
    # split by the label/sentence separator
    sents = sentence.split(' val is ')
    # seperate out the label and the next sentence
    sents = [sents[0]] + [i for i in sents[1].split('. ', 1)]
    # remove any trailing whitespace
    sents = [i.strip() for i in sents]
    # remove stop words in every 1,3,5... sentence
    for i in range(0, len(sents), 2):
        # remove stop words
        sents[i] = [word for word in word_tokenize(sents[i]) if word not in sw_spacy]
    # join the words in each sentence back together
    for i in range(0, len(sents), 2):
        sents[i] = ' '.join(sents[i])
    # use bag of words to self-classify the sentences
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    # full_sents = every 1, 3, 5... sentence
    full_sents = []
    for i in range(0, len(sents), 2):
        full_sents.append(sents[i])
    # fit the vectorizer to the full sentences
    X = vectorizer.fit_transform(full_sents)
    # get the feature names
    feature_names = vectorizer.get_feature_names()
    # get the bag of words for each sentence
    bow_sents = []
    for i in range(0, len(sents)):
        if i % 2 == 0:
            vec = vectorizer.transform([sents[i]])
            feature_name = feature_names[vec.nonzero()[1][0]]
            bow_sents.append(feature_name)
        else: 
            bow_sents.append(float(sents[i]))

    print(bow_sents)
    breaksys()
    # flatten sents 
    return_sents = []
    for i in range(len(sents)):
        if i % 2 == 0:
            return_sents.extend(sents[i])
        else:
            return_sents.append(sents[i])
    # repeat the label 3 times (loc = 1,3,5..) for every sentence 
    final_sents = []
    for i in range(0, len(return_sents)):
        if i % 2 == 0:
            final_sents.append(return_sents[i])
        else:
            final_sents.append(return_sents[i])
            final_sents.append(return_sents[i])
            final_sents.append(return_sents[i])

    return return_sents

# # apply the custom tokenizer to the dataframe
# train_df['tokenized'] = train_df['X_train'].apply(custom_tokenizer)


# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# target = train_df['y_train'].values
# target = to_categorical(target)

# # convert train_df['tokenized'] to a tensor
# # and pad the sequences to be the same length
# max_len = 200
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(train_df['tokenized'].values)
# X = tokenizer.texts_to_sequences(train_df['tokenized'].values)
# X = pad_sequences(X, maxlen=max_len)
