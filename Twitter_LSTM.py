

### much of the code was copied and then modified https://github.com/WillKoehrsen/recurrent-neural-networks/tree/master/notebooks


from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, concatenate
from keras.models import Sequential, load_model, Model
import pandas as pd
import json

train2_df= pd.read_csv("df2.csv")
sn_index = json.load(open("sn_index.json", "r"))
word_index = json.load(open("word_index.json", "r"))
index_sn=json.load(open("index_sn.json", "r"))
index_word=json.load(open("index_word.json", "r"))



model = Sequential()
model.add(
    Embedding(
        input_dim=75001,
        output_dim=80,
        weights=None,
        trainable=True))

model2 = Sequential()
model2.add(
    Embedding(
        input_dim=5001,
        output_dim=8,
        weights=None,
        trainable=True))
concat_layers = concatenate([model.output, model2.output])
layer = LSTM(64, return_sequences=False,  dropout=0.1,recurrent_dropout=0.1 )(concat_layers)
layer1    = Dense(64, activation='relu')(layer)
layerout = Dense(75001, activation='softmax')(layer1)
final_model = Model([model.input, model2.input], [layerout])

final_model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


import random 
import numpy as np

TS = 10000
L = len(train2_df)
for i in range(1000):
    rs = random.sample(range(L),TS)
    X = train2_df.features.iloc[rs]
    A = train2_df.authors.iloc[rs]
    y = train2_df['labels'].iloc[rs]
    y= [min(p,75000) for p in list(y)]
    A = np.asarray(A)
    AA = np.resize(A, [TS, 1])
    AA = np.repeat(AA, 8, axis = 1) 
    X = list(X)
    X = [np.asarray(l) for l in X]
    X = np.asarray(X)
    y_train = np.zeros((len(y), 75001), dtype=np.int8)
    for example_index, word_index in enumerate(y):  y_train[example_index, word_index] = 1
    final_model.fit([X,AA], y_train, epochs = 2, batch_size = 155, verbose = 1)



def get_embeddings(model):
    """Retrieve the embeddings in a model"""
    embeddings = model.get_layer(index = 0)
    embeddings = embeddings.get_weights()[0]
    embeddings = embeddings / np.linalg.norm(embeddings, axis = 1).reshape((-1, 1))
    embeddings = np.nan_to_num(embeddings)
    return embeddings


def find_closest(query, embedding_matrix, word_idx, idx_word, n = 10):
    idx = word_idx.get(query, None)
    if idx is None:
        print('{query} not found in vocab.')
        return
    else:
        vec = embedding_matrix[idx]
        if np.all(vec == 0):
            print('{query} has no pre-trained embedding.')
            return
        else:
            # Calculate distance between vector and all others
            dists = np.dot(embedding_matrix, vec)
            idxs = np.argsort(dists)[::-1][:n]
            sorted_dists = dists[idxs]
            closest = [idx_word[i] for i in idxs]
    print('Query: {query}\n')
    # Print out the word and cosine distances
    for word, dist in zip(closest, sorted_dists):
        print 'Word', word, 'Cosine Similarity:', dist
        

embeddings = get_embeddings(model)
find_closest('lol', embeddings, word_index, index_word, n = 10)


embeddings = get_embeddings(model2)
embeddings = get_embeddings(model2)
find_closest('BarCampEugene', embeddings, sn_index, index_sn, n = 10)






