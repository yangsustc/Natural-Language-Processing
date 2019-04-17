
# -*- coding: utf-8 -*-
from keras import regularizers#regularization
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Flatten,Activation, Input
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import SeparableConv1D

from keras import regularizers

import numpy as np
import pandas as pd

from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional

from keras.utils import plot_model#save flow diagram
from IPython.display import SVG#draw flow diagram
from keras.utils.vis_utils import model_to_dot

def plot_diagram(model,name):
    plot_model(model, to_file=name)#save file
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
# load embedding as a dict
def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename,'r',encoding='utf-8')
    lines = file.readlines()
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab,vocab_size,embedding_size):
    # total vocabulary size plus 0 for unknown words
    # vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, embedding_size))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
    	if i >= vocab_size:
    		continue
    	vector = embedding.get(word)
    	if vector is not None:
    		weight_matrix[i]=vector
    return weight_matrix
def get_embedding_layer(word_index,USE_PRE=0,vocab_size=10000,embedding_size=100,max_length=256):
    if USE_PRE != 0 :
        if USE_PRE == 1:
            # load embedding from GloVe
            raw_embedding = load_embedding('glove.6B.300d.txt')
        elif USE_PRE == 2: 
            # load embedding from GoogleNews
            raw_embedding = load_embedding('GoogleNews-vectors-negative300.txt')
        else:
            print('USE_PRE can only takes values: 0 for non pre-trained, 1 for using GloVe and 2 for GoogleNews Word2vec')      
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, word_index,vocab_size,embedding_size)
        # create the embedding layer
        del raw_embedding
        embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_vectors], input_length=max_length, trainable=False)
    else:
        embedding_layer=Embedding(vocab_size, embedding_size, input_length=max_length)
    return embedding_layer

def single_LSTM(embedding_layer,cell_num=100,dropout_rate=0.2,recurrent_dropout_rate=0.1):
    # word embedding+LSTM+dense
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(cell_num, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def Bi_GRU(embedding_layer,cell_num=128,dropout_rate=0.2,recurrent_dropout_rate=0.1):
    # word embedding +Bi-GRU +dense
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(GRU(cell_num, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate, return_sequences=True)))
    model.add(Bidirectional(GRU(cell_num, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def C_RNN_series(embedding_layer,filter_size=128,kernel_size=3,cell_num=128,dropout_rate=0.2,recurrent_dropout_rate=0.1):
    # word embedding +conv+maxpooling+Bi-GRU*2 +dense
    #series connecting CNN+RNN
    model = Sequential()
    model.add(embedding_layer)    
    model.add(Conv1D(filter_size, kernel_size, padding='same', strides = 1,activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GRU(cell_num, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate, return_sequences=True))
    model.add(GRU(cell_num, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def C_RNN_Parallel(embedding_layer,max_length=256,filter_size=256,kernel_size=3,cell_num=256,dropout_rate=0.2,recurrent_dropout_rate=0.1):
    # 词嵌入-卷积池化-全连接 ---拼接-全连接  -双向GRU-全连接
    #structure: word_embedding--conv--dense--concatenate--dense--BI-GRU--dense
    main_input = Input(shape=(max_length,), dtype='float64')
    embed = embedding_layer(main_input)
    cnn = Conv1D(filter_size, kernel_size, padding='same', strides = 1, activation='relu')(embed)
    cnn = MaxPooling1D(pool_size=4)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(256)(cnn)
    rnn = Bidirectional(GRU(cell_num, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))(embed)
    rnn = Dense(256)(rnn)
    con = concatenate([cnn,rnn], axis=-1)
    main_output = Dense(1, activation='sigmoid')(con)
    model = Model(inputs = main_input, outputs = main_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def text_cnn(embedding_layer,max_length=256):
    #structure: pre_trained glove--conv*3--concatenate--dense
    # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
    main_input = Input(shape=(max_length,), dtype='float64')
    # word2vec
    embedder = embedding_layer
    embed = embedder(main_input)
    # filter size 3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides = 1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=4)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides = 1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=4)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides = 1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=4)(cnn3)
    # combine 
    cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs = main_input, outputs = main_output)

    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def Channel_cnn(embedding_layer,filter_sizes,feature_maps,dropout_rate=.3,hidden_units=100,max_length=256):
    def building_block(input_layer, filter_sizes, feature_maps):
        """ 
        Creates several CNN channels in parallel and concatenate them 
        
        Arguments:
            input_layer : Layer which will be the input for all convolutional blocks
            filter_sizes: Array of filter sizes
            feature_maps: Array of feature maps
            
        Returns:
            x           : Building block with one or several channels
        """
        channels = []
        for ix in range(len(filter_sizes)):
            x = create_channel(input_layer, filter_sizes[ix], feature_maps[ix])
            channels.append(x)
            
        # Checks how many channels, one channel doesn't need a concatenation
        if (len(channels)>1):
            x = concatenate(channels)
        return x
    def create_channel( x, filter_size, feature_map):
        """
            Creates a layer, working channel wise
            
            Arguments:
                x           : Input for convoltuional channel
                filter_size : Filter size for creating Conv1D
                feature_map : Feature map 
                
            Returns:
                x           : Channel including (Conv1D + GlobalMaxPooling + Dense + Dropout)
        """
        x = SeparableConv1D(feature_map, kernel_size=filter_size, activation='relu', strides=1, padding='same',depth_multiplier=4)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(hidden_units)(x)
        x = Dropout(dropout_rate)(x)
        return x
    #channel cnn
    word_input = Input(shape=(max_length,), dtype='int32', name='word_input')
    x = embedding_layer(word_input)
    x = Dropout(dropout_rate)(x)
    x  = building_block(x, filter_sizes, feature_maps)
    x = Activation('relu')(x)
    prediction = Dense(1, activation='sigmoid')(x)
    model=Model(inputs=word_input, outputs=prediction)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def C_RNN_series2(embedding_layer,reg_coef=0.00,dropout_rate_conv=0,filter_size=128,kernel_size=3,cell_num=128,dropout_rate=0.2,recurrent_dropout_rate=0.1):
    # word embedding +conv+maxpooling+Bi-GRU*2 +dense
    #series connecting CNN+RNN
    model = Sequential()
    model.add(embedding_layer)    
    model.add(Conv1D(filter_size, kernel_size, padding='same', strides = 1,activation='relu',kernel_regularizer=regularizers.l2(reg_coef)))
    model.add(Dropout(dropout_rate_conv))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GRU(cell_num, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate, return_sequences=True))
    model.add(GRU(cell_num, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def C_RNN_Parallel2(embedding_layer,reg_coef=0.00,dropout_rate_conv=0,max_length=256,filter_size=256,kernel_size=3,cell_num=256,dropout_rate=0.2,recurrent_dropout_rate=0.1):
    # 词嵌入-卷积池化-全连接 ---拼接-全连接  -双向GRU-全连接
    #structure: word_embedding--conv--dense--concatenate--dense--BI-GRU--dense
    main_input = Input(shape=(max_length,), dtype='float64')
    embed = embedding_layer(main_input)
    cnn = Conv1D(filter_size, kernel_size, padding='same', strides = 1, activation='relu',kernel_regularizer=regularizers.l2(reg_coef))(embed)
    cnn = Dropout(dropout_rate_conv)(cnn)
    cnn = MaxPooling1D(pool_size=4)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(256)(cnn)
    rnn = Bidirectional(GRU(cell_num, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))(embed)
    rnn = Dense(256)(rnn)
    con = concatenate([cnn,rnn], axis=-1)
    main_output = Dense(1, activation='sigmoid')(con)
    model = Model(inputs = main_input, outputs = main_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def save_history(history,model_name):
   #history: fit history
   # model_name:String, to generate a csv file to save the history
    import pandas as pd
    history_crnn=pd.DataFrame({'acc_train':history.history["acc"],'acc_val':history.history["val_acc"],
                                       'loss_train':history.history["loss"],'loss_val':history.history["val_acc"]
                         
                          })
    history_crnn.to_csv(model_name+'.csv')

def transfer_sentence_to_idx(string,word_index):
    import numpy as np
    from tensorflow import keras
    def map_words(x):
        flag=word_index.get(x)
        if flag is None:
            return 0
        elif flag>9999 :
            return 0
        else:
            return flag
    aa=[[map_words(word) for word in string.split(' ') ]]
    return  keras.preprocessing.sequence.pad_sequences(np.array(aa),
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

def predict_list(reviews,word_index,model):
    import numpy as np
    evalt = []
    for i, (string,_) in enumerate(reviews):
        evalt.append(round(float(model.predict(transfer_sentence_to_idx(string,word_index))[0][0]),2))
    return np.ndarray.transpose(np.array(evalt))
        
def printer(reviews_data,models,word_index):
    import numpy as np
    short_rv = []
    stars = []
    for ii in range(len(reviews_data)):
        short_rv.append(reviews_data[ii][0][:15]+'...')
        stars.append(reviews_data[ii][1])
    data = predict_list(reviews_data,word_index,globals()[models[0]])
    for ii in range(len(models)-1):
        data = np.row_stack((data,predict_list(reviews_data,word_index,globals()[models[ii]])))
    data = np.ndarray.transpose(data)
    dash = '-' * (25+15*len(models)+15)
    row_format ="{:<25}"+"{:^15}" * (len(models))+"{:>15}"
    print(dash)
    print(row_format.format("REVIEWS", *models,'Real stars'))
    print(dash)
    for review, row,star in zip(short_rv,data,stars):
        print(row_format.format(review, *row,star))
    print(dash)


