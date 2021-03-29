import math
import numpy as np
from keras.layers import Embedding

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D
from keras.layers import MaxPool1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from embeddings import MyEmbedding
from arguments import parse_arguments

args = parse_arguments()

class TextClassifier(object):
    def __init__(self, t_words, params):
        super(TextClassifier, self).__init__()

		# Parameters regarding text preprocessing
        self.t_words = t_words
        self.max_seq_len = params.max_seq_len
        self.embedding_size = params.embedding_size
        self.max_seq_len = params.max_seq_len
        self.dropout = params.dropout
        
        # Load pre-saved npy weight matrix
        embedding_matrix = np.load('models/embedding_weights.npy')
        # embedding_matrix = MyEmbedding(params).load_embeddings(self.t_words)
        
        self.embedding_layer = Embedding(len(self.t_words.word_index) + 1,
                                        self.embedding_size,
                                        weights = [embedding_matrix],
                                        input_length = self.max_seq_len,
                                        trainable = False)

    def create_model(self, num_targets, params):
        
        filter_sizes = [3,4,5]
        num_filters = 512
        drop =self.dropout

        inputs = Input(shape=(self.max_seq_len,), dtype='int32')
        embedding = self.embedding_layer(inputs)

        print(embedding.shape)
        reshape = Reshape((self.max_seq_len,self.embedding_size,1))(embedding)
        print(reshape.shape)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], self.embedding_size), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], self.embedding_size), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], self.embedding_size), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(self.max_seq_len - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(self.max_seq_len - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(self.max_seq_len - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=num_targets, activation='softmax')(dropout)

        # this creates a model that includes
        self.model = Model(inputs=inputs, outputs=output)

        self.checkpoint = ModelCheckpoint(args.out_dir + '/weights_cnn_sentence.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        adam = Adam(lr=params.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
        print(self.model.summary())
        