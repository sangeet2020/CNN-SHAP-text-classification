#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-02-23 01:52:54
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-03-18 01:32:15

"""
This is the main script
"""

import os
import sys
import time
import numpy as np
from arguments import parse_arguments

from dataset_loader import DataLoader
from parameters import Parameters
from model import TextClassifier

import keras
import shap
import matplotlib.pyplot as plt


def prepare_data():
    """ Load and pre-process the dataset """
    dl = DataLoader(max_seq_len=Parameters.max_seq_len)
    dl.load_data()
    dl.get_targets()
    dl.get_sentences()
    dl.get_id2target()
    dl.get_target2id()
    dl.targets2one_hot()
    dl.tokenize()
    dl.build_vocabulary()
    dl.padding()
    dl.split_data()
    
    return (
        {
            'x_train': dl.X_train,
            'y_train': dl.y_train,
            'x_test': dl.X_test,
            'y_test': dl.y_test,
            'x_valid': dl.X_validation,
            'y_valid': dl.y_validation
        },
        dl.t_words,
        dl.target2id
    )

def training_report(model_training, args):
    """ Plot training history"""
    # summarize history for accuracy
    plt.plot(model_training.history['acc'])
    plt.plot(model_training.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.results_dir + '/accuray_plot.eps', format='eps')
    plt.close()
    
    # summarize history for loss
    plt.plot(model_training.history['acc'])
    plt.plot(model_training.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.results_dir + '/loss_plot.eps', format='eps')
    plt.close()
    
def main():
    """ main method """
    
    # Prepare the data
    args = parse_arguments()
    print("Loading dataset....")
    data, t_words, target2id = prepare_data()
    print("--Done--")
    
    # Initialize the model
    start = time.time()
    tc = TextClassifier(t_words, Parameters)
    tc.create_model(len(target2id), Parameters)
    model = tc.model
    
    # Train and Evaluate the pipeline
    print("Traning Model...")
    batch_size = Parameters.batch_size
    epochs = Parameters.epochs
    model_training = model.fit(data["x_train"], data["y_train"], 
                                batch_size=batch_size, 
                                epochs=epochs, 
                                verbose=1, 
                                callbacks=[tc.checkpoint], 
                                validation_data=(data["x_valid"], data["y_valid"]))
    
    end = time.time()
    print("*** Training Complete ***")
    print("Training runtime: {:.2f} s".format(end-start))
    
    loss, accuracy = model.evaluate(data['x_test'], data['y_test'], verbose = 1)
    print("Test Loss: {:.2f} %,\nTest Accuracy: {:.2f} %".format(loss*100, accuracy*100))
    
    if args.save_model:
        PATH = args.out_dir + "/model_epochs_" + str(epochs) +"_batch_size_" + str(batch_size) + ".h5"
        model.save(PATH)
        print("Model saved: ",PATH)
        keras.utils.plot_model(model, args.out_dir + '/model_arch.png', show_shapes=True)
        
    training_report(model_training, args)
    

if __name__ == "__main__":
    main()