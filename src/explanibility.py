#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-02-23 01:52:54
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-03-18 01:32:15

"""
Local model explanation using the SHAP DeepExplainer class.
"""

import os
import sys
import argparse
import numpy as np

from tensorflow import keras
import shap
import torch

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from main import prepare_data
import matplotlib.pyplot as plt


def main():
    """ main method """
    args = parse_arguments()
    os.makedirs(args.results_dir, exist_ok=True)
    data, t_words, target2id = prepare_data()
    
    ## Load model
    model = keras.models.load_model(args.model_f)
    
    ## Explanability
    distrib_samples = data['x_train'][:100]
    session = tf.compat.v1.keras.backend.get_session()
    explainer = shap.DeepExplainer(model, distrib_samples, session)
    num_explanations = 10
    # Get SHAP values
    shap_values = explainer.shap_values(data['x_test'][:num_explanations])
    num2word = {}
    for w in t_words.word_index.keys():
        num2word[t_words.word_index[w]] = w
    x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), data['x_test'][i]))) for i in range(num_explanations)])
    
    plt.figure()
    shap.summary_plot(shap_values, feature_names = list(num2word.values()), class_names = list(target2id.keys()), show=False)
    plt.savefig(args.results_dir + ' /1-summary_plot.pdf', format='pdf', dpi=1200)
    plt.close()
    
    ## Deep Exaplainer
    # init the JS visualization code
    shap.initjs()
    # create dict to invert word_idx k,v order
    num2word = {}
    for w in t_words.word_index.keys():
        num2word[t_words.word_index[w]] = w
    x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), data['x_test'][i]))) for i in range(10)])

    # plot the explanation of a given prediction
    class_num = 2
    input_num = 2
    shap.force_plot(explainer.expected_value[class_num], shap_values[class_num][input_num], x_test_words[input_num], show=False, matplotlib=True)
    plt.savefig(args.results_dir + '/2-explanation_plot_of_given_prediction.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.close()
    
    
    # reverse idx for labels
    num2label = {}
    for w in target2id.keys():
        num2label[target2id[w]] = w
    x_test_labels = np.stack([np.array(list(map(lambda x: num2label.get(x, "NONE"), data['x_test'][i]))) for i in range(10)])
    # generate 10 predictions
    y_pred = model.predict(data['x_test'][:10])
    sample = 8
    true_class = list(data['y_test'][sample]).index(1)
    pred_class = list(y_pred[sample]).index(max(y_pred[sample]))
    # one hot encoded result
    print(f'Predicted vector is {y_pred[sample]} = Class {pred_class} = {num2label[pred_class]}')
    # filter padding words
    print(f'Input features/words:')
    print(x_test_words[sample][np.where(x_test_words[sample] != 'NONE')])
    print(f'True class is {true_class} = {num2label[true_class]}')
    max_expected = list(explainer.expected_value).index(max(explainer.expected_value))
    print(f'Explainer expected value is {explainer.expected_value}, i.e. class {max_expected} is the most common.')
    
    
    ## Kernel Explainer
    # high time consuming step
    kernel_explainer = shap.KernelExplainer(model.predict, distrib_samples)
    kernel_shap_values = kernel_explainer.shap_values(data['x_test'][:num_explanations])
    
    # plot the explanation of a given prediction
    class_num = 2
    input_num = 8
    shap.force_plot(kernel_explainer.expected_value[class_num], kernel_shap_values[class_num][input_num], x_test_words[input_num], show=False, matplotlib=True)
    plt.savefig(args.results_dir + '/3-kernel_explainer_plot.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.close()
    
    # explanations of the output for the given class 
    # y center value is base rate for the given background data
    f = shap.force_plot(kernel_explainer.expected_value[class_num], kernel_shap_values[class_num], x_test_words[:10])
    shap.save_html(args.results_dir + "/4-output_expectation_for_gven_class.htm", f)

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_f", help="full path to trained model")
    parser.add_argument("-results_dir",     default='results', type=str, help="path to save training plots")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()