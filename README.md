# CNN-SHAP-text-classification
A CNN based framework for text classification problem implemented in Keras with a local model explanation using SHAP DeepExplainer class.


## Requirements
1.  keras: `2.4.3`
2.  scikit-learn: `0.24.1`
3.  Numpy: `1.19.2`
4.  shap: `0.39.0`
5.  tensorflow: `2.4.1`
6.  matplotlib: `3.3.4`


## Dataset
The [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) is collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups.

## Directory Structure
```
├── models
│   ├── embedding_weights.npy
│   ├── model_arch.png
│   ├── model_epochs_10_batch_size_64.h5
│   └── weights_cnn_sentence.hdf5
├── model_summary.txt
├── README.md
├── results
│   ├── 1-summary_plot.pdf
│   ├── 2-explanation_plot_of_given_prediction.pdf
│   ├── 3-kernel_explainer_plot.pdf
│   ├── 4-output_expectation_for_gven_class.htm
│   ├── accuray_plot.eps
│   └── loss_plot.eps
├── src
│   ├── arguments.py
│   ├── dataset_loader.py
│   ├── embeddings.py
│   ├── explanibility.py
│   ├── main.py
│   ├── model.py
│   └── parameters.py
└── wiki-news-300d-1M.vec
```

## Getting started
-   **Help**: for instructions on how to run the script with appropriate arguments.
    ```
    python src/main.py --help
    usage: main.py [-h] 
                    [-emb_f EMB_F] 
                    [-out_dir OUT_DIR] 
                    [-results_dir RESULTS_DIR] 
                    [-learning_rate LEARNING_RATE] 
                    [-epochs EPOCHS] [-dropout DROPOUT] 
                    [-embedding_size EMBEDDING_SIZE] 
                    [-max_seq_len MAX_SEQ_LEN] 
                    [-batch_size BATCH_SIZE]
                    [-save_model SAVE_MODEL]

    optional arguments:
    -h, --help                        show this help message and exit
    -emb_f EMB_F                      path to pre-trained fastext embeddings (*.vec file)
    -out_dir OUT_DIR                  path to save trained model [default: models]
    -results_dir RESULTS_DIR          path to save training plots [default: results]
    -learning_rate LEARNING_RATE      learning rate [default: 0.001]
    -epochs EPOCHS                    number of training epochs [default: 10]
    -dropout DROPOUT                  the probability for dropout [default: 0.25]
    -embedding_size EMBEDDING_SIZE    number of embedding dimension [default: 300]
    -max_seq_len MAX_SEQ_LEN          maximum sequence length [default: 100]
    -batch_size BATCH_SIZE            batch size while training [default: 64]
    -save_model SAVE_MODEL            save model [default: True]

    ```
-   **Training and Evaluation**- Start training followed by evaluation. The trained model is thus saved in the default directory `models/*.h5` and the evaluation plots are saved in `results`. The approximate training time is 400 s in CPU with 24 cores and 16G of memory. Loading pre-trained embeddings can also take some additional time.
    ```
    python src/main.py
    ```
-   **SHAP-DeepExplained**- The main goal of this task is to use SHAP to assess the importance of local features in CNN-based text classification models. This would compute the Shapely values to allow generate multiple model interpretability graphics. This would then be followed by using the obtained results from SHAP to demonstrate the local explanation and thus analyze how Shapely values support the findings obtained from CNN.

    Simply run `python src/explanibility.py models/weights_cnn_sentence.hdf5` and check the saved plots in `results/`.
    ```
    python src/explanibility.py --help
    usage: explanibility.py [-h] 
                            [-results_dir RESULTS_DIR] 
                            model_f

    Local model explanation using the SHAP DeepExplainer class.

    positional arguments:
    model_f                     full path to trained model

    optional arguments:
    -h, --help                  show this help message and exit
    -results_dir RESULTS_DIR    path to save training plots

    ```


## Runtime
-   Training and evaluation- 400 s
-   Explanability- 3067 s


## Results
All results and plots can be found in `results/`.


## References
1.  Yoon Kim. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
2.  Patricia Ferreiro. [Explaining CNNs for Text Classification using SHAP](https://www.kaggle.com/patricia92fa/explaining-cnns-for-text-classification-using-shap/comments)
3.  Wei Zhao et al.. [SHAP values for Explaining CNN-based Text Classification Models](https://arxiv.org/abs/2008.11825)
4.  [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)

