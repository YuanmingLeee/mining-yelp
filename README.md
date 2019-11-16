# Data Mining for Yelp Dataset

## Install

#### Option 1. Step-by-step Installation
Please make sure that you have installed Conda.
```shell script
# install Conda env
conda env create -f environment.yml
# activate Conda env
conda activate ntu-dm

# create data folders
mkdir -p {data,output}
```

## Download Data
1. Download dataset  
We are using [Yelp dataset](https://www.yelp.com/dataset/challenge) provided by [Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset/download). The dataset contains 5 JSON files, 8 GB after unzipped.

2. Download nltk model
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```
3. Download pre-trained weights

## Prepare Data

## Example
1. Train user elite classification
    ```shell script
    python train-user-elite.py
    ```
    You may see script arguments by
    ```shell script
    python train-user-elite.py -h
    ```
2. Train LSTM usefulness classification
 
3. Train multimodal classifier using pretrained LSTM and user elite model
    ```shell script
    python train-multimodal-classifier.py
    ```
    You may want to change the configuration by supplying another configuration files:
    ```shell script
    python train-multimodal-classifier.py --config=<path/to/config.yaml>
    ```
    You may see script arguments by
    ```shell script
    python train-multimodal-classifier.py -h
    ```
4. Visualize loss and accuracy
    ```shell script
    python helper.py plot <path/to/your/statistic/result.pkl>
    ```
5. Find confusion matrix
    ```shell script
    python helper.py confusion-mtx --name <model-name> --model-weight <model/weight/path.pth> \
    --split-ratio 0.2 <model/configuration/path.yaml>
    ```
