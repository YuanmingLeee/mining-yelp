# Data Mining for Yelp Dataset

## Authors
Group 16 _(Ordered by alphabet)_  
- Bian WU [\[Email\]](mailto:BWU007@e.ntu.edu.sg) [\[GitHub\]](https://github.com/BB-27)
- Lingzhi CAI [\[Email\]](mailto:LCAI004@e.ntu.edu.sg)[\[GitHub\]](https://github.com/lzcaisg)
- Shenggui LI [\[Email\]](mailto:C170166@e.ntu.edu.sg) [\[GitHub\]](https://github.com/FrankLeeeee)
- Yanxi ZENG [\[Email\]](mailto:ZENG0112@e.ntu.edu.sg) [\[GitHub\]](https://github.com/Splashingsplashes)
- Yuanming LI [\[Email\]](mailto:yli056@e.ntu.edu.sg) [\[GitHub\]](https://github.com/YuanmingLeee)

## Install

Please make sure that you have installed Conda, and have at least one CUDA device.  
```shell script
# install Conda env
conda env create -f environment.yml
# activate Conda env
conda activate ntu-dm

# add pytroch with specific cuda version
conda install pytorch cudatoolkit=<your cuda version> -c pytorch -y

# create data folders
mkdir -p {data,output}

# set python path when run
export PYTHONPATH="${PWD}"
```

## Download Data
1. Download dataset  
We are using [Yelp dataset](https://www.yelp.com/dataset/challenge) provided by [Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset/download). The dataset contains 5 JSON files, 8 GB after unzipped. Please download the data from Yelp or Kaggle, move it into `data` folder and unzip it:
    ```shell script
    mv yelp-dataset.zip data/
    cd data
    unzip yelp-dataset.zip
    cd -
    ```

2. Download nltk model
    ```shell script
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
    ```
3. Download pre-trained weights
Download [weights and data](https://drive.google.com/open?id=1_l5U6HwtmzvNqYLaz0uWshDO5-zIycQG), move and unzip it:
    ```shell script
    mv 4032-data.zip data/
    cd data
    unzip 4032-data.zip
    cd -
    ```

## Prepare Data
You should seek for help if you do not have a high performance computer with memory larger than
40GB, as creating the database from Yelp dataset invokes large data frame processing. 
If you do not
have access to such resources, please drop us a email for the request of linux-built SQLite 
database file. The reason that we do not provide a processed file via google drive is as following from 
_Yelp Dataset Term of Use_:
> 4  
> A. display, perform, or distribute any of the Data, or use the Data to update or create
your own business listing information (i.e. you may not publicly display any of the Data to any
third party, especially reviews and other user generated content, as this is a private data set
challenge and not a license to compete with or disparage with Yelp);  
> ...  
> E. create, redistribute or disclose any summary of, or metrics related to, the Data (e.g.,
the number of reviewed business included in the Data and other statistical analysis) to any third
party or on any website or other electronic media not expressly covered by this Agreement, this
provision however, excludes any disclosures necessary for academic purposes, including
without limitation the publication of academic articles concerning your use of the Data;  
> ...  
> H. rent, lease, sell, transfer, assign, or sublicense, any part of the Data;  
> ...  
> I. modify, rate, rank, review, vote or comment on, or otherwise respond to the content
contained in the Data;    
```shell script
# create SQLite database
python scripts/create_tables.py

# loading data into the database (slow)
python scripts/load_data.py

# process dataset (very slow)
python scripts/process_dataset.py

# pretrain model
python scripts/pretrain-model.py
```

## Examples
**Note**: You output files are all in `output/`
#### Statistical Learning Models
**Note**: SVM and XGBoost may takes a very long time in prediction and testing.
1. Train XGBoost model for predicting usefulness
    ```shell script
    python train-statistical-learning-models.py xgboost
    ```

2. Train SVM model for predicting usefulness
    ```shell script
    python train-statistical-learning-models.py svm
    ```

3. Train Logistic Regression model for predicting usefulness
    ```shell script
    python train-statistical-learning-models.py logistic
    ```
4. Predict summary report
    ```shell script
    python helper.py pred-statistical <path/to/saved/model.pkl>
    ```
5. Plot ROC graph
    ```shell script
    python helper.py plot-roc <path/to/saved/model.pkl>
    ```
   **Note**: when you run it in shell, you shall enable X server first.
    

#### Deep Learning Models
1. Train user elite classification
    ```shell script
    python train-user-elite.py
    ```
    You may see script arguments by
    ```shell script
    python train-user-elite.py -h
    ```
2. Train LSTM usefulness classification  
    ```shell script
    python train_text_lstm.py
    ```
   
    You may see script arguments by
    ```shell script
    python train_text_lstm.py -h
    ```
3. Train multimodal classifier using pretrained LSTM and user elite model
    For pretrained TextLSTM model, you need to put the mapping.pickle, 
    pretrained_weights.npy and useful_pred_lstm_weights.pth in ./data folder.
    
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
      **Note**: when you run it in shell, you shall enable X server first.
5. Find confusion matrix
    ```shell script
    python helper.py confusion-mtx --name <model-name> --model-weight <model/weight/path.pth> \
    --split-ratio 0.2 <model/configuration/path.yaml>
    ```
   split-ratio is not needed for visualizing the TextLSTM alone.
