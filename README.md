# Data Mining for Yelp Dataset

## Install

Please make sure that you have installed Conda, and have at least one CUDA device.  
```shell script
# install Conda env
conda env create -f environment.yml
# activate Conda env
conda activate ntu-dm

# add pytroch with specific cuda version
conda install pytorch cudatoolkit=<your cuda version> -c pytorch

# create data folders
mkdir -p {data,output}
```

## Download Data
1. Download dataset  
We are using [Yelp dataset](https://www.yelp.com/dataset/challenge) provided by [Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset/download). The dataset contains 5 JSON files, 8 GB after unzipped. Please download the data from Yelp or Kaggle, move it into `data` folder and unzip it:
    ```shell script
    mv yelp-dataset.zip data/
    unzip yelp-dataset.zip
    ```

2. Download nltk model
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```
3. Download pre-trained weights


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

# loading data into the database
python scripts/load_data.py

# process dataset
python process_dataset.py
```

## Example
1. Train XGBoost model for predicting usefulness
    You may want to change the connection to MongoDB to read in the csv data file as Pandans Dataframe
    ```shell script
    python Doc2Vec_with_XGBoost.py
    ```

2. Train SVM model for predicting usefulness
     You may want to change the connection to MongoDB to read in the csv data file as Pandans Dataframe
    ```shell script
    python Doc2Vec_with_SVM.py
    ```

3. Train Logistic Regression model for predicting usefulness
    You may want to change the connection to MongoDB to read in the csv data file as Pandans Dataframe
    ```shell script
    python Doc2Vec_with_Logistic_Regression.py
    ```
4. Train user elite classification
    ```shell script
    python train-user-elite.py
    ```
    You may see script arguments by
    ```shell script
    python train-user-elite.py -h
    ```
5. Train LSTM usefulness classification  
    You need to download the text data file and GloVe pre-trained word embedding file and put them
    in the ./data folder. The merged.csv contains sampled text data and their labels while 
    glove.6B.50d.text is the word embedding for 50 dimensions.
    
    First, preprocess the data by process_dataset.py. Remember to comment out the user_elite_cleaned_csv()
    and multimodal_classifier() function calls and only leave text_lstm() function call.
    
    ```shell script
    python scripts/process_dataset.py
    ```
    
    Then train the model.
    ```shell script
    python train_text_lstm.py
    ```
   
    You may see script arguments by
    ```shell script
    python train_text_lstm.py
    ```
6. Train multimodal classifier using pretrained LSTM and user elite model
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
7. Visualize loss and accuracy
    ```shell script
    python helper.py plot <path/to/your/statistic/result.pkl>
    ```
8. Find confusion matrix
    ```shell script
    python helper.py confusion-mtx --name <model-name> --model-weight <model/weight/path.pth> \
    --split-ratio 0.2 <model/configuration/path.yaml>
    ```
   split-ratio is not needed for visualizing the TextLSTM alone.

## Credits

## About Us
_(Ordered by alphabet)_
- Bian WU [\[GitHub\]]()
- Lingzhi CAI [\[GitHub\]]()
- Shenggui LI [\[GitHub\]]()
- Yanxi ZENG [\[GitHub\]]()
- Yuanming LI [\[GitHub\]](https://github.com/YuanmingLeee)
