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

2. download nltk model
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
## Example