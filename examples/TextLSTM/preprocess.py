from utils import *
import random

random.seed(22)

# filter_json_preprocess("./data/yelp_academic_dataset_review.json", "./data")
# sample_review("./data/filtered_data.csv", "./data", 30000)
# merge_csv([
#     # "./data/useless.csv",
#     "./data/not_very_useful.csv",
#     "./data/very_useful.csv"
# ], "./data")

# count_review_length("./data/merged_data.csv")

review_preprocessing(
    merged_review_csv_dir = "./data/merged_data.csv", 
    glove_embedding_dir = "/Users/shengguili/Documents/NTU Course/Year 3 Sem 1/CZ4032 Data Analytics & Mining/Assignment/data/glove.6B.50d.txt",
    output_dir = "./data", 
    train_ratio = 0.9, 
    keep_ratio = 0.2,
    review_length_max = 300,
    review_length_min = 10, 
    seq_length = 200,
    embedding_length=50
    )