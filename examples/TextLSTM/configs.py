from os.path import dirname, abspath, join

common_dir = dirname(abspath(__file__))
train_x_dir = join(common_dir, "data", "train_x.npy")
train_y_dir = join(common_dir, "data", "train_y.npy")
test_x_dir = join(common_dir, "data", "test_x.npy")
test_y_dir = join(common_dir, "data", "test_y.npy")
embedding_dir = join(common_dir, "data", "pretrained_weights.npy")

# vocab size +1 for the 0 padding
config = {
    "train_x" : train_x_dir,
    "train_y" : train_y_dir,
    "test_x" : test_x_dir,
    "test_y" : test_y_dir,
    "embedding_dir": embedding_dir,
    "vocab_size" : 20275, 
    "output_size" : 3,
    "embedding_length" : 50,
    "hidden_size" : 256,
    "batch_size" : 32,
    "epoch": 1000,
    "load_epoch": 140
}
