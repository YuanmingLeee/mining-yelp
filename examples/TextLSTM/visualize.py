from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from os.path import join
import pandas as pd
import seaborn as sn

def visualize_loss_curve(history_dir):
    data = np.genfromtxt(history_dir, delimiter="\n")
    x = np.arange(0, data.shape[0])
    
    # fig = plt.figure()
    # ax = plt.axese()
    plt.plot(x, data)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, save_dir, title="Confusion Matrix", normalize=False, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)

    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #plt.savefig(join(save_dir, "confusion_matrix.png"))
    plt.show()

def plot_cm(matrix):
    df_cm = pd.DataFrame(matrix, index = ["0", "1"],
                  columns = ["0", "1"])
    plt.figure(figsize = (7,7))
    sn.heatmap(df_cm, annot=True, fmt='.5f', cmap = "RdBu_r") #cmap=sn.diverging_palette(220, 20, as_cmap=True))
    plt.title("Confusion Matrix")
    plt.xlabel("prediction")
    plt.ylabel("ground truth")
    plt.show()

if __name__ == "__main__":
    cm = [[0.79502046, 0.20497954], [0.30339173, 0.69660827]]
    plot_cm(cm)