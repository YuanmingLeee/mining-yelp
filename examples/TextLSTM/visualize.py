from matplotlib import pyplot as plt
import numpy as np

def visualize_loss_curve():
    data = np.genfromtxt("history.txt", delimiter="\n")
    x = np.arange(0, data.shape[0])
    
    # fig = plt.figure()
    # ax = plt.axese()
    plt.plot(x, data)
    plt.show()


if __name__ == "__main__":
    visualize_loss_curve()