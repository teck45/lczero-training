import matplotlib.pyplot as plt
import pickle
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

if __name__ == "__main__":
    with open('/home/privateclient/spsa/lc0/nets/t1d.p', 'rb') as f:
        data = pickle.load(f)
        elo = data['elo']
        x = list(elo.keys())
        y = list(elo.values())
        y = moving_average(y, 2)
        print(x, y)
        plt.plot(x, y)
        plt.show()