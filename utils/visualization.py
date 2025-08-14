# visualization.py
import matplotlib.pyplot as plt

def plot_pred_vs_actual(actual, predicted, ylabel='Orders'):
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
