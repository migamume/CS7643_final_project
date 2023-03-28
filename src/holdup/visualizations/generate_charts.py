import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

dummy_nodes_data = {
    10: (93, 87, 95),
    20: (97, 98, 99),
    30: (95, 90, 99),
    40: (95, 90, 99),
    50: (95, 90, 99),
    60: (95, 90, 99),
    70: (98, 93, 99),
    80: (98, 93, 99),
    90: (98, 93, 99),
    100: (98, 93, 99),
}

dummy_epoch_data = {
    10: (59, 60, 65),
    20: (62, 68, 78),
    30: (88, 96, 93),
    40: (97, 98, 99),
    50: (98, 98, 99),
    60: (98, 98, 99),
}

def hidden_nodes_plot(data, arrow_x, filepath=None, line=2):
    x = list(data.keys())
    y = [np.mean(value) for value in data.values()]
    low = [min(value) for value in data.values()]
    high = [max(value) for value in data.values()]

    plt.figure()

    y_bars = np.array([np.array(y) - np.array(low), np.array(high) - np.array(y)])
    plt.errorbar(x, y, yerr=y_bars, fmt='o', markersize=4, capsize=3, ecolor='black', elinewidth=1, color='black', label="Test accuracy % (Average, Hi/Low)")

    y_arrow = min(data[arrow_x])
    plt.arrow(arrow_x, y_arrow - 15, 0, 10, head_width=2, head_length=2, linewidth=2, color='black', zorder=3)
    
    if line is not None:
        x_smooth = np.linspace(min(x), max(x), 300)
        spl = make_interp_spline(x, y, k=line)
        y_smooth = spl(x_smooth)
        plt.plot(x_smooth, y_smooth, linestyle='-', linewidth=1, color='black', alpha=0.6)

    plt.xticks([0] + x)
    plt.yticks(np.arange(0, 101, 10))

    plt.xlabel("Number of hidden nodes")
    plt.ylabel("Test accuracy %")
    plt.title("Hidden nodes vs Test accuracy")
    plt.legend()
    plt.grid()
    
    if filepath:
        plt.savefig(filepath)
    else:
        print("WARNING: No filepath set for hidden nodes plot. Not saving")

    

def epochs_plot(data, arrow_x, filepath=None, line=2):
    x = list(data.keys())
    y = [np.mean(value) for value in data.values()]
    low = [min(value) for value in data.values()]
    high = [max(value) for value in data.values()]

    plt.figure()

    y_bars = np.array([np.array(y) - np.array(low), np.array(high) - np.array(y)])
    plt.errorbar(x, y, yerr=y_bars, fmt='o', markersize=4, capsize=3, ecolor='black', elinewidth=1, color='black', label="Test accuracy % (Average, Hi/Low)")

    y_arrow = min(data[arrow_x])
    plt.arrow(arrow_x, y_arrow - 15, 0, 10, head_width=1, head_length=2, linewidth=2, color='black', zorder=3)  

    if line is not None:
        x_smooth = np.linspace(min(x), max(x), 300)
        spl = make_interp_spline(x, y, k=line)
        y_smooth = spl(x_smooth)
        plt.plot(x_smooth, y_smooth, linestyle='-', linewidth=1, color='black', alpha=0.6)
    
    plt.xticks([0] + x)
    plt.yticks(np.arange(0, 101, 10))

    plt.xlabel("Number of epochs")
    plt.ylabel("Test accuracy %")
    plt.title("Epochs vs Test accuracy")
    plt.legend()
    plt.grid()
    
    if filepath:
        plt.savefig(filepath)
    else:
        print("WARNING: No filepath set for hidden nodes plot. Not saving")



hidden_nodes_plot(dummy_nodes_data, arrow_x=20, filepath="charts/hidden_nodes_accuracy",line=2)
epochs_plot(dummy_epoch_data, arrow_x=40, filepath="charts/epoch_accuracy",line=2)

