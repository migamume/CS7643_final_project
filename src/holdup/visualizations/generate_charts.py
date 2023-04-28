import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import pickle 



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


def hidden_nodes_plot(data, arrow_x, filepath=None, line=2, stage=None):
    x_ = list(data.keys())
    x = [int(x) for x in x_]
    y = [np.mean(value) for value in data.values()]
    low = [min(value) for value in data.values()]
    high = [max(value) for value in data.values()]

    plt.figure()

    y_bars = np.array([np.array(y) - np.array(low), np.array(high) - np.array(y)])
    plt.errorbar(x, y, yerr=y_bars, fmt='o', markersize=4, capsize=3, ecolor='black', elinewidth=1, color='black', label="Test accuracy % (Average, Hi/Low)")

    y_arrow = min(data[str(arrow_x)])
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
    plt.title(f"Hidden nodes vs Test Accuracy for {stage} Stage")
    plt.legend()
    plt.grid()
    
    if filepath:
        plt.savefig(filepath)
    else:
        print("WARNING: No filepath set for hidden nodes plot. Not saving")
    

def epochs_plot(data, arrow_x, filepath=None, line=2, stage=None):
    x_ = list(data.keys())
    x = [int(x) for x in x_]
    y = [np.mean(value) for value in data.values()]
    low = [min(value) for value in data.values()]
    high = [max(value) for value in data.values()]

    plt.figure()

    y_bars = np.array([np.array(y) - np.array(low), np.array(high) - np.array(y)])
    plt.errorbar(x, y, yerr=y_bars, fmt='o', markersize=4, capsize=3, ecolor='black', elinewidth=1, color='black', label="Test accuracy % (Average, Hi/Low)")

    y_arrow = min(data[str(arrow_x)])
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
    plt.title(f"Epochs vs Test Accuracy for {stage} Stage")
    plt.legend()
    plt.grid()
    
    if filepath:
        plt.savefig(filepath)
    else:
        print("WARNING: No filepath set for hidden nodes plot. Not saving")
        
if __name__ == "__main__":

    with open('flop_nd.pickle', 'rb') as flop_nd:
        flop_nd = pickle.load(flop_nd)
        
    with open('turn_nd.pickle', 'rb') as turn_nd:
        turn_nd = pickle.load(turn_nd)
        
    with open('river_nd.pickle', 'rb') as river_nd:
        river_nd = pickle.load(river_nd)
        
    with open('flop_ed.pickle', 'rb') as flop_ed:
        flop_ed = pickle.load(flop_ed)
        
    with open('turn_ed.pickle', 'rb') as turn_ed:
        turn_ed = pickle.load(turn_ed)
        
    with open('river_ed.pickle', 'rb') as river_ed:
        river_ed = pickle.load(river_ed)

    def get_arrow(exp):
        arrow = (None, -np.inf)
        for item in exp:
            avg = np.mean(exp[item])
            # print(item, avg)
            if avg > arrow[1]:
                arrow = (int(item), avg)
        return int(arrow[0])
            
    # print(get_arrow(flop_nd))
    # print(get_arrow(turn_nd))
    hidden_nodes_plot(flop_nd, arrow_x=get_arrow(flop_nd), filepath="holdup/visualizations/charts/flop_nodes",line=2, stage='Flop')
    hidden_nodes_plot(turn_nd, arrow_x=get_arrow(turn_nd), filepath="holdup/visualizations/charts/turn_nodes",line=2, stage='Turn')
    hidden_nodes_plot(river_nd, arrow_x=get_arrow(river_nd), filepath="holdup/visualizations/charts/river_nodes",line=2, stage='River')

    epochs_plot(flop_ed, arrow_x=get_arrow(flop_ed), filepath="holdup/visualizations/charts/flop_epochs",line=2, stage='Flop')
    epochs_plot(turn_ed, arrow_x=get_arrow(turn_ed), filepath="holdup/visualizations/charts/turn_epochs",line=2, stage='Turn')
    epochs_plot(river_ed, arrow_x=get_arrow(river_ed), filepath="holdup/visualizations/charts/river_epochs",line=2, stage='River')
    plt.show()
# epochs_plot(dummy_epoch_data, arrow_x=40, filepath="holdup/visualizations/charts/epoch_accuracy",line=2)

