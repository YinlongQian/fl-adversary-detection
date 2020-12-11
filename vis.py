import matplotlib.pyplot as plt


def display_detection_stats(line_xs, line_ys, line_labels, N_round):
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    for idx in range(len(line_xs)):
        plt.plot(line_xs[idx], line_ys[idx], colors[idx%6], label=line_labels[idx], linewidth=2)
    
    plt.xlabel("Communication Round")
    plt.ylabel("Detection Rate")
    
    plt.xlim(0, N_round)
    plt.ylim(0,1)
    
    plt.legend()
    plt.show() 


def display_handling_stats(line_xs, line_ys, line_labels, N_round, cluster_round):
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    for idx in range(len(line_xs)):
        plt.plot(line_xs[idx], line_ys[idx], colors[idx%6], label=line_labels[idx], linewidth=2)
    
    # mark round at which clustering and adversary handling processed
    plt.axvline(x=cluster_round, linewidth=2, color='k')
    
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    
    plt.xlim(0, N_round)
    plt.ylim(0,1)
    
    plt.legend()
    plt.show()