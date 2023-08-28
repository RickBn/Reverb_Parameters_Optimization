import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


def mean_absorb_graph(dataFrame, X, planar_projection):
    bins = ["0.0 to 0.1", "0.1 to 0.2", "0.2 to 0.3", "0.3 to 0.4", "0.4 to 0.5",
                "0.5 to 0.6", "0.6 to 0.7", "0.7 to 0.8", "0.8 to 0.9", "0.9 to 1.0"]

    colors = ListedColormap(["navy", "turquoise", "darkorange", "green", "red", "purple", "yellow", "lightgreen", "brown", "grey"])

    dataFrame['mean_absorption'] = dataFrame.mean(axis=1)
    
    # def add_labels(row):
    #     return bins[(int)(np.floor(row['mean_absorption'] * 10))]
    def add_colors(row):
        return (int)(np.floor(row['mean_absorption'] * 10))

    #df['labels'] = df.apply(add_labels, axis = 1)
    dataFrame['colors'] = dataFrame.apply(add_colors, axis = 1)

    #build graph
    fig,ax = plt.subplots()
    sc = plt.scatter(planar_projection.transpose()[0], planar_projection.transpose()[1], c=dataFrame['colors'].array, cmap=colors)
    #plt.legend(handles=sc.legend_elements()[0], labels=bins, loc="lower left", title="Mean_absorption")

    #values on mousehover
    annot = ax.annotate("", xy=(0,0), xytext=(10,10),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)


    def update_annot(ind):
        
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(" ".join([np.array2string(X.iloc[n].array) for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)
        

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


    plt.show()