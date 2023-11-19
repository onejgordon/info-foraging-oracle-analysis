import os
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import linregress


from trial import Trial



def bars(x='cond_oracle_acc', y='oracle_requested', df=None, figure_attr="map_kind", figure_vals=["random", "spatial"], 
         ylabel=None, xlabel=None, figure_labels=None, hue=None, xticklabels=None, palette=None, suptitle=None, figsize=None, save=None):
    fig, axs = plt.subplots(1, len(figure_vals), dpi=300, figsize=figsize)
    COND_ORDERS = {
        'cond_oracle_acc': ["I", "A"],
        'cond_best_strat': ["OF", "NO"],        
        'cond_map_size': ["S", "L"],
        'cond_gap_size': ["S", "L"],
        'map_kind': ["random", "spatial"],
        'correct_strat': [0, 1],
        'retry_response': [0, 1],
        'oracle_requested': [0, 1],
        'ius_type': ["low", "high"]
    }
    order = COND_ORDERS[x]
    hue_order = {}
    if hue:
        hue_order = COND_ORDERS[hue]
    for i, val in enumerate(figure_vals):
        sns.barplot(data=df[df[figure_attr] == val], order=order, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette, ax=axs[i])
        if figure_labels is not None:
            figlabel = figure_labels[i]
        else:
            figlabel = "%s: %s" % (figure_attr, val)
        axs[i].set_title(figlabel)
    
    ymax = {
        'performance': 1.1,
        'oracle_requested': 1.0,
        'n_moves': 50
    }.get(y, 1)
    for ax in axs:
        ax.set_ylim((0, ymax))
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        if xticklabels:
            ax.set_xticklabels(xticklabels)
    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()
    if save:
        plt.savefig("./figures/%s.png" % save)
    plt.show()
    
def plot_linreg(X, Y, ax, text_yoffset=5, text_xpct=0.5, show_text=True):
    # Lin reg
    m, b, r, p, _ = linregress(X, Y)
    xaxis = np.arange(X.min(), X.max())
    sig = p <= 0.05
    alpha = 1.0 if sig else 0.4
    ax.plot(xaxis, m*xaxis+b, color='red', alpha=alpha)
    text_x = X.min() + (X.max() - X.min()) * text_xpct
    text = "r=%.2f p=%.3f" % (r, p)
    if show_text:
        ax.text(text_x, text_x*m+b - text_yoffset, text, color='red', fontsize=6, alpha=alpha)
    else:
        print(text)
    