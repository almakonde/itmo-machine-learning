#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import mpld3 as mpld3
import pandas as pd


def plot_clusters_mpl(df: pd.DataFrame, cluster_names, cluster_colors):
    # group by cluster
    groups = df.groupby('label')

    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  # show legend with only 1 point

    # add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

    plt.show()  # show the plot

    # uncomment the below to save the plot if need be
    # plt.savefig('clusters_small_noaxes.png', dpi=200)


# define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """

    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}


def plot_clusters_d3(df: pd.DataFrame, cluster_names, cluster_colors):
    # group by cluster
    groups = df.groupby('label')

    # define custom css to format the font and to remove the axis labeling
    css = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }

    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }

    svg.mpld3-figure {
    margin-left: -200px;}
    """

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))  # set plot size
    ax.margins(0.03)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    # note that I use the cluster_name and cluster_color dicts with the 'name' lookup
    # to return the appropriate color/label
    for name, group in groups:
        points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18,
                         label=cluster_names[name], mec='none',
                         color=cluster_colors[name])
        ax.set_aspect('auto')
        labels = [i for i in group.title]

        # set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                                 voffset=10, hoffset=10, css=css)
        # connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())

        # set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        # set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    ax.legend(numpoints=1)  # show legend with only one dot

    # mpld3.display()  # show the plot
    mpld3.show()

    # uncomment the below to export to html
    # html = mpld3.fig_to_html(fig)
    # print(html)
