'''
utilities related to visualization
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import seaborn as sns

def graph_hist(x):
    var = 'Age (years)'
    bins = np.arange(0, 100, 5.0)
    plt.figure(figsize=(10, 8))
    # the histogram of the data
    plt.hist(x, bins, alpha=0.8, histtype='bar', color='gold',ec='black',weights=np.zeros_like(x) + 100. / x.size)
    plt.xlabel(var)
    plt.ylabel('percentage')
    plt.xticks(bins)
    plt.show()
    fig.savefig(var + ".pdf", bbox_inches='tight')


def graph_hist_percentage(x):
    var = 'Age (years)'
    bins = np.arange(0, 100, 5.0)
    ########################################################################
    hist, bin_edges = np.histogram(x, bins,
                                   weights=np.zeros_like(x) + 100. / x.size)
    # make the histogram
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 2, 1)
    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)), hist, width=1, alpha=0.8, ec='black',
           color='gold')  # # Set the ticks to the middle of the bars
    ax.set_xticks([0.5 + i for i, j in enumerate(hist)])
    # Set the xticklabels to a string that tells us what the bin edges were
    labels =['{}'.format(int(bins[i+1])) for i,j in enumerate(hist)]
    labels.insert(0,'0')
    ax.set_xticklabels(labels)
    plt.xlabel(var)
    plt.ylabel('percentage')
    #now the count chart
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 2, 1)
    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)), hist, width=1, alpha=0.8, ec='black',
           color='gold')  # # Set the ticks to the middle of the bars
    ax.set_xticks([0.5 + i for i, j in enumerate(hist)])
    # Set the xticklabels to a string that tells us what the bin edges were
    labels =['{}'.format(int(bins[i+1])) for i,j in enumerate(hist)]
    labels.insert(0,'0')
    ax.set_xticklabels(labels)
    plt.xlabel(var)
    plt.ylabel('percentage')


def graph_pieplot(labels, percentage):
    '''
    plot a piechart given labels and their percentage
    :param labels:
    :param percentage:
    :return:
    '''
    # Data to plot
    labels = labels
    sizes = percentage
    colors = ['gold', 'yellowgreen', 'lightcoral', 'blue', 'lightskyblue', 'green','red']
    explode = (0, 0.1, 0, 0, 0, 0)  # explode 1st slice
    # Plot
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()


def graph_barplot(labels, percentage):
    '''
    plot a barplot given labels and their percentage
    :param labels:
    :param percentage:
    :return:
    '''
    labels = labels
    missing = percentage
    ind = [x for x, _ in enumerate(labels)]
    plt.figure(figsize=(10, 8))
    plt.bar(ind, missing, width=0.8, label='missing', color='gold')
    plt.xticks(ind, labels)
    plt.ylabel("percentage")
    plt.show()


def graph_barplot_stacked():
    '''
    plot a barplot with stacked data in a single bar
    :return:
    '''
    labels = ['missing', '<25', '25-34', '35-44', '45-54', '55-64', '65+']
    missing = np.array([0.000095, 0.024830, 0.028665, 0.029477, 0.031918, 0.037073, 0.026699])
    man = np.array([0.000147, 0.036311, 0.038684, 0.044761, 0.051269, 0.059542, 0.054259])
    women = np.array([0.004035, 0.032935, 0.035351, 0.041778, 0.048437, 0.056236, 0.048091])
    ind = [x for x, _ in enumerate(labels)]
    plt.figure(figsize=(10, 8))
    plt.bar(ind, women, width=0.8, label='women', color='gold', bottom = man + missing)
    plt.bar(ind, man, width=0.8, label='man', color='silver', bottom=missing)
    plt.bar(ind, missing, width=0.8, label='missing', color='#CD853F')
    plt.xticks(ind, labels)
    plt.ylabel("percentage")
    plt.legend(loc="upper left")
    plt.title("demo")
    plt.show()


def graph_box_violinplot(x):
    '''
    make box and violinplot to check the normality of the data
    :param x:
    :return:
    '''
    #x = df.select(var).toPandas()
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax = sns.boxplot(data=x)
    ax = fig.add_subplot(1, 2, 2)
    ax = sns.violinplot(data=x)


def graph_scatterplot():
    sns.set(style="ticks")
    df = sns.load_dataset("iris")
    sns.pairplot(df, hue="species")
    plt.show()