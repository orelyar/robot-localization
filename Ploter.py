"""
a module for plotting configurations
"""


def config_plot(plt, lim):
    """

    :param plt: the plt object to configure with
    :param lim: the limits of the plot
    :return: a plt object configured with the configurations needed
    """
    plt.xlim((0, lim))
    plt.ylim((0, lim))
    plt.xticks([x for x in range(0, lim, 10)])
    plt.yticks([x for x in range(0, lim, 10)])
    plt.xlabel("location")
    plt.ylabel("y")
    plt.title("robot world")
    return plt
