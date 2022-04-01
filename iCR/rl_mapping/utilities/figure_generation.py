import numpy as np
import matplotlib.pyplot as plt;  # plt.ion()


def active_latex_text():
    # activate latex text rendering
    plt.rc('text', usetex=True)
    plt.rc('axes', linewidth=2)
    plt.rc('font', weight='bold', size=20)
    plt.rcParams['font.family'] = 'serif'

    # plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # plt.rc('font',**{'family':'serif','serif':['Times']})
    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']


def gen_data():
    dev = np.pi  # domain of the function
    n = 100  # number of samples
    x = np.sort(2 * dev * np.random.rand(n) - dev)  # training points
    std = 0.1  # noise standard deviation
    y = np.sin(x) + np.random.normal(0, std, n)  # training labels
    return x, y


if __name__ == '__main__':
    plt.close("all")

    info = np.load('map_6_entropy.npy', allow_pickle=True)
    info_enhanced = np.load('map_6_entropy_enhanced.npy', allow_pickle=True)
    info_random = np.load('map_6_entropy_random.npy', allow_pickle=True)
    info_frontier = np.load('map_6_entropy_frontier.npy', allow_pickle=True)

    active_latex_text()
    f, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(info.shape[0]), info, 'b-', linewidth=2, label=r'\textbf{iCR}')
    ax.plot(np.arange(info_enhanced.shape[0]), info_enhanced, 'g-', linewidth=2, label=r'\textbf{iCR + frontiers}')
    ax.plot(np.arange(info_random.shape[0]), info_random, 'r-', linewidth=2, label=r'\textbf{random}')
    ax.plot(np.arange(info_frontier.shape[0]), info_frontier, 'c-', linewidth=2, label=r'\textbf{frontiers}')
    ax.legend(loc="lower right")

    ax.tick_params(axis='x')
    ax.set_xlabel(r'\textbf{Time Step [$k$]}')

    ax.tick_params(axis='y')
    ax.set_ylabel(r'$\log |Y_k|$')
    ax.ticklabel_format(style='sci', scilimits=(0, 3))
    f.canvas.draw()

    # plt.show()

plt.savefig("map_6_perf.pdf", format='pdf', bbox_inches='tight')
