import numpy as np 
from matplotlib import pyplot as plt
import os
import sys

def plot_vec(yvec, fname, xlab, ylab, title):
    n = len(yvec)
    xvec = np.arange(1,n+1)
    plt.figure()
    plt.plot(xvec, yvec)    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(fname)

def plot_all_in_logfolder(experiment_folder):
    root = experiment_folder
    
    cdloss = np.load(root + 'condloss.npy')
    mgloss = np.load(root + 'margloss.npy')
    pcorrs = np.load(root + 'pears_corrs.npy')
    scorrs = np.load(root + 'spear_corrs.npy')
    kldivs = np.load(root + 'kldivs.npy')

    plot_vec(cdloss, root + 'cond_loss.png', 'steps', 'loss', 'Conditional loss vs steps')
    plot_vec(mgloss, root + 'marg_loss.png', 'steps', 'loss', 'Marginal loss vs steps')
    plot_vec(pcorrs, root + 'pearcorrs.png', 'steps', 'correlation coef', 'Pearson Correlation vs steps')
    plot_vec(scorrs, root + 'spearcorrs.png', 'steps', 'correlation coef', 'Spearman Correlation vs steps')
    plot_vec(kldivs, root + 'kldivs.png', 'steps', 'KL value', 'Avg KL divergence vs steps')


# logfolder = '../log/'
logfolder = '/home/ninad/Desktop/Link-to-sem4/dsis/prob-emb/test_code/log/'
# for root, subdir, _ in os.walk(logfolder):
#     print(subdir)

for exp_dir in os.listdir(logfolder):
    exp_folder = logfolder + exp_dir + os.sep
    plot_all_in_logfolder(exp_folder)