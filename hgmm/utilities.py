import os
from collections import Counter
import numpy as np
from scipy.stats import norm
import pandas as pd
from matplotlib import pyplot as plt
from .histogram_gmm import HistogramGaussianMixture

def prepare_xhist(x):
    t = pd.DataFrame(x)
    t.columns = [f'f{i}' for i in range(t.shape[-1])]
    t = pd.DataFrame(t.value_counts())
    t.reset_index(inplace=True)
    t.columns = t.columns[:-1].tolist() + ['Count']
    return t.values

def print_model_parameters(model_type, model):
    print()
    print('~'*15)
    print('model_type:', model_type)
    print(f'{model_type}.covariances_:\n', model.covariances_)
    print(f'{model_type}.means_:\n', model.means_)
    print(f'{model_type}.weights_:\n', model.weights_)
    print('~'*15)
    print()

def explore_optimal_componentsnumber(x, max_ncomponents=10, inf_criterion='bic', random_state=42):
    # TODO: allow to pass the keyword arguments of HistogramGaussianMixture
    ics = []
    min_ic = 0
    counter=1
    for i in range (max_ncomponents):
        hgmm = HistogramGaussianMixture(n_components=counter, 
                                        max_iter=1000,
                                        random_state=random_state, 
                                        covariance_type = 'full',
                                        init_params='kmeans')
        hgmm.fit(x)
        
        if inf_criterion == 'bic':
            ic = hgmm.bic(x)
        elif inf_criterion == 'aic':
            ic = hgmm.aic(x)
        ics.append(ic)
        if ic < min_ic or min_ic == 0:
            min_ic = ic
            opt_ic = counter
        counter = counter + 1

    # plot
    fig, ax = get_fig_ax(figsize=(6,4), nrows=1, constrained_layout=True)
    ax.plot(np.arange(1,max_ncomponents+1), ics, 'o-', lw=3, c='black', label=inf_criterion.upper(), alpha=0.7)
    ax.legend(frameon=False, fontsize=15)
    plt.xlabel('Number of components', fontsize=20)
    plt.ylabel('Information criterion', fontsize=20)
    plt.xticks(np.arange(1,max_ncomponents+1, 1))
    plt.title('Opt. components = '+str(opt_ic), fontsize=20)
    return opt_ic

#### files reading/processing
def get_filenames_in_dir(dirpath):
    for root, dirs, files in os.walk(dirpath, topdown=True):
        return files

def merge_fragcount_files(data_path, file_names):
    fraglen_count_dict = Counter({})
    for i in range(len(file_names)):
        fraglen_counts_df = pd.read_csv(os.path.join(data_path, file_names[i]), header=None, sep=' ')
        fraglen_counts_df.columns = ['frag_len', 'count']
        tmp = fraglen_counts_df.to_dict(orient='list')
        tmp = Counter(dict(zip(tmp['frag_len'], tmp['count'])))
        fraglen_count_dict.update(tmp)
    return dict(fraglen_count_dict)

def create_df_from_dict(dic, column_names):
    
    df = pd.DataFrame.from_dict({elkey:[dic[elkey]] for elkey in dic}, 
                           orient='index')
    df.reset_index(inplace=True)
    df.columns = column_names
    return df

### plotting

def get_fig_ax(figsize=(6,3), nrows=1, constrained_layout=True):
    fig, ax = plt.subplots(figsize=figsize, 
                           nrows=nrows, 
                           constrained_layout=constrained_layout)
    return fig, ax

def plot_1D_hgmm_res(hgmm, x, title='', xlabel='Fragment length', plot_separate_comp=True):
    
    mean = hgmm.means_  
    covs  = hgmm.covariances_
    weights = hgmm.weights_
    n_components = mean.shape[0]

    # plotting
    min_x = x[:,0].min()
    max_x = x[:,0].max()
    x_axis = np.arange(min_x-1, max_x+1, 1.)

    y_axis_lst = []
    y_colors = []
    for i in range(n_components):
        if hgmm.covariance_type == 'full':
            cov = covs[i,0, 0]
        elif hgmm.covariance_type == 'diag':
            cov = covs[i,0]
        elif hgmm.covariance_type == 'spherical':
            cov = covs[i]
        elif hgmm.covariance_type == 'tied':
            cov = covs[0,0]      

        y_axis = norm.pdf(x_axis, float(mean[i,0]), np.sqrt(float(cov)))*weights[i]
        y_axis_lst.append(y_axis)
        y_colors.append(f'C{i+1}')
    
    fig, ax = get_fig_ax(figsize=(9,5), nrows=1, constrained_layout=True)

    ax.hist(x[:,:-1], weights=x[:,-1:], density=True, color='grey',ec='blue', histtype='bar', bins=50, alpha=0.5)
  
    y_sum = 0.
    for i in range(n_components):
        ax.plot(x_axis, y_axis_lst[i], label=y_colors[i], lw=3, c=y_colors[i], alpha=0.8)
        y_sum += y_axis_lst[i]

    ax.plot(x_axis, y_sum, lw=3, c=f'C{len(y_colors)+1}', ls='dashed', label='Sum', alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(r"Density", fontsize=15)
    ax.set_title(title)
    # plt.subplots_adjust(wspace=0.3)
    ax.legend()

    if plot_separate_comp:
        fig, ax = get_fig_ax(figsize=(9,6), nrows=n_components)
        for i in range(n_components):
            ax[i].plot(x_axis, y_axis_lst[i], label=y_colors[i], lw=3, c=y_colors[i], alpha=0.8)
            ax[i].set_xlabel(r"Fragment length", fontsize=15)
            ax[i].set_ylabel(r"Density", fontsize=15)
            ax[i].legend()