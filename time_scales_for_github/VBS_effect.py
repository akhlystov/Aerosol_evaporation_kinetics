# this is to use integrals to calculate t0.5 and t90 for different fi and bi

import numpy as np
import matplotlib.pyplot as plt
import aerosol_kinetics as ak
from scipy.integrate import quad


def plot_one(data, legend_loc):
    fig, ax = plt.subplots()
    ax.axhline(0, linewidth=1, color='k', linestyle=':')
    vp = ax.violinplot(data, showextrema=False)
    # Adding mean, median, 25th, and 75th quantiles for each violin
    colors = ['red', 'blue', 'green']
    labels = ['Mean', 'Median', '25th & 75th Quantiles']
    for i in range(data.shape[1]):
        mean = np.mean(data[:, i])
        median = np.median(data[:, i])
        q25, q75 = np.percentile(data[:, i], [25, 75])
        ax.scatter(i + 1, mean, color=colors[0], label=labels[0] if i == 0 else "")
        ax.scatter(i + 1, median, marker='x', color=colors[1], label=labels[1] if i == 0 else "")
        ax.hlines([q25, q75], i + 1 - 0.1, i + 1 + 0.1, color=colors[2], label=labels[2] if i == 0 else "")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['90%', '50%', '10%'])
    ax.legend(loc=legend_loc)
    fs = 14
    ax.set_ylabel('Effective time', fontsize=fs)
    ax.set_xlabel('Mass fraction remaining', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim([1e-2,10])
    return fig, ax

def f_cont(xi, bi):
    return (xi + bi)**(2/3) / xi

def calc_t_eff(x_final, points=250):
    f_i = np.linspace(0, 1, points+1)
    n = len(f_i)
    t50 = np.zeros((n, n)) * np.nan
    for i in range(n):
        for j in range(n):
            if f_i[i] > 0: # to avoid divisions by zero (empty bins are not relevant anyway)
                b = f_i[j]/f_i[i]
                # only the following b_s are allowed
                if b < 1 / f_i[i] - 1:
                    res = quad(f_cont, x_final, 1, args=(b))
                    t50[i, j] = res[0] * f_i[i]**(2/3)
    x, y = np.meshgrid(f_i, f_i)  # Create a meshgrid for X and Y coordinates
    return x, y, t50.T


def remove_nans(x_coord, y_coord, z_coord):
    x_flat = x_coord.flatten()
    y_flat = y_coord.flatten()
    z_flat = z_coord.flatten()
    mask = ~np.isnan(z_flat)
    x_filtered = x_flat[mask]
    y_filtered = y_flat[mask]
    z_filtered = z_flat[mask]
    return x_filtered, y_filtered, z_filtered


def calc_stats(data):
    median = np.median(data)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)
    print(f"Min: {np.min(data)}")
    print(f"Mean: {np.mean(data)}")
    print(f"Median: {median}")
    print(f"25th percentile: {percentile_25}")
    print(f"75th percentile: {percentile_75}")
    print(f"Max: {np.max(data)}")
    print(f"STD: {np.std(data)}")


# One can play with the MFR to see how it changes Figure 2 (the saved figure file name will be automatically updated)
fraction = 0.5  # MFR, can be changed to other values

X, Y, t50 = calc_t_eff(fraction)
x, y, z50 = remove_nans(X, Y, t50)
X10, Y10, t10 = calc_t_eff(0.1)
x10, y10, z10 = remove_nans(X10, Y10, t10)
X90, Y90, t90 = calc_t_eff(0.9)
x90, y90, z90 = remove_nans(X90, Y90, t90)
calc_stats(z50)

font_size = 14

### Figure 2:
plt.figure()
cp = plt.contourf(X, Y, t50)
tmp = np.linspace(1e-5, 1, 251)
plt.plot(tmp,1-tmp,'k',lw=0.5)
plt.ylim([np.min(X), 1])
cbar = plt.colorbar(cp)
cbar.set_label('Effective time', rotation=90, labelpad=15, fontsize=12)
plt.xlabel('Mass fraction in the bin', fontsize=font_size)
plt.ylabel('Mass fraction of less volatile bins', fontsize=font_size)
plt.savefig('./figures/t'+str(int(100*fraction))+'_sensitivity.png')

### Figure 3:
dat = np.array([z90, z50, z10])
fig, ax = plot_one(dat.T, "upper left")
fig.savefig('./figures/effective_time_violine.png')

plt.show()
