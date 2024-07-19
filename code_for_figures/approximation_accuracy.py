
import numpy as np
import matplotlib.pyplot as plt
import aerosol_kinetics as ak


def plot_one(bin_ind, fs, cs, x_final, legend_loc):
    n_sim, n_bins = fs.shape
    t_eff_exact = np.zeros((n_sim, n_bins))
    t_eff_appr = np.zeros((n_sim, n_bins))

    #ci = cs[bin_ind]

    for i in range(n_sim):
        if i % 10 == 0:
            print(i)
        f = fs[i, :]
        for j in range(len(x_final)):
            t_eff_exact[i, j] = ak.calc_t_eff_cont_cmw(x_final[j], bin_ind, cs, f)
            t_eff_appr[i, j] = ak.calc_t_eff_appr(x_final[j], bin_ind, f)

    data = (1 - t_eff_appr / t_eff_exact) * 100  # error in percent

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
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(x_final)
    ax.legend(loc=legend_loc)
    fs = 14
    ax.set_xlabel('Final MFR', fontsize=fs)
    ax.set_ylabel('Error in simplified caclulations, %', fontsize=fs)

    return fig, ax


Cs = np.logspace(-5,5, 11)  # ug/m3
X_final = np.array([0.1, 0.25, 0.5, 0.75, 0.9])  # final MFR

N_sim = 100  # the paper used 1000, but it takes more time, of course
# generate random VBS
Fs = np.zeros((N_sim, len(Cs)))
for i in range(N_sim):
    f = np.random.random(len(Cs))  # VBS
    f = f / f.sum()
    Fs[i, :] = f

#### Figure 1 is "Error_midle_bin.png". For the most and least volatile bins see the corresponding figures produced.
# Change the indexes of the bins (the 1st parameter of plot_one) to calculate errors for those bins.
fig, ax = plot_one(5, Fs, Cs, X_final, "upper left")
fig.savefig('./figures/Error_midle_bin.png')
fig, ax = plot_one(-1, Fs, Cs, X_final, "lower right")
fig.savefig('./figures/Error_last_bin.png')
fig, ax = plot_one(0, Fs, Cs, X_final, "upper left")
fig.savefig('./figures/Error_first_bin.png')


plt.show()