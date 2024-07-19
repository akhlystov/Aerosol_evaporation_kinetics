import numpy as np
import matplotlib.pyplot as plt
import aerosol_kinetics as ak


def generate_vbs(n_sim, n_bins):
    # generate VBSs
    fs = np.zeros((n_sim,n_bins))
    for i in range(n_sim):
        f = np.random.random(n_bins)  # VBS
        fs[i, :] = f / f.sum()
    return fs


def calc_ratios(vbs, cs, mw_sensitivity, xi=0.5, n_sim = 2):
    # generate molecular weights
    n_sim = vbs.shape[0]
    n_bins = len(cs)
    mw0 = 100  # Mw of the most volatile bin
    mw1 = mw0 * mw_sensitivity * np.log10(cs[-1] / cs[0])  # Mw of the least volatile bin
    mw = np.linspace(mw1, mw0, n_bins)  # molecular weights
    mw_equal = np.ones_like(cs)  # equal Mw

    mw_tmp = vbs * mw  # weighted Mw with the VBS

    ratio = np.zeros((n_sim, n_bins))  # ratio of teff for constant Mw to that of variable Mw
    mw_ratio_all = np.zeros((n_sim, n_bins)) # ratio of all Mw to that of a bin

    for i in range(n_sim):
        print(f'\r Processing {i+1} of {n_sim} simulations', end='', flush=True)
        for bin in range(len(cs)):
            # calculate Mw ratios
            mw_ratio_all[i, bin] = mw_tmp[i, :].sum() / mw[bin]
            # calculate effective times
            t = ak.calc_t_eff_cont(xi, bin, cs, vbs[i], mw)
            t1 = ak.calc_t_eff_cont(xi, bin, cs, vbs[i], mw_equal)
            ratio[i, bin] = t1/t

    return ratio, mw_ratio_all


def plot_ratios(ratio, mw_ratio, num=None):
    # calculate regression:
    slope, intercept = np.polyfit(mw_ratio.flatten(), ratio.flatten(), 1)
    x = np.linspace(0.1, np.ceil(np.max(mw_ratio)), 2)
    regr_line = slope * x + intercept

    # plot data
    plt.figure(num=num)
    symbols = ['o', 's', '^', 'D', 'x', 'v', 'p', '*', 'h', '+', '1']
    symbols = symbols[::-1]
    colors = ['C'+str(x) for x in range(len(Cs))]
    for i in range(ratio.shape[1]):
        plt.scatter(mw_ratio.T[i], ratio.T[i], marker=symbols[i], color=colors[i], label=f'{int(np.log10(Cs[i]))}')

    # plot regression
    plt.plot(x, regr_line, '--k', lw=1)
    equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
    plt.text(0.65, 0.55, equation_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    font_size = 16
    plt.xlabel(r'$\overline{M} / M_i$', fontsize=font_size)
    plt.ylabel('Effective time ratio', fontsize=font_size)
    plt.legend(ncols=3, title=r'$\log_{10}(C_s)$')
    plt.subplots_adjust(bottom=0.13, top=0.9)


if __name__ == '__main__':
    Cs = np.logspace(-5, 3, 9)  # SVC, ug/m3
    Mw_sensitivity = 2 / 3  # (2/3 or 2/10) mol weight change per decade of Cs:
    N_sim = 100  # number of Monte Carlo simulations; the more, the longer it will take to calculate
    VBS = generate_vbs(N_sim, len(Cs))  # generate VBS

    print('working on 50% MFR')
    Ratio, Mw_ratio = calc_ratios(VBS, Cs, Mw_sensitivity, xi=0.5, n_sim=N_sim)
    print()
    print('working on 10% MFR')
    Ratio10, Mw_ratio10 = calc_ratios(VBS, Cs, Mw_sensitivity, xi=0.1, n_sim=N_sim)
    print()
    print('working on 90% MFR')
    Ratio90, Mw_ratio90 = calc_ratios(VBS, Cs, Mw_sensitivity, xi=0.9, n_sim=N_sim)

    #### Figure 4 is "Mw_sensitivity_50MFR.png" calculated for Mw_sensitivity = 2/3. The other two figures are for
    # MFR 10% and 90%. The sensitivity to the Mw_sensitivity can be tested by changing its value above.
    plot_ratios(Ratio, Mw_ratio, num=50)
    plt.savefig('./figures/Mw_sensitivity_50MFR.png')
    plot_ratios(Ratio10, Mw_ratio10, num=10)
    plt.savefig('./figures/Mw_sensitivity_10MFR.png')
    plot_ratios(Ratio90, Mw_ratio90, num=90)
    plt.savefig('./figures/Mw_sensitivity_90MFR.png')

    plt.show()