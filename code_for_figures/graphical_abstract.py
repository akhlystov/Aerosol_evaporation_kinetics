import numpy as np
import aerosol_kinetics as ak
import matplotlib.pyplot as plt

# t = np.logspace(-3, 8, 100)  # times to calculate xi for using the classical method
Cs = np.logspace(-5, 3, 7)  # saturation concentrations
f = np.array([1, 2, 2.5, 3, 2.5, 2, 1])  # vbs
f = f / f.sum()  # normalize the vbs
d_start = 300  # the initial diameter

x_final = np.linspace(1e-3, 0.999, 100)  # final xi values for calculations using the new method

char_t = ak.char_time(Cs, d_start)  # characteristic times

ref_bin = 3  # the reference bin

t_eff_cont_i = np.zeros((len(Cs),len(x_final)))
for i in range(len(Cs)):
    t_eff_cont_i[i, :] = np.array([ak.calc_t_eff_cont_cmw(x, i, Cs, f) for x in x_final])
    plt.semilogx(t_eff_cont_i[i, :], x_final)
plt.xlabel("Compound's effective time", fontsize=16)
plt.ylabel("Compound's MFR", fontsize=16)
plt.savefig('./figures/graph_abstract_1.png')
# # this can also be used, but does not produce as smooth a picture
# t_eff_cont_ref = np.array([ak.calc_t_eff_cont_cmw(x, ref_bin, Cs, f) for x in x_final])
# for i in range(len(Cs)):
#     a_ij = char_t[i]/char_t[ref_bin]
#     plt.semilogx(t_eff_cont_ref*a_ij, x_final**a_ij)


plt.figure()
for i in range(len(Cs)):
    plt.semilogx(t_eff_cont_i[i, :]*char_t[i], x_final)
plt.xlabel("Time", fontsize=16)
plt.ylabel("Compound's MFR", fontsize=16)
plt.savefig('./figures/graph_abstract_2.png')
plt.show()