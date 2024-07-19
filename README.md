[](https://doi.org/10.5281/zenodo.12786252)

# Aerosol_evaporation_kinetics
This repository contains Python code used in "On evaporation kinetics of multicomponent aerosols: Characteristic times and implications for volatility measurements" by Andrey Khlystov published in Aerosol Science and Technology.

The main library is aerosol_kinetics.py that contains functions for both the new and the traditional approach to modeling evaporation of a multicomponent particle in vapor-free conditions. The folder "code_for_figures" contains the code for generating figures in the paper (as well as a few more). The folder "figures" contains the figures produced by that code. 

## aerosol_kinetics.py

The main functions that one will need to model aerosol evaporation are:

calc_xi_trans: the traditional way to calculate the mass fraction remaining (MFR) of individual compounds as a function of time in the transition regime by numerically integrating a set of ODEs.   

calc_xi_cont: the traditional way to calculate the MFR of individual compounds as a function of time in the continuum regime by numerically integrating a set of ODEs. 

calc_t_eff_trans: the new, faster approach to calculate the effective time needed for a compound to reach a certain MFR in the transition regime.

calc_t_eff_cont: as the previous, but for the continuum regime.

calc_t_eff_cont_cmw: as calc_t_eff_cont but for a mixture in which all compounds have the same molecular weight.

calc_t_eff_appr_num: calculate the effective time needed for a compound to reach a certain MFR  in the continuum regime using numerical integration of the approximation formula.

calc_t_eff_appr: calculate the effective time needed for a compound to reach a certain MFR  in the continuum regime using the analytical solution for the approximation formula (faster).

calc_t_eff_fm: calculate the effective time needed for a compound to reach a certain MFR in the free molecular regime.

calc_t_eff_fm_appr_num: calculate the effective time needed for a compound to reach a certain MFR  in the free molecular regime using numerical integration of the approximation formula.

calc_t_eff_fm_appr: calculate the effective time needed for a compound to reach a certain MFR  in the free molecular regime using the analytical solution for the approximation formula (faster)

## code_for_figures

approximation_accuracy.py: produces figure 1 of the paper as well as similar plots for the least and most volatile bins.

VBS_effect.py: produces figures 2 and 3 of the paper

Mw_effect.py: produces figure 4 of the paper as well as similar plots for 10% and 90% MFR.

## figures

Error_midle_bin.png: Figure 1.

t50_sensitivity.png: Figure 2.

effective_time_violine.png: Figure 3.

Mw_sensitivity_50MFR.png: Figure 4.

There are a few other figures (for the graphical abstract and variations of Fig.1 and 4 for other input parameters).
