from .ethanol_pbe_util import ethanol_pbe_energy_mean, ethanol_pbe_coeff_mean, ethanol_pbe_reparams, ethanol_pbe_energy_std, ethanol_pbe_grad_mean, ethanol_pbe_delta_coeff_mean, ethanol_pbe_delta_coeff_std, ethanol_pbe_linear_coeffs
from .qmugs_pbe_util import qmugs_pbe_energy_mean, qmugs_pbe_coeff_mean, qmugs_pbe_reparams, qmugs_pbe_energy_std, qmugs_pbe_grad_mean, qmugs_pbe_delta_coeff_mean, qmugs_pbe_delta_coeff_std, qmugs_pbe_linear_coeffs
from .chignolin_filter500_pbe_util import chignolin_filter500_pbe_energy_mean, chignolin_filter500_pbe_coeff_mean, chignolin_filter500_pbe_reparams, chignolin_filter500_pbe_energy_std, chignolin_filter500_pbe_grad_mean, chignolin_filter500_pbe_delta_coeff_mean, chignolin_filter500_pbe_delta_coeff_std, chignolin_filter500_pbe_linear_coeffs

# 477 dim analytical orbital integrals for basis set retb
orbital_integrals = [0.2984739472532342, 0.5934189838751444, 1.179821869426472, 2.345694494785236, 4.663655425831783, 9.27217162304913, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.009288556467748859, 0.018467292678251337, 0.0367162432664684, 0.07299838385028164, 0.14513369480857588, 0.2885514480976771, 0.5736913010385687, 1.14059974766065, 2.2677139813839986, 4.508616376526205, 8.963926578727637, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.815332931017734e-16, 5.815332931017734e-16, 0.0, 0.0, 0.0, 0.0, -9.19484870709944e-16, 9.19484870709944e-16, 0.0, 1.1561910145183755e-15, -1.1561910145183755e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -2.9649399070918274e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -1.8751926463876392e-31, 0.0, 0.0, 0.0, 0.0, -1.4912858224713124e-31, 0.0, -0.0, 0.0, 9.860761315262648e-32, -0.0, 9.860761315262648e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 0.0, 8.881784197001252e-16, 0.0, 0.0, 0.0, 4.3790577010150533e-47, 1.7763568394002505e-15, 0.0, 0.0, 0.0, 0.0, 0.007825504714384476, 0.01555848709294633, 0.03093302342230045, 0.061500320200052846, 0.12227351116225742, 0.24310136082728642, 0.48332849097344466, 0.9609425031258255, 1.9105236119102722, 3.7984587629263507, 7.552007671460073, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.4737025140300107e-16, 6.947405028060021e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.684256285075024e-16, 1.7368512570150047e-15, 0.0, 1.0919874164787477e-15, 2.1839748329574953e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.927224689253172e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.9708898757012685e-31, 0.0, 0.0, 0.0, 0.0, 3.1579324853026515e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 0.0, 0.0, -8.881784197001252e-16, 0.0, 0.0, 0.0, 8.881784197001252e-16, 0.006528035680193452, 0.012978889232011888, 0.0258043267453223, 0.05130356434023891, 0.10200055750302266, 0.20279514425018574, 0.40319260539566104, 0.8016181928162331, 1.593758710984729, 3.1686741289091525, 6.299884459307203, 0.0, 0.0, 0.0, 0.0, -2.6005448098502363e-16, 2.6005448098502363e-16, 0.0, 3.2700119792941394e-16, -3.2700119792941394e-16, 0.0, 4.111822378228114e-16, -4.111822378228114e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0279555945570284e-15, 1.0279555945570284e-15, -0.0, -6.6301924946229735e-31, 0.0, 0.0, 0.0, 0.0, -5.272798017064641e-31, 0.0, -0.0, 0.0, 0.0, -4.193301921672457e-31, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -1.6773207686689826e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.440892098500626e-16, 0.0, 0.0, 0.0, 0.0, 0.005281561679856289, 0.0105006785154186, 0.020877205638763656, 0.041507576357397886, 0.08252440124775454, 0.16407310180341564, 0.3262063381056821, 0.6485558805843419, 1.2894437694961645, 2.563642215678369, 5.096974032901284, 1.927107543796966e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 3.04702456724555e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 4.817768859492415e-16, 0.0, 0.0, -6.058023620411189e-16, 0.0, 0.0, -7.617561418113875e-16, 0.0, 0.0, 9.578576379799339e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -1.9721522630525295e-31, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.881784197001252e-16]

atomic_linear_coeffs = {}
atomic_linear_coeffs.update(qmugs_pbe_linear_coeffs)
atomic_linear_coeffs.update(chignolin_filter500_pbe_linear_coeffs)
atomic_linear_coeffs.update(ethanol_pbe_linear_coeffs)

energy_mean = {}
energy_mean.update(qmugs_pbe_energy_mean)
energy_mean.update(chignolin_filter500_pbe_energy_mean)
energy_mean.update(ethanol_pbe_energy_mean)

energy_std = {}
energy_std.update(qmugs_pbe_energy_std)
energy_std.update(chignolin_filter500_pbe_energy_std)
energy_std.update(ethanol_pbe_energy_std)

coeff_mean = {}
coeff_mean.update(qmugs_pbe_coeff_mean)
coeff_mean.update(chignolin_filter500_pbe_coeff_mean)
coeff_mean.update(ethanol_pbe_coeff_mean)

grad_mean = {}
grad_mean.update(qmugs_pbe_grad_mean)
grad_mean.update(chignolin_filter500_pbe_grad_mean)
grad_mean.update(ethanol_pbe_grad_mean)

grad_std = {}

reparam_factors = {}
reparam_factors.update(qmugs_pbe_reparams)
reparam_factors.update(chignolin_filter500_pbe_reparams)
reparam_factors.update(ethanol_pbe_reparams)

delta_coeff_mean = {}
delta_coeff_mean.update(qmugs_pbe_delta_coeff_mean)
delta_coeff_mean.update(chignolin_filter500_pbe_delta_coeff_mean)
delta_coeff_mean.update(ethanol_pbe_delta_coeff_mean)

delta_coeff_std = {}
delta_coeff_std.update(qmugs_pbe_delta_coeff_std)
delta_coeff_std.update(chignolin_filter500_pbe_delta_coeff_std)
delta_coeff_std.update(ethanol_pbe_delta_coeff_std)
