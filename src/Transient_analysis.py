import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from AMC_Density_Evolution import *
import glob
import healpy as hp
import os

NFW = True
nside = 8
t_obs = 1.0

# run through each mass and each NS, determine coupling for which this would be observable

def fwhm_radio(mass_a, dsize=15):
    Dsize = 15 # m, for ska mid, 35 ska low
    freq = mass_a / (2*np.pi) / 6.58e-16 / 1e9 # GHz
    fwhm = 0.7 * (1 / freq) * (15 / Dsize)
    return fwhm
    
def fov_suppression(ang_dist, mass_a, dsize=15):
    FWHM = fwhm_radio(mass_a, dsize=dsize)
    Sense_StdDev = FWHM / 2.355
    suppress_F = np.exp(- ang_dist**2 / (2 * Sense_StdDev**2)) / (Sense_StdDev * np.sqrt(2*np.pi))
    return suppress_F

def sense_compute(mass, bwidth=1e-4, t_obs=1, SNR=5):
    # t_obs days, bwidth fractional
    SEFD = 0.098*1e3 #mJy
    return SNR * SEFD / np.sqrt(2 * mass * bwidth * t_obs * 24 * 60**2 / 6.58e-16)
    
def Find_Ftransient(NFW=True, nside=8, t_obs=1, bwidth=2e-5):
    # t_obs in days

    if NFW:
        orig_F = np.loadtxt('../encounter_data/Interaction_params_NFW_AScut_wStripping.txt')
    else:
        orig_F = np.loadtxt('../encounter_data/Interaction_params_PL_AScut_wStripping.txt')
        
    AxionMass = [1.0e-6, 5.0e-6, 1.0e-5, 3.0e-5] # eV
    glist = np.zeros(len(AxionMass), dtype=object)
    for i in range(len(AxionMass)):
        glist[i] = []
        
    files = glob.glob('results/Minicluster_PeriodAvg*')
    # cycle through output files
    for i in range(len(files)):
        find1 = files[i].find('MassAx_')
        find2 = files[i].find('_ThetaM_')
        
        axM = float(files[i][find1+len('MassAx_'):find2])
        indx = -1
        # identify mass so we know how to store
        for j in range(len(AxionMass)):
            if axM == AxionMass[j]:
                indx = j
        if indx == -1:
            print('Problem with mass!', axM)
            break
        
        # identify NS in original file
        find1 = files[i].find('_rotPulsar_')
        find2 = files[i].find('_B0_')
        periodN = (2 * np.pi) / float(files[i][find1+len('_rotPulsar_'):find2])
        
        find1 = files[i].find('_B0_')
        find2 = files[i].find('_rNS_')
        B0 = float(files[i][find1+len('_B0_'):find2])
        
        possible = np.where(np.round(orig_F[:, 6] / periodN, 3) == 1)[0]
        holdI = np.where(np.round(orig_F[:,7][possible] / B0, 3) == 1)[0]
        if len(holdI) != 0:
            NSIndx = possible[holdI]
        else:
            print('index failure...???', holdI)
            print(possible)
            print(periodN, B0, orig_F[:, 6][possible], orig_F[:, 7][possible], np.round(orig_F[:,7][possible] / B0, 2))
            return
        # print(possible)
        # print(NSIndx)
        # print(B0, periodN)
        # print(orig_F[:,7][possible])
        dist = orig_F[NSIndx, 0] # pc
        dens_amc = orig_F[NSIndx, 3] # M/pc^3
        rad_amc = orig_F[NSIndx, 4] # pc
        bparam = orig_F[NSIndx, 5] # pc
        vel = orig_F[NSIndx, -1] * 3.086*10**13 / 2.998e5 # unitless
        dens_amc *= 3.8 * 10**10 # eV/cm^3
        rad_amc *= 3.086*10**13 # km
        bparam *= 3.086*10**13 # km
        
        
        # compute flux density
        file_in = np.load(files[i])
        Theta = file_in[:,2]
        Phi = file_in[:,3]
        
        pixel_indices = hp.ang2pix(nside, Theta, Phi)
        indxs = hp.nside2npix(nside)
        viewA = int(np.random.rand(1) * indxs)
        rel_rows = file_in[pixel_indices == viewA]
        if len(rel_rows[:,0]) == 0 or np.sum(rel_rows[:, 5]) == 0:
            continue
        
        bins = np.linspace(np.min(rel_rows[:,6]), np.max(rel_rows[:,6]), 60)
        vals, bbs = np.histogram(rel_rows[:,6], bins=bins, weights=rel_rows[:,5])
        peakF = (bbs[np.argmax(vals)] + bbs[np.argmax(vals)+1]) / 2
        rel_rows2 = rel_rows[np.abs(peakF - rel_rows[:,6]) <= (bwidth / 2)]
        rate_TEST = np.sum(rel_rows[:, 5])  / hp.pixelfunc.nside2resol(nside) # missing rho [eV / cm^3], will be in [eV / s]
        rate = np.sum(rel_rows2[:, 5])  / hp.pixelfunc.nside2resol(nside) # missing rho [eV / cm^3], will be in [eV / s]
        print('Rate ratio: ', rate/rate_TEST, 'max width...', np.max(np.abs(peakF - rel_rows[:,6]) ), '90 percent', np.percentile(np.abs(peakF - rel_rows[:,6]), 90))
        
        t_shift = t_obs / 2.0 * 24.0 * 60.0**2 # seconds
        t_mid = Transient_Time(bparam, rad_amc, vel) / 2
        tlist = np.linspace(t_mid - t_shift, t_mid + t_shift, 200)
        dense_scan = np.zeros_like(tlist)
        for j in range(len(tlist)):
            dense_scan[j] = Transient_AMC_DensityEval(bparam, rad_amc, dens_amc, vel, tlist[j], nfw=NFW)[0]

        
        rate *= np.trapz(dense_scan.flatten(), tlist.flatten()) / (2*t_shift)  / (dist * 3.086*10**18)**2 * 1.6022e-12 # erg / s / cm^2
        bw_norm = axM * bwidth / 6.58e-16 # Hz
        rate *= (1.0/bw_norm) * 1e26 # mJy
        
        glong = orig_F[NSIndx, 1]
        glat = orig_F[NSIndx, 2]
        ang_dist = np.sqrt(glat**2 + glong**2)
        fovS = fov_suppression(ang_dist, axM, dsize=15)
        rate *= fovS
        print('Rate/RateT', rate / sense_compute(axM, bwidth=bwidth, t_obs=t_obs, SNR=5))
        #print(rate, sense_compute(axM, bwidth=bwidth, t_obs=t_obs, SNR=5))
        glim = np.sqrt(sense_compute(axM, bwidth=bwidth, t_obs=t_obs, SNR=5)  / rate) * 1e-12 # GeV^-1
        glist[indx].append(glim)

    if not os.path.isdir("amc_glims"):
        os.mkdir("amc_glims")
    for i in range(len(AxionMass)):
        fileN = "amc_glims/glim_"
        if NFW:
            fileN += "NFW_"
        else:
            fileN += "PL_"
        fileN += "AxionMass_{:.2e}_".format(AxionMass[i])
        fileN += ".dat"
        np.savetxt(fileN, glist[i])
        
    return

Find_Ftransient(NFW=NFW, nside=nside, t_obs=t_obs)
