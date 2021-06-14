import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from AMC_Density_Evolution import *
import glob
import healpy as hp

NFW = True
nside = 8

# run through each mass and each NS, determine coupling for which this would be observable
def Find_Ftransient(NFW=True, nside=8):
    # For now im computing min flux from SEFD = 0.098Jy, t_obs = 1 second, bandwidth 1e-4, and we take SNR = 5
    # this gives (for each mass): [0.177, 0.079, 0.0562, 0.032] mJy
    if NFW:
        orig_F = np.loadtxt('../encounter_data/Interaction_params_NFW_AScut_wStripping.txt')
    else:
        orig_F = np.loadtxt('../encounter_data/Interaction_params_PL_AScut_wStripping.txt')
        
    AxionMass = [1e-6, 5e-6, 1e-5, 3e-5] # eV
    fluxD_thresh = [0.177, 0.079, 0.0562, 0.032] # mJy
    glist = np.zeros(len(AxionMass), dtype=object)
    for i in range(len(AxionMass)):
        glist[i] = []
        
    files = glob.glob('results/Minicluster_PeriodAvg*')
    # cycle through output files
    for i in range(len(files)):
        find1 = files[i].find('MassAx_')
        find2 = files[i].find('_ThetaM_')
        print(files[i], find1, find2)
        axM = float(files[i][find1+len(find1):find2])
        indx = -1
        # identify mass so we know how to store
        for j in range(len(AxionMass)):
            if axM == AxionMass[i]:
                indx = j
        if indx == -1:
            print('Problem with mass!', axM)
            break
        
        # identify NS in original file
        find1 = files[i].find('_rotPulsar_')
        find2 = files[i].find('_B0_')
        periodN = float(files[i][find1+len(find1):find2])
        
        find1 = files[i].find('_B0_')
        find2 = files[i].find('_rNS_')
        B0 = float(files[i][find1+len(find1):find2])
        
        possible = np.where(np.round(orig_F[:, 6], 4) == periodN)[0]
        NSIndx = np.where(np.round(orig_F[:,7][possible], 4) == B0)[0]
        
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
        
        rate = np.sum(rel_rows[:, 5])  / hp.pixelfunc.nside2resol(n_side) # missing rho [eV / cm^3], will be in [eV / s]
        
        epsshift = 1e-3
        tmax = Transient_Time(bparam, rad_amc, dens_amc, vel)
        tlist = np.linspace(epsshift, tmax-epsshift, 200)
        dense_scan = np.zeros_like(tlist)
        for i in range(len(tlist)):
            dense_scan[i] = Transient_AMC_DensityEval(bparam, rad_amc, dens_amc, vel, tlist[i], nfw=NFW)
    
        rate *= np.max(dense_scan) * (1 / (dist * 3.086*10**18))**2 * 1.6022e-12 # erg / s / cm^2
        bwidth = axM * 1e-4 / 6.58e-16 # Hz
        rate *= (1/bwidth) * 1e26 # mJy
        glim = np.sqrt(fluxD_thresh[indx] / rate) * 1e-12 # GeV^-1
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

Find_Ftransient(NFW=NFW, nside=nside)
