import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad, dblquad
from AMC_Density_Evolution import *
import glob
import healpy as hp
import os
from scipy.stats import poisson


NFW = True
nside = 8
t_obs = 0.4166 # days
bwidth = 9.5e-6
tele_name = 'GBT' # SKA-Mid, SKA-Low, Hirax, GBT
ax_mass = 4.1564e-5 # eV
NS_filename = 'Interaction_params_SJW_PL_AScut_ma_41_564mueV_delta_a.txt'
fov_hit = False # apply FoV suppression of rate...
fileTag = '_'

Time_int = 10 # time interval [days]
enc_rate = 1 # events / day
n_realize = 5 # number of realizations

# run through each mass and each NS, determine coupling for which this would be observable

haslam = hp.read_map('../haslam/haslam408_dsds_Remazeilles2014.fits')

def tele_details(tele_name):
    if tele_name == 'SKA-Mid':
        dsize = 15
        ndish = 5659
        T_rec = 20
        eta_coll = 0.8
        fname = '_SKA_Mid_'
    elif tele_name == 'SKA-Low':
        dsize = 35
        ndish = 1000
        T_rec = 40
        eta_coll = 0.8
        fname = '_SKA_Low_'
    elif tele_name == 'Hirax':
        dsize = 6
        T_rec = 50
        ndish = 1024
        eta_coll = 0.6
        fname = '_Hirax_'
        
    elif tele_name == 'GBT':
        dsize = 100
        T_rec = 20
        ndish = 1
        eta_coll = 0.7
        fname = '_GBT_'
    else:
        print('Telescope not included...')
        return
    return dsize, ndish, T_rec, eta_coll, fname


def fwhm_radio(mass_a, dsize=15): # dsize in m
    freq = mass_a / (2*np.pi) / 6.58e-16 / 1e9 # GHz
    fwhm = 0.7 * (1 / freq) * (15 / dsize)
    return fwhm # deg
    
def fov_suppression(ang_dist, mass_a, dsize=15):
    FWHM = fwhm_radio(mass_a, dsize=dsize)
    Sense_StdDev = FWHM / 2.355
    suppress_F = np.exp(- ang_dist**2 / (2 * Sense_StdDev**2)) / (Sense_StdDev * np.sqrt(2*np.pi))
    return suppress_F

def sense_compute(mass, bwidth=1e-3, t_obs=1, SNR=5, SEFD=None):
    # t_obs days, bwidth fractional
    if SEFD is None:
        SEFD = 0.098*1e3 #mJy
    
    return SNR * SEFD / np.sqrt(2 * mass * bwidth / (2*np.pi) * t_obs * 24 * 60**2 / 6.58e-16)
    
def sky_temp(mass, dsize=15):
    # sky temperatue
    rad_ang = fwhm_radio(mass, dsize=15) / 2 * np.pi/180.0
    Tsky = dblquad(lambda x,y: hp.get_interp_val(haslam, np.pi/2 + x, y)*np.cos(x), -rad_ang, rad_ang, lambda x: -rad_ang, lambda x: rad_ang, epsabs=1e-4, epsrel=1e-4)[0] /  (2*rad_ang)**2
    # this is value at 408 MHz, we then scale with freq nu^-2.55
    freq = mass / (2*np.pi) / 6.58e-16 / 1e6 # MHz
    return Tsky * (408.0 / freq)**2.55 # K
    
def SEFD_tele(mass, dsize=15, ndish=2000, T_rec=20, eta_coll=0.8):
    Aeff = np.pi * (dsize / 2)**2 * ndish * eta_coll * (1e2)**2 # cm ^2
    skyT = sky_temp(mass, dsize=dsize)
    T_tot = skyT + T_rec # K
    SEFD = 2 * T_tot / Aeff * 1.38e-16 / 1e-23 * 1e3 # mJy
    print('Freq [GHz]: {:.2e}, Sky Temp [K]: {:.2e}, SEFD [Jy]: {:.2e}'.format(mass / (2*np.pi) / 6.58e-16 / 1e9, skyT, SEFD * 1e-3))
    return SEFD
 
 
def Time_Ftransient(NFW=True, NS_filename='', mass=1e-5, nside=8, t_obs=1, bwidth=2e-5, dsize=15, ndish=2000, T_rec=20, eta_coll=0.8, tele_tag='', fov_hit=True, fileTag='', Time_int=365, enc_rate=1, n_realize=100, andromeda=True):
    # t_obs in days, Time_int [days], enc_rate [#/day]
    if enc_rate**-1 > Time_int:
        print('Encounter rate too small! Increase Time_int')
        return
    
    t_bin_center = Time_int / 2

    orig_F = np.loadtxt('../encounter_data/'+NS_filename)
    sefd_list = SEFD_tele(mass, dsize=dsize, ndish=ndish, T_rec=T_rec, eta_coll=eta_coll)
    
    files = glob.glob('results/Minicluster_PeriodAvg*')
    
    avgN_events = (enc_rate * Time_int)
    print('Avg number of events: {:.0f}'.format(avgN_events))
    if avgN_events > len(files):
        print('Not enough files generated, reduce Time_int')
        return
    
    g_realize = np.zeros(n_realize)
    n_cnt = 0
    
    
    while n_cnt < n_realize:
        Nevts = poisson.rvs(avgN_events)
        NSs = np.random.randint(0,high=len(files),size=Nevts)
        print('Realization {:.0f}, drawn events {:.0f}'.format(n_cnt, Nevts))
        
        rate = 0.0
        # cycle through output files
        for i in range(len(NSs)):
            # identify NS in original file
            find1 = files[NSs[i]].find('_rotPulsar_')
            find2 = files[NSs[i]].find('_B0_')
            periodN = (2 * np.pi) / float(files[NSs[i]][find1+len('_rotPulsar_'):find2])
            
            find1 = files[NSs[i]].find('_B0_')
            find2 = files[NSs[i]].find('_rNS_')
            B0 = float(files[NSs[i]][find1+len('_B0_'):find2])
            
            possible = np.where(np.round(orig_F[:, 6] / periodN, 3) == 1)[0]
            holdI = np.where(np.round(orig_F[:,7][possible] / B0, 3) == 1)[0]
            if len(holdI) != 0:
                NSIndx = possible[holdI]
            else:
                print('index failure...???', holdI)
                print(possible)
                print(periodN, B0, orig_F[:, 6][possible], orig_F[:, 7][possible], np.round(orig_F[:,7][possible] / B0, 2))
                return
            
            dist = orig_F[NSIndx, 0] # pc
            dens_amc = orig_F[NSIndx, 3] # M/pc^3
            rad_amc = orig_F[NSIndx, 4] # pc
            bparam = orig_F[NSIndx, 5] # pc
            vel = orig_F[NSIndx, -1] * 3.086*10**13 / 2.998e5 # unitless
            dens_amc *= 3.8 * 10**10 # eV/cm^3
            rad_amc *= 3.086*10**13 # km
            bparam *= 3.086*10**13 # km
        
        
            # compute flux density
            file_in = np.load(files[NSs[i]])
            Theta = file_in[:,2]
            Phi = file_in[:,3]
            
            pixel_indices = hp.ang2pix(nside, Theta, Phi)
            indxs = hp.nside2npix(nside)
            viewA = int(np.random.rand(1) * indxs)
            rel_rows = file_in[pixel_indices == viewA]
            if len(rel_rows[:,0]) == 0 or np.sum(rel_rows[:, 5]) == 0:
                print('Rate is 0...', rel_rows[:, 5])
                continue
            
            b_low = np.percentile(rel_rows[:,6], 10)
            b_high = np.percentile(rel_rows[:,6], 90)
            bins = np.linspace(b_low, b_high, 5000)
            rate_hold = np.zeros_like(bins)
            for kk in range(len(bins)):
                rate_hold[kk] = np.sum(rel_rows[np.abs(bins[kk] - rel_rows[:,6]) <= (bwidth / 2), 5]) / hp.pixelfunc.nside2resol(nside) # missing rho [eV / cm^3], will be in [eV / s]
            peakF = bins[np.argmax(rate_hold)]
            rate_temp = rate_hold[np.argmax(rate_hold)]
        
            # time analysis....
            # sample peak time
            t_mid = Transient_Time(bparam, rad_amc, vel) / 2
            peakT = np.random.rand() * Time_int
            t0 = peakT - t_mid / (60**2 * 24) # days
            t_min = t_bin_center - t_obs / 2
            t_max = t_bin_center + t_obs / 2
            
            print(' \t Start time and peak time of event: {:.2f} and {:.2f}. Transit time: {:.2f}'.format(t0[0], peakT, t_mid[0] / (60**2 * 24) * 2))
            tlist = np.linspace(t_min - t0, t_max - t0, 200) * (60**2 * 24)
        
        
            dense_scan = np.zeros_like(tlist)
            for j in range(len(tlist)):
                dense_scan[j] = Transient_AMC_DensityEval(bparam, rad_amc, dens_amc, vel, tlist[j], nfw=NFW)[0]
         
                
            bw_norm = mass * bwidth / (2*np.pi) / 6.58e-16 # Hz
            if not andromeda:
                rate_temp *= np.trapz(dense_scan.flatten(), tlist.flatten()) / (2*t_obs  * (60**2 * 24) )  / (dist * 3.086*10**18)**2 * 1.6022e-12 / bw_norm * 1e26 # mJy
            else:
                rate_temp *= np.trapz(dense_scan.flatten(), tlist.flatten()) / (2*t_obs  * (60**2 * 24) )  / (765.0 * 1e3 * 3.086*10**18)**2 * 1.6022e-12 / bw_norm * 1e26 # mJy
        
            glong = orig_F[NSIndx, 1]
            glat = orig_F[NSIndx, 2]
            if not andromeda:
                ang_dist = np.sqrt(glat**2 + glong**2)
            else:
                ang_dist = np.arctan((dist - 8.5e3) * np.sin(glat) / (765.0 * 1e3)) * 180 / np.pi
            fovS = fov_suppression(ang_dist, mass, dsize=dsize)
            if fov_hit:
                rate_temp *= fovS
            
            print(' \t\t Rate: ',rate_temp)
#            if rate_temp == 0:
#                print(rel_rows[:, 5], fovS, rate, rate_hold)
#                continue
            #print(rate, sense_compute(axM, bwidth=bwidth, t_obs=t_obs, SNR=5))
            rate += rate_temp
        if rate > 0:
            #glim = np.sqrt(sense_compute(mass, bwidth=bwidth, t_obs=t_obs, SNR=5, SEFD=sefd_list)  / rate) * 1e-12 # GeV^-1
            glim = np.sqrt(1.3  / rate) * 1e-12 # GeV^-1
        else:
            glim = 1e10
       
        g_realize[n_cnt] = glim
        n_cnt += 1

    if not os.path.isdir("amc_glims"):
        os.mkdir("amc_glims")
    
    fileN = "amc_glims/glim_TimeAnalysis_"
    fileN += tele_tag + "_"
    if NFW:
        fileN += "NFW_"
    else:
        fileN += "PL_"
    fileN += "AxionMass_{:.2e}_Bwdith_{:.2e}_Tbin_{:.3f}_days_".format(mass, bwidth, t_obs)
    fileN += fileTag
    fileN += ".dat"
    np.savetxt(fileN, g_realize)
        
    return
    

def Find_Ftransient(NFW=True, NS_filename='', mass=1e-5, nside=8, t_obs=1, bwidth=2e-5, dsize=15, ndish=2000, T_rec=20, eta_coll=0.8, tele_tag='', fov_hit=True, fileTag='', andromeda=True):
    # t_obs in days

    
    orig_F = np.loadtxt('../encounter_data/'+NS_filename)
        
    AxionMass = [mass] # eV
    glist = np.zeros(len(AxionMass), dtype=object)
    sefd_list = np.zeros(len(AxionMass))
    for i in range(len(AxionMass)):
        glist[i] = []
        sefd_list[i] = SEFD_tele(AxionMass[i], dsize=dsize, ndish=ndish, T_rec=T_rec, eta_coll=eta_coll)
        
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
        dens_amc *= 3.8*10**10 # eV/cm^3
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
        
        b_low = np.percentile(rel_rows[:,6], 10)
        b_high = np.percentile(rel_rows[:,6], 90)
        bins = np.linspace(b_low, b_high, 5000)
        rate_hold = np.zeros_like(bins)
        for kk in range(len(bins)):
            rate_hold[kk] = np.sum(rel_rows[np.abs(bins[kk] - rel_rows[:,6]) <= (bwidth / 2), 5]) / hp.pixelfunc.nside2resol(nside) # [cm/s] -- missing rho [eV / cm^3], will be in [eV / s]
        peakF = bins[np.argmax(rate_hold)]
        rate = rate_hold[np.argmax(rate_hold)]
        # rel_rows2 = rel_rows[np.abs(peakF - rel_rows[:,6]) <= (bwidth / 2)]
        rate_TEST = np.sum(rel_rows[:, 5])  / hp.pixelfunc.nside2resol(nside) # missing rho [eV / cm^3], will be in [eV / s]
        
        # print('Rate ratio: ', rate/rate_TEST, 'max width...', np.max(np.abs(peakF - rel_rows[:,6]) ), '90 percent', np.percentile(np.abs(peakF - rel_rows[:,6]), 90))
        
        
        t_shift = t_obs / 2.0 * 24.0 * 60.0**2 # seconds
        t_mid = Transient_Time(bparam, rad_amc, vel) / 2
        tlist = np.linspace(t_mid - t_shift, t_mid + t_shift, 200)
        dense_scan = np.zeros_like(tlist)
        for j in range(len(tlist)):
            dense_scan[j] = Transient_AMC_DensityEval(bparam, rad_amc, dens_amc, vel, tlist[j], nfw=NFW)[0]
         
                
        bw_norm = axM * bwidth / (2*np.pi) / 6.58e-16 # Hz
        if not andromeda:
            rate_temp *= np.trapz(dense_scan.flatten(), tlist.flatten()) / (tlist[-1] - tlist[0])  / (dist * 3.086*10**18)**2 * 1.6022e-12 / bw_norm * 1e26 # mJy
        else:
            rate_temp *= np.trapz(dense_scan.flatten(), tlist.flatten()) / (tlist[-1] - tlist[0])   / (765.0 * 1e3 * 3.086*10**18)**2 * 1.6022e-12 / bw_norm * 1e26 # mJy
        
        glong = orig_F[NSIndx, 1]
        glat = orig_F[NSIndx, 2]
        if not andromeda:
            ang_dist = np.sqrt(glat**2 + glong**2)
        else:
            ang_dist = np.atan((dist - 8.5e3) / (765.0 * 1e3)) * 180 / np.pi
        fovS = fov_suppression(ang_dist, axM, dsize=dsize)
        if fov_hit:
            rate *= fovS
            
        
        if rate == 0:
            print(rel_rows[:, 5], fovS, rate, rate_hold)
            continue
        #print(rate, sense_compute(axM, bwidth=bwidth, t_obs=t_obs, SNR=5))
        #glim = np.sqrt(sense_compute(axM, bwidth=bwidth, t_obs=t_obs, SNR=5, SEFD=sefd_list[indx])  / rate) * 1e-12 # GeV^-1
        glim = np.sqrt(1.3 / rate) * 1e-12 # GeV^-1
        if glim == 0 or np.isnan(glim):
            print('Here...', hp.pix2ang(nside, viewA), dense_scan.flatten(), )
        glist[indx].append(glim)

    if not os.path.isdir("amc_glims"):
        os.mkdir("amc_glims")
    for i in range(len(AxionMass)):
        fileN = "amc_glims/glim_"
        fileN += tele_tag + "_"
        if NFW:
            fileN += "NFW_"
        else:
            fileN += "PL_"
        fileN += "AxionMass_{:.2e}_Bwdith_{:.2e}_Tbin_{:.3f}_days_".format(AxionMass[i], bwidth, t_obs)
        fileN += fileTag
        fileN += ".dat"
        np.savetxt(fileN, glist[i])
        
    return

dsize, ndish, T_rec, eta_coll, tele_tag = tele_details(tele_name)
Find_Ftransient(NFW=NFW, NS_filename=NS_filename, mass=ax_mass, nside=nside, bwidth=bwidth, t_obs=t_obs, dsize=dsize, ndish=ndish, T_rec=T_rec, eta_coll=eta_coll, tele_tag=tele_tag, fov_hit=fov_hit, fileTag=fileTag)

Time_Ftransient(NFW=NFW, NS_filename=NS_filename, mass=ax_mass, nside=nside, t_obs=t_obs, bwidth=bwidth, dsize=dsize, ndish=ndish, T_rec=T_rec, eta_coll=eta_coll, tele_tag=tele_tag, fov_hit=fov_hit, fileTag=fileTag, Time_int=Time_int, enc_rate=enc_rate, n_realize=n_realize)
