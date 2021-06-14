import os
import numpy as np

fileN = '../encounter_data/Interaction_params_NFW_AScut_wStripping.txt'
#fileN = '../encounter_data/Interaction_params_PL_AScut_wStripping.txt'
Nruns = 1000

def Stripped_Files_For_RT(fileN, num_ns):
    loadF = np.loadtxt(fileN)
    if len(loadF[:,0]) > num_ns:
        selct_ints = np.random.choice(len(loadF), num_ns, replace=False)
        loadF = loadF[selct_ints,:]
    else:
        num_ns = len(loadF[:,0])
        
    result_dict = np.load("Population_Histogram.npz")
    flat_histogram = result_dict['Histogram'].flatten()
    draw_locs = np.random.choice(np.arange(len(flat_histogram)), size = num_ns, p = flat_histogram)
    draw_locs = np.vstack(np.unravel_index(draw_locs, result_dict['Histogram'].shape))
    Theta_M = result_dict['Theta_M_Centers'][draw_locs[1]] # Radians
    P = loadF[:, 6]
    B = loadF[:, 7]
    glong = loadF[:, 1]
    glat = loadF[:, 2]
    velM = loadF[:, -1] * 3.086e+13 / 2.998e5
    
    script_dir = "scripts"
    
    cnt = 1
    batches = 10
    
    AxionMass = [1e-6, 5e-6, 1e-5, 3e-5] # eV
    check = []
    cmds = []
    
    for i in range(num_ns):
        B0 = B[i]
        Period = P[i]
        ThetaM = Theta_M[i]
        glg = glong[i]
        glt = glat[i]
        velnorm = velM[i]
        if np.sqrt(glt**2 + glg**2) > 1:
            continue
    
        for j in range(len(AxionMass)):
            cmd = 'julia Transient_runner.jl ' +\
                    '--B0 {:.3e} --P {:.4f} --ThetaM {:.3f} --mass {:.3e} --vel {:.4e} \n '.format(B0, Period, ThetaM, AxionMass[j], velnorm)
                    
            cmds.append(cmd)

    for i in range(batches):
        fout=open(script_dir+'/Anteater_commands_{:d}.sh'.format(cnt), 'w+')
        fout.write('#! /bin/bash\n')
        fout.write('#SBATCH --time=40:00:00\n')
        fout.write('#SBATCH --mem=20G\n')
        fout.write('#SBATCH --nodes=1\n')
        fout.write('#SBATCH --cpus-per-task=1\n')
        fout.write('#SBATCH --ntasks=1\n')
        fout.write('cd ../ \n')
        for cmd in cmds[i::batches]:
            fout.write('{}'.format(cmd))
        fout.close()
        cnt +=1
    

    fout = open(script_dir + '/simall_Anteater.sh', 'w')
    fout.write('for ((i = 1 ; i < {:d} ; i++)); do \n'.format(cnt))
    fout.write('\t sbatch Anteater_commands_"$i".sh \n')
    fout.write('done \n')
    fout.close()
    
    return
    

Stripped_Files_For_RT(fileN, Nruns)
