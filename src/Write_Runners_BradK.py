import os
import numpy as np
import random


file_Load_Brad = np.loadtxt("../../../M31_encounter_data/Stripping_Models/Interaction_params_PL_M_AMC_1.00e-14_M31_youngNS_delta_1.txt")


freqRun = 8.0e9 # GHz
MassA = 2*np.pi * freqRun * 6.58e-16 # eV
Axg = 1e-14

Num_RUN = 10
NS_Vel_T_list = []
NS_Vel_M_list=[]
M_MC_list=[]
R_MC_list=[]
is_AS_list=[]
B0_List=[]
ThetaM_List=[]
rotW_List=[]

cnt = 0

def test_plamsaF(B, P, mass):
    op = 69.2 * np.sqrt(2) * np.sqrt(B / 1e14 / P) * 1e-6 # eV
    if op  < mass:
        return False
    else:
        return True



for i in range(Num_RUN):
    B0, P, ThM, Age, xx, yy, zz, MC_Den, MC_Rad, MC_Mass, b, velNS = file_Load_Brad[i, :]
    if test_plamsaF(B0, P, MassA):
        NS_Vel_T_list.append(np.arccos(1.0 - 2.0 * random.random()))
        NS_Vel_M_list.append(velNS * 3.086e13) # km/s

        # Mass = MC_Rad**3 * 4*np.pi * MC_Den / 3 # solar masses
        M_MC_list.append(MC_Mass)
        R_MC_list.append(MC_Rad * 3.086e13) # km
        is_AS_list.append(0)

        B0_List.append(B0)
        ThetaM_List.append(ThM)
        rotW_List.append(2*np.pi / P)
        cnt += 1
    else:
        print("fail.... \n")
        
Trajs = 10000 # trajs per task
trace_trajs = 1
theta_cut_trajs = 1
tagF = "_PL_"
fix_time = 0.0

print("frac survive: {:.2e}".format(float(cnt) / Num_RUN))
fileTag = "Server_Runner_"
batchSize = 10

total_runners = Num_RUN
file_out = []
for i in range(cnt):


    file_out_HOLD += "MassA={:.2e} \n".format(MassA)
    file_out_HOLD += "Axg={:.2e} \n".format(Axg)
    file_out_HOLD += "B0={:.2e} \n".format(B0_List[i])
    file_out_HOLD += "ThetaM={:.2e} \n".format(ThetaM_List[i])
    file_out_HOLD += "rotW={:.2e} \n".format(rotW_List[i])
    file_out_HOLD += "NS_vel_M={:.2e} \n".format(NS_Vel_M_list[i])
    file_out_HOLD += "NS_vel_T={:.2e} \n".format(NS_Vel_T_list[i])
    file_out_HOLD += "M_MC={:.2e} \n".format(M_MC_list[i])
    file_out_HOLD += "R_MC={:.2e} \n".format(R_MC_list[i])
    file_out_HOLD += "is_AS={:.0f} \n".format(is_AS_list[i])

    file_out_HOLD += "trace_trajs={:.0f} \n".format(trace_trajs)
    file_out_HOLD += "theta_cut_trajs={:.0f} \n".format(theta_cut_trajs)
    file_out_HOLD += "Trajs={:.0f} \n".format(Trajs)
    
    file_out_HOLD += "fix_time={:.2f} \n".format(fix_time)
    
    file_out_HOLD += "tagF=\"" + tagF + "\" \n"
    
    file_out_HOLD += "declare -i memPerjob \n"
    file_out_HOLD += "memPerjob=$((SLURM_MEM_PER_NODE/SLURM_NTASKS)) \n"

    file_out_HOLD += "for ((i = 0; i < $SLURM_NTASKS ; i++)); do \n"
    file_out_HOLD += "srun --ntasks=1 --exclusive --mem=$memPerjob julia --threads 1 Run_RayTracer_Server.jl --MassA $MassA --Axg $Axg --B0 $B0 --ThetaM $ThetaM --rotW $rotW --NS_vel_M $NS_vel_M --NS_vel_T $NS_vel_T --M_MC $M_MC --R_MC $R_MC --is_AS $is_AS --trace_trajs $trace_trajs --theta_cut_trajs $theta_cut_trajs --Nts $Trajs --ftag $tagF$i --run_RT 1 --fixed_time $fix_time & \n"
    file_out_HOLD += "sleep 3 \n"
    file_out_HOLD += "done \n"
    file_out_HOLD += "wait \n"
    
    file_out_HOLD += "srun --ntasks=1 --exclusive --mem=$memPerjob julia --threads 1 Run_RayTracer_Server.jl --MassA $MassA --Axg $Axg --B0 $B0 --ThetaM $ThetaM --rotW $rotW --NS_vel_M $NS_vel_M --NS_vel_T $NS_vel_T --M_MC $M_MC --R_MC $R_MC --is_AS $is_AS --trace_trajs $trace_trajs --theta_cut_trajs $theta_cut_trajs --Nts $Trajs --run_RT 0 --run_Combine 1 --ftag $tagF --side_runs $SLURM_NTASKS --fixed_time $fix_time \n"

    file_out_HOLD += "wait \n"
    
    file_out.append(file_out_HOLD)



for i in range(batchSize):

    fout=open(fileTag + '{}.sh'.format(i), 'w')
    fout.write('#! /bin/bash\n')
    fout.write('#SBATCH --mem=10G \n')
    fout.write('#SBATCH --nodes=1 \n')
    fout.write('#SBATCH --ntasks=10 \n')
    fout.write('#SBATCH --cpus-per-task=1 \n')
    
    
    for cmd in file_out[i::batchSize]:
        fout.write('{}'.format(cmd))
        fout.write('sleep 3 \n')
        fout.write('wait \n')
        
    fout.write('wait \n')
    fout.close()
    cnt +=1
    
    
    fout.write('{}\n'.format(file_out[i]))
    fout.close()
    

fout = open('simall_Anteater.sh', 'w')
fout.write('for ((i = {:d} ; i < {:d} ; i++)); do \n'.format(1, cnt+init_indx))
fout.write('\t sbatch Anteater_commands_"$i".sh \n')
fout.write('done \n')
fout.close()





    
