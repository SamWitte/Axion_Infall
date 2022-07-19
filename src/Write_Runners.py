import os
import numpy as np


NS_Vel_T_list = [0.0]
NS_Vel_M_list = [0.0]

M_MC_list = [1e12]
R_MC_list = [1.86e9]

is_AS_list = [0]

Trajs = 500000
trace_trajs = 1
theta_cut_trajs = 1
tagF = "_NFW_"

MassA = 2.6e-5
Axg = 1e-14

B0 = 1.6e14
ThetaM=0.2
rotW=1.67

fileTag = "Server_Runner_"

total_runners = len(M_MC_list)
file_out = []
for i in range(total_runners):

    file_out_HOLD = "#!/bin/bash \m"
    file_out_HOLD += "#SBATCH --time=100:00:00 \n"
    file_out_HOLD += "#SBATCH --mem=90G \n"
    file_out_HOLD += "#SBATCH --nodes=1 \n"
    file_out_HOLD += "#SBATCH --ntasks=10 \n"
    file_out_HOLD += "#SBATCH --cpus-per-task=1 \n"

    file_out_HOLD += "MassA={:.2e} \n".format(MassA)
    file_out_HOLD += "Axg={:.2e} \n".format(Axg)

    file_out_HOLD += "B0={:.2e} \n".format(B0)
    file_out_HOLD += "ThetaM={:.2e} \n".format(ThetaM)
    file_out_HOLD += "rotW={:.2e} \n".format(rotW)


    file_out_HOLD += "NS_vel_M={:.2e} \n".format(NS_Vel_M_list[i])
    file_out_HOLD += "NS_vel_T={:.2e} \n".format(NS_Vel_T_list[i])
    file_out_HOLD += "M_MC={:.2e} \n".format(M_MC_list[i])
    file_out_HOLD += "R_MC={:.2e} \n".format(R_MC_list[i])
    file_out_HOLD += "is_AS={:.0f} \n".format(is_AS_list[i])

    file_out_HOLD += "trace_trajs={:.0f} \n".format(trace_trajs)
    file_out_HOLD += "theta_cut_trajs={:.0f} \n".format(theta_cut_trajs)
    file_out_HOLD += "Trajs={:.0f} \n".format(Trajs)
    
    file_out_HOLD += "tagF=\"" + tagF + "\" \n"
    
    file_out_HOLD += "declare -i memPerjob \n"
    file_out_HOLD += "memPerjob=$((SLURM_MEM_PER_NODE/SLURM_NTASKS)) \n"

    file_out_HOLD += "for ((i = 0; i < $SLURM_NTASKS ; i++)); do \n"
    file_out_HOLD += "srun --ntasks=1 --exclusive --mem=$memPerjob julia --threads 1 Run_RayTracer_Server.jl --MassA $MassA --Axg $Axg --B0 $B0 --ThetaM $ThetaM --rotW $rotW --NS_vel_M $NS_vel_M --NS_vel_T $NS_vel_T --M_MC $M_MC --R_MC $R_MC --is_AS $is_AS --trace_trajs $trace_trajs --theta_cut_trajs $theta_cut_trajs --Nts $Trajs --ftag $tagF$i --run_RT 1 --fixed_time $fix_time & \n"
    file_out_HOLD += "sleep 3 \n"
    file_out_HOLD += "done \n"
    file_out_HOLD += "wait \n"
    
    file_out_HOLD += "srun --ntasks=1 --exclusive --mem=$memPerjob julia --threads 1 Run_RayTracer_Server.jl --MassA $MassA --Axg $Axg --B0 $B0 --ThetaM $ThetaM --rotW $rotW --NS_vel_M $NS_vel_M --NS_vel_T $NS_vel_T --M_MC $M_MC --R_MC $R_MC --is_AS $is_AS --trace_trajs $trace_trajs --theta_cut_trajs $theta_cut_trajs --Nts $Trajs --run_RT 0 --run_Combine 1 --ftag $tagF --side_runs $SLURM_NTASKS --fixed_time $fix_time \n"

    file_out.append(file_out_HOLD)



for i in range(total_runners):
    fout=open(fileTag + '{}.sh'.format(i), 'w')
    
    fout.write('{}\n'.format(file_out[i]))
    fout.close()





    
