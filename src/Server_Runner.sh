#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --mem=90G
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
echo This job was running on:
hostname
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
MassA=1.0e-5
Axg=1e-12

B0=1.6e14
ThetaM=0.2
rotW=1.67

NS_vel_M=200.0
NS_vel_T=0.0
M_MC=1e-10
R_MC=3.0e9
is_AS=0

trace_trajs=1
theta_cut_trajs=1
Trajs=100000
tagF=""


declare -i memPerjob
memPerjob=$((SLURM_MEM_PER_NODE/SLURM_NTASKS))
echo $memPerjob
echo $SLURM_NTASKS
for ((i = 0; i < $SLURM_NTASKS ; i++)); do
    srun --ntasks=1 --exclusive --mem=$memPerjob julia --threads 1 Run_RayTracer_Server.jl --MassA $MassA --Axg $Axg --B0 $B0 --ThetaM $ThetaM --rotW $rotW --NS_vel_M $NS_vel_M --NS_vel_T $NS_vel_T --M_MC $M_MC --R_MC $R_MC --is_AS $is_AS --trace_trajs $trace_trajs --theta_cut_trajs $theta_cut_trajs --Nts $Trajs --ftag $tagF$i --run_RT 1 &
    sleep 3
done
wait

srun --ntasks=1 --exclusive --mem=$memPerjob julia --threads 1 Run_RayTracer_Server.jl --MassA $MassA --Axg $Axg --B0 $B0 --ThetaM $ThetaM --rotW $rotW --NS_vel_M $NS_vel_M --NS_vel_T $NS_vel_T --M_MC $M_MC --R_MC $R_MC --is_AS $is_AS --trace_trajs $trace_trajs --theta_cut_trajs $theta_cut_trajs --Nts $Trajs --run_RT 0 --run_Combine 1 --ftag $tagF --side_runs $SLURM_NTASKS
