using LinearAlgebra
using Statistics
using NPZ
using NLsolve
using SpecialFunctions
using Dates
include("../src/photon_raytrace_mcmc.jl")

Mass_a = 1e-5;
Ax_g = 1e-12;
Mass_NS = 1;
θm = 0.0;
ωPul = (2 .* π);
B0 = 1e14;
rNS = 10;
gammaF = [1.0, 1.0]
batchsize = 5;
Ntajs = 10000
NS_vel_M = 0.000667;
NS_vel_T = π ./ 2.0;
CLen_Scale = false

M_MC = 1e-10 # Solar mass
R_MC = 3.06e9 # km
axion_star_nodisp = false
errSlve = 1e-24
fix_time= 0.0
period_average=true
trace_trajs = true;

file_tag = ""
single_density_field = true # if true, assume asymptotic density described by 1 number
RadApprox = false

function run_all()
    main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, Ntajs, gammaF, batchsize; ode_err=1e-5, maxR=Nothing, cutT=1000, fix_time=fix_time, CLen_Scale=false, file_tag=file_tag, ntimes=1000, v_NS=[0 0 0], period_average=period_average, errSlve=errSlve, M_MC=M_MC, R_MC=R_MC,  save_more=true, vmean_ax=220.0, ntimes_ax=10000, dir_tag="results", trace_trajs=trace_trajs, axion_star_nodisp=axion_star_nodisp)
end

time0=Dates.now()

run_all()


time1=Dates.now()
print("\n")
print("time diff: ", time1-time0)
print("\n")
