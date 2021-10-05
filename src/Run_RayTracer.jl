using LinearAlgebra
using Statistics
using NPZ
using NLsolve
using SpecialFunctions

include("../src/photon_raytrace.jl")

Mass_a = 4.1e-5;
Ax_g = 1e-12;
Mass_NS=1;
θm = 1.217;
ωPul = 16.912;#(2 .* π) ./ 0.3069;
B0 = 8.1e12;
rNS = 10;
#NS_vel_M = 200.0 ./ 2.998e5;
NS_vel_M = 0.00029;
NS_vel_T = 2.746;
vel_disp = 1e-5; # km/s, velocity dispersion of AMC
n_times = 2;
t_list = LinRange(0.0, 2.0 * π / ωPul, n_times)
CLen_Scale = true

phiVs = 20
thetaVs = 20
ln_tend = 20
threshold = 0.0001
sve = true;

file_tag = ""
single_density_field = true # if true, assume asymptotic density described by 1 number


function run_all()

##### This part unnecessary....
#
#    if !single_density_field
#        for i in 1:n_times
#            surface_solver(Mass_a, θm, ωPul, B0, rNS, t_list[i], NS_vel_M, NS_vel_T; nsteps=10, ln_tstart=-15, ln_tend=ln_tend,
#                            ode_err=1e-10, phiVs=phiVs, thetaVs=thetaVs, threshold=threshold, sve=sve);
#        end
#    end

    main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, t_list; ode_err=1e-5, CLen_Scale=CLen_Scale, NS_vel_M=NS_vel_M, NS_vel_T=NS_vel_T, file_tag=file_tag, RadApprox=RadApprox, phiVs=phiVs, thetaVs=thetaVs, vel_disp=vel_disp, single_density_field=single_density_field)
    period_average(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, t_list; ode_err=1e-5, CLen_Scale=CLen_Scale, NS_vel_M=NS_vel_M, NS_vel_T=NS_vel_T, file_tag=file_tag, RadApprox=RadApprox)
end

run_all()
