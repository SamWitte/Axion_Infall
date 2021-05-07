using LinearAlgebra
using Statistics
using NPZ
using NLsolve
using SpecialFunctions

include("../src/photon_raytrace.jl")

Mass_a = 1e-6;
Ax_g = 1e-12;
Mass_NS=1;
θm = 0.2;
ωPul = 1.0;
B0 = 1e14;
rNS = 10;
NS_vel = [0 0 200.0] ./ 2.998e5;
n_times = 2;
t_list = LinRange(0.0, 2.0 * π / ωPul, n_times)

phiVs = 20
thetaVs = 20
ln_tend = 20
threshold = 0.0001
sve = true;

file_tag = "TEST_"
RadApprox = false


for i in 1:n_times
    surface_solver(Mass_a, θm, ωPul, B0, rNS, t_list[i], NS_vel; nsteps=10, ln_tstart=-15, ln_tend=ln_tend, ode_err=1e-10, phiVs=phiVs, thetaVs=thetaVs, threshold=threshold, sve=sve);
end

main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, t_list; ode_err=1e-7, CLen_Scale=true, v_NS=NS_vel, file_tag=file_tag, RadApprox=RadApprox)
period_average(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, t_list; ode_err=1e-7, CLen_Scale=true, v_NS=NS_vel, file_tag=file_tag, RadApprox=RadApprox)
