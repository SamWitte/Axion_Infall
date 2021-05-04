using LinearAlgebra
using NLsolve
using NPZ

include("../src/axion_trajs_out.jl")


Mass_a = 1e-6;
θm = 0.2;
ωPul = 1.0;
B0 = 1e14;
rNS = 10;
n_times = 2;
t_list = LinRange(0.0, 2.0 * π / ωPul, n_times)
NS_vel = [0 0 200.0] ./ 2.998e5;
ln_tend = 20
threshold = 0.0001
phiVs = 10
thetaVs = 10
sve = true;

for i in 1:n_times
    main_runner(Mass_a, θm, ωPul, B0, rNS, t_list[i], NS_vel; nsteps=10, ln_tstart=-15, ln_tend=ln_tend, ode_err=1e-10, phiVs=phiVs, thetaVs=thetaVs, threshold=threshold, sve=sve);
end
