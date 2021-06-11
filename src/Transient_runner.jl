using ArgParse
using SpecialFunctions
using LinearAlgebra
using NPZ
using Dates
using Statistics
include("photon_raytrace.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--B0"
            arg_type = Float64
            default = 1.0e14
            
        "--P"
            arg_type = Float64
            default = 3.0
            
        "--ThetaM"
            arg_type = Float64
            default = 0.1
            
        "--mass"
            arg_type = Float64
            default = 1.0e-6
            
    end

    return parse_args(s)
end


function main_runner()
    parsed_args = parse_commandline()
    
    Mass_a = parsed_args["mass"]; # eV
    Ax_g = 1e-12; # 1/GeV
    θm = parsed_args["ThetaM"]; # rad
    ωPul = round(2π ./ parsed_args["P"], digits=4); # 1/s
    B0 = parsed_args["B0"]; # G
    rNS = 10.0; # km
    NS_vel = [141.421 0 141.421] ./ 2.998e5;
    Mass_NS = 1.0; # solar mass
    
    n_times = 20;
    t_list = LinRange(0.0, 2.0 * π / ωPul, n_times)

    phiVs = 100
    thetaVs = 100
    ln_tend = 20
    threshold = 0.0001
    CLen_Scale = true # if true, perform cut due to de-phasing
    sve = true;

    file_tag = ""
    
    print(Mass_a, "\t", θm, "\t", ωPul, "\t", B0, "\n")
    for i in 1:n_times
        @inbounds @fastmath surface_solver(Mass_a, θm, ωPul, B0, rNS, t_list[i], NS_vel; nsteps=10, ln_tstart=-15, ln_tend=ln_tend, ode_err=1e-10, phiVs=phiVs, thetaVs=thetaVs, threshold=threshold, sve=sve);
    end
    @inbounds @fastmath main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, t_list; ode_err=1e-5, CLen_Scale=CLen_Scale, v_NS=NS_vel, file_tag=file_tag, RadApprox=false, phiVs=phiVs, thetaVs=thetaVs)
    period_average(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, t_list; ode_err=1e-5, CLen_Scale=CLen_Scale, v_NS=NS_vel, file_tag=file_tag, RadApprox=false)

end

main_runner()
