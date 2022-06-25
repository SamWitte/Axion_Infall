using ArgParse
using LinearAlgebra
using Statistics
using NPZ
using NLsolve
using SpecialFunctions
using Dates
include("../src/photon_raytrace_mcmc.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--ThetaM"
            arg_type = Float64
            default = 0.2

        "--Nts"
            arg_type = Int
            default = 1000000

        "--ftag"
            arg_type = String
            default = ""

        "--rotW"
            arg_type = Float64
            default = 1.0

        "--MassA"
            arg_type = Float64
            default = 1e-5
            
        "--Axg"
            arg_type = Float64
            default = 1e-12
        
        "--B0"
            arg_type = Float64
            default = 1e14
            
        "--run_RT"
            arg_type = Int
            default = 1
            
        "--run_Combine"
            arg_type = Int
            default = 0
            
        "--side_runs"
            arg_type = Int
            default = 0
            
        "--rNS"
            arg_type = Float64
            default = 10.0
            
        "--Mass_NS"
            arg_type = Float64
            default = 1.0
            
        "--NS_vel_M"
            arg_type = Float64
            default = 200.0
            
        "--NS_vel_T"
            arg_type = Float64
            default = 0.0
            
        "--M_MC"
            arg_type = Float64
            default = 1e-10
            
        "--R_MC"
            arg_type = Float64
            default = 3e9
            
        "--is_AS"
            arg_type = Int
            default = 0 # 1 = AS, 0 = AMC

        "--trace_trajs"
            arg_type = Int
            default = 1 # 1 = trace
            
        "--theta_cut_trajs"
            arg_type = Int
            default = 1 # perform cut
            
        "--fixed_time"
            arg_type = Float64
            default = 0.0 # perform cut
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

# axion params
Mass_a = parsed_args["MassA"]; # eV
Ax_g = parsed_args["Axg"]; # 1/GeV

# neutron star params
θm = parsed_args["ThetaM"]; # rad
ωPul = parsed_args["rotW"]; # 1/s
B0 = parsed_args["B0"]; # G
rNS = parsed_args["rNS"]; # km
Mass_NS = parsed_args["Mass_NS"]; # solar mass
gammaF = [1.0, 1.0]

# AMC/AS Props
NS_vel_M = parsed_args["NS_vel_M"] ./ c_km
NS_vel_T = parsed_args["NS_vel_T"]
M_MC = parsed_args["M_MC"]
R_MC = parsed_args["R_MC"]
if parsed_args["is_AS"] == 1
    axion_star_moddisp = true
else
    axion_star_moddisp = false
end


# run params
Ntajs = parsed_args["Nts"];
batchSize = 5;

# analysis options
CLen_Scale = false # if true, perform cut due to de-phasing
fix_time = parsed_args["fixed_time"]; # eval at fixed time = 0?
ode_err = 1e-5;
errSlve = 1e-20;
if parsed_args["trace_trajs"] == 1
    trace_trajs = true
else
    trace_trajs = false
end
if parsed_args["theta_cut_trajs"] == 1
    theta_cut_trajs = true
else
    theta_cut_trajs = false
end

file_tag = parsed_args["ftag"];  #


print("Parameters: ", Mass_a, "\n")
print(Ax_g, "\n")
print(θm, "\n")
print(ωPul, "\n")
print(B0, "\n")
print(rNS, "\n")
print(Mass_NS, "\n")
print(Ntajs, "\n")
print(gammaF, "\n")
print(batchSize, "\n")

time0=Dates.now()

if parsed_args["run_RT"] == 1
    @inbounds @fastmath main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, Ntajs, gammaF, batchSize; ode_err=ode_err, fix_time=fix_time, CLen_Scale=CLen_Scale, file_tag=file_tag, ntimes=1000,  errSlve=errSlve, M_MC=M_MC, R_MC=R_MC,  save_more=true, ntimes_ax=10000, dir_tag="results", trace_trajs=trace_trajs, axion_star_moddisp=axion_star_moddisp, theta_cut_trajs=theta_cut_trajs)
end


function combine_files(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, Ntajs, NS_vel_M, NS_vel_T, R_MC, M_MC, Nruns, ode_err, fix_time, file_tag; trace_trajs=true, theta_cut_trajs=true)
   
    fileL = String[];
    
    for i = 0:(Nruns-1)
        file_tagL = file_tag * string(i)
        if fix_time != Nothing
            file_tagL *= "_fixed_time_"*string(fix_time);
        end
        file_tagL *= "_odeErr_"*string(ode_err);
        
        
        fileN = "results/Minicluster__MassAx_"*string(Mass_a)*"_AxG_"*string(Ax_g);
        fileN *= "_ThetaM_"*string(θm)*"_rotPulsar_"*string(round(ωPul, digits=3))*"_B0_"*string(B0)*"_rNS_";
        fileN *= string(rNS)*"_MassNS_"*string(Mass_NS)*"_Ntrajs_"*string(Ntajs);
        fileN *= "_NS_Mag_"*string(round(NS_vel_M, digits=5))*"_NS_Theta_"*string(round(NS_vel_T, digits=3))
        fileN *= "_Mmc_"*string(M_MC)*"_Rmc_"*string(R_MC)*"_"
        if trace_trajs
            fileN *= "_trace_trags_"
        end
        if theta_cut_trajs
            fileN *= "_thetaCN_"
        end
        fileN *= "_"*file_tagL*"_.npz";
    
        push!(fileL, fileN);
    end
    
    hold = npzread(fileL[1]);
    
    
    # divide off by num files combining...
    
    for i = 2:Nruns
        hold = vcat(hold, npzread(fileL[i]));
        Base.Filesystem.rm(fileL[i])
    end
    hold[:, 6] ./= Nruns;
    
    fileN = "results/Minicluster__MassAx_"*string(Mass_a)*"_AxG_"*string(Ax_g);
    fileN *= "_ThetaM_"*string(θm)*"_rotPulsar_"*string(round(ωPul, digits=3))*"_B0_"*string(B0)*"_rNS_";
    fileN *= string(rNS)*"_MassNS_"*string(Mass_NS)*"_Ntrajs_"*string(Ntajs * Nruns);
    fileN *= "_NS_Mag_"*string(round(NS_vel_M, digits=5))*"_NS_Theta_"*string(round(NS_vel_T, digits=3))
    fileN *= "_Mmc_"*string(M_MC)*"_Rmc_"*string(R_MC)*"_"
    if trace_trajs
        fileN *= "_trace_trags_"
    end
    if theta_cut_trajs
        fileN *= "_thetaCN_"
    end
    fileN *= "_"*file_tag*"_.npz";
    npzwrite(fileN, hold);
    
    for i = 1:Nruns
        Base.Filesystem.rm(fileL[i])
    end
    
end

if parsed_args["run_Combine"] == 1
    combine_files(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, Ntajs, NS_vel_M, NS_vel_T, R_MC, M_MC, parsed_args["side_runs"], ode_err, fix_time, file_tag; trace_trajs=trace_trajs, theta_cut_trajs=theta_cut_trajs)
end


time1=Dates.now()
print("\n")
print("time diff: ", time1-time0)
print("\n")








