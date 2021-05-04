__precompile__()

# Simple ray-tracer
#
# Author: Christoph Weniger, July 2020

module RayTracer

using ForwardDiff: gradient, derivative, Dual, Partials, hessian
using OrdinaryDiffEq
# using CuArrays
using DifferentialEquations
using NLsolve
using Plots
using NPZ

# CuArrays.allowscalar(false)

### Parallelized derivatives with ForwardDiff.Dual

# Seed 3-dim vector with dual partials for gradient calculation
seed = x -> [map(y -> Dual(y, (1., 0., 0.)), x[:,1]) map(y -> Dual(y, (0., 1., 0.)), x[:,2]) map(y -> Dual(y, (0., 0., 1.)), x[:,3])]

# Extract gradient from dual
grad = x -> [map(x -> x.partials[1], x) map(x -> x.partials[2], x) map(x -> x.partials[3], x)]


### Parallelized crossing calculations

struct Crossings
    i1
    i2
    weight
end

"""
Calculate values of matrix X at crossing points
"""
function apply(c::Crossings, A)
    A[c.i1] .* c.weight .+ A[c.i2] .* (1 .- c.weight)
end

"""
calcuates crossings along 2 axis
"""
function get_crossings(A)
    # Matrix with 2 for upward and -2 for downward crossings
    sign_A = sign.(A)
    #cross = sign_A[:, 2:end] - sign_A[:, 1:end-1]
    cross = sign_A[2:end] - sign_A[1:end-1]
    #print(cross)
    # Index just before crossing
    i1 = Array(findall(x -> x .!= 0., cross))

    # Index just behind crossing
    #i2 = map(x -> x + CartesianIndex(0, 1), i1)
    i2 = i1 .+ 1

    # Estimate weight for linear interpolation
    weight = A[i2] ./ (A[i2] .- A[i1])

    return Crossings(i1, i2, weight)
end



function func_axion!(du, u, Mvars, lnt)
    @inbounds begin
        t = exp.(lnt)
        x = u[:,1:3]
        v = u[:,4:6]
        
        r = sqrt.(sum(x .* x, dims=2))
        
        xhat = x ./ r
        massL = ones(length(r))
        if sum(r .< 10) > 0
            massL[r .< 10] .= 1.0 .* (r[r .< 10] ./ 10.0) .^ 3;
        end
        
        G = 132698000000.0 # km M_odot^-1 * (km/s)^2
        du[:,1:3] = -v .* 2.998e5 .* t;  # v is km/s, x in km, t [s]
        du[:,4:6] = G .* massL ./ r.^2 .* xhat ./ 2.998e5 .* t; # unitless, assume 1M NS
    end
end



#
function propagateAxion(x0::Matrix, k0::Matrix, nsteps::Int, NumerP::Array)
    ln_tstart, ln_tend, ode_err = NumerP
    tspan = (ln_tstart, ln_tend)
    saveat = (tspan[2] - tspan[1]) / (nsteps-1)

    # u0 = cu([x0 k0])
    u0 = ([x0 k0])
    
    probAx = ODEProblem(func_axion!, u0, tspan, [ln_tstart], reltol=ode_err*1e-4, abstol=ode_err, maxiters=1e6);
    sol = solve(probAx, Tsit5(), saveat=saveat);
    
    
    x = cat([u[:, 1:3] for u in sol.u]...,dims=3);
    v = cat([u[:, 4:6] for u in sol.u]...,dims=3);
    
    return x, v
    
end


function Find_Conversion_Surface(Ax_mass, t_in, θm, ω, B0, rNS; thetaVs=100, phiVs=100)

    # rayT = RayTracer;
    rL = 10 .^ LinRange(log10.(rNS), 3, 5000)
    θL = LinRange(0.02 , π - 0.02, thetaVs)
    ϕL = LinRange(0.02, 2 * π - 0.02, phiVs)
    finalVec = zeros(length(θL)*length(ϕL), 3)

    indx = 1
    for i in 1:length(θL)
        for k in 1:length(ϕL)

            x0 = [rL .* sin.(θL[i]) .* cos.(ϕL[k]) rL .* sin.(θL[i]) .* sin.(ϕL[k]) rL .* cos.(θL[i])];
            ωpL = GJ_Model_ωp_vec(x0, t_in, θm, ω, B0, rNS)

            cx_list = get_crossings(log.(ωpL) .- log.(Ax_mass))
            cross = apply(cx_list, rL)
            if length(cx_list.i1) != 0
                finalVec[indx, :] .= [θL[i], ϕL[k], cross[1]]
            else
                finalVec[indx, :] .= [θL[i], ϕL[k], 0.0]
            end
            indx += 1
        end
    end
    return finalVec
end



function Get_Normal_CosTheta(x_list, v_list, Ax_mass, t_in, θm, ω, B0, rNS)
    
    r = sqrt.(sum(x_list .^2, dims=2))
    ϕ = atan.(view(x_list, :, 2), view(x_list, :, 1))
    θ = acos.(view(x_list, :, 3) ./ r)
    
    
    ctheta = zeros(length(θ))
    shift = 0.001
    
    for i in 1:length(θ)
        rL = LinRange(r[i] ./ 4, r[i].*4, 5000);
        x0 = [rL .* sin.(θ[i] ) .* cos.(ϕ[i]) rL .* sin.(θ[i] ) .* sin.(ϕ[i]) rL .* cos.(θ[i])];
        ωpL = GJ_Model_ωp_vec(x0, t_in, θm, ω, B0, rNS);
        cx_list = get_crossings(log.(ωpL) .- log.(Ax_mass));
        cross = apply(cx_list, rL);
        
        xMEAN = [cross[1] .* sin.(θ[i]) .* cos.(ϕ[i]) cross[1] .* sin.(θ[i]) .* sin.(ϕ[i]) cross[1] .* cos.(θ[i])];
        
        
        
        x0 = [rL .* sin.(θ[i] .+ shift) .* cos.(ϕ[i]) rL .* sin.(θ[i] .+ shift) .* sin.(ϕ[i]) rL .* cos.(θ[i] .+ shift)];
        ωpL = GJ_Model_ωp_vec(x0, t_in, θm, ω, B0, rNS);
        cx_list = get_crossings(log.(ωpL) .- log.(Ax_mass));
        cross = apply(cx_list, rL);
        if length(cx_list.i1) != 0
            rx = cross[1]
        else
            rx = 10
        end
        finalVec_N1 = [cross[1] .* sin.(θ[i] .+ shift) .* cos.(ϕ[i]) cross[1] .* sin.(θ[i] .+ shift) .* sin.(ϕ[i]) cross[1] .* cos.(θ[i] .+ shift)];
        
        finalVec_N1 .-= xMEAN;
        
        x0 = [rL .* sin.(θ[i] .- shift) .* cos.(ϕ[i] .+ 2 .* shift) rL .* sin.(θ[i] .- shift) .* sin.(ϕ[i] + 2 .* shift) rL .* cos.(θ[i] .- shift)];
        ωpL = GJ_Model_ωp_vec(x0, t_in, θm, ω, B0, rNS);
        cx_list = get_crossings(log.(ωpL) .- log.(Ax_mass));
        cross = apply(cx_list, rL);
        if length(cx_list.i1) != 0
            rx = cross[1]
        else
            rx = 10
        end
        finalVec_N2 = [cross[1] .* sin.(θ[i] .- shift) .* cos.(ϕ[i] + 2 .* shift) cross[1] .* sin.(θ[i] .- shift) .* sin.(ϕ[i] .+ 2 .* shift) cross[1] .* cos.(θ[i] .- shift)];
        finalVec_N2 .-= xMEAN;
        
        val = [(finalVec_N1[2] .* finalVec_N2[3] .- finalVec_N1[3] .* finalVec_N2[2]) (finalVec_N1[3] .* finalVec_N2[1] .- finalVec_N1[1] .* finalVec_N2[3]) (finalVec_N1[1] .* finalVec_N2[2] .- finalVec_N1[2] .* finalVec_N2[1])];
        val ./= sqrt.(sum(val.^2));
        khat = v_list[i, :] ./  sqrt.(sum(v_list[i, :].^2));
        
        ctheta[i] = abs.(val[1] .* khat[1] .+ val[2] .* khat[2] .+ val[3] .* khat[3]);
    end
    
    return ctheta
end



function ωNR_e(x, k, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, Magnetic field, non-relativstic e only
    Bvec, ωpL = GJ_Model_vec(x, t, θm, ωPul, B0, rNS)

    kmag = sqrt.(sum(k .* k, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    ωp = sqrt.(sum(ωpL .* ωpL, dims=2))

    cθ = sum(k .* Bvec, dims=2) ./ (kmag .* Bmag)
    
    cθTerm = (1.0 .- 2.0 .* cθ.^2)
    
    # abs not necessary, but can make calculation easier
    return sqrt.(abs.(0.5 .* (kmag .^2 + ωp .^2 + sqrt.(abs.(kmag .^4 + ωp .^4 + 2.0 .* cθTerm .*kmag .^2 .* ωp .^2 )))))

end


function kNR_e(x, k, ω, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, Magnetic field, non-relativstic e only
    Bvec, ωpL = GJ_Model_vec(x, t, θm, ωPul, B0, rNS)

    kmag = sqrt.(sum(k .* k, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    ωp = sqrt.(sum(ωpL .* ωpL, dims=2))

    cθ = sum(k .* Bvec, dims=2) ./ (kmag .* Bmag)
    
    # abs not necessary, but can make calculation easier (numerical error can creap at lvl of 1e-16 for ctheta)
    return sqrt.(((ω.^2 .-  ωp.^2) ./ (1 .- cθ.^2 .* ωp.^2 ./ ω.^2)))
end


function dk_dl(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    ωErg = ω(x0, k0, t_start, θm, ωPul, B0, rNS, gammaF)
    dkdr_grd = grad(kNR_e(seed(x0), k0, ωErg, t_start, θm, ωPul, B0, rNS, gammaF))
    
    dkdr_proj = abs.(sum(k0 .* dkdr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    return dkdr_proj
end

# goldreich julian model
function GJ_Model_vec(x, t, θm, ω, B0, rNS)
    # For GJ model, return \vec{B} and \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]

    r = sqrt.(sum(x.^2, dims=2))
   
    ϕ = atan.(view(x, :, 2), view(x, :, 1))
    θ = acos.(view(x, :, 3) ./ r)
    
    ψ = ϕ .- ω.*t
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    
    Br = 2 .* Bnorm .* (cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θm) .* sin.(θ) .- sin.(θm) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θm) .* sin.(ψ)
    
    Bx = Br .* sin.(θ) .* cos.(ϕ) .+ Btheta .* cos.(θ) .* cos.(ϕ) .- Bphi .* sin.(ϕ)
    By = Br .* sin.(θ) .* sin.(ϕ) .+ Btheta .* cos.(θ) .* sin.(ϕ) .+ Bphi .* cos.(ϕ)
    Bz = Br .* cos.(θ) .- Btheta .* sin.(θ)

    ωp = (69.2e-6 .* sqrt.( abs.(3 .* cos.(θ).*( cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ) ) .- cos.(θm))).*
            sqrt.(B0./1e14 .* (rNS./r).^3 .* ω ./ (2 .* π )))

    # format: [e-, e+] last two -- plasma mass and gamma factor
    return [Bx By Bz], ωp
end

function GJ_Model_ωp_vec(x, t, θm, ω, B0, rNS)
    # For GJ model, return \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]

    r = sqrt.(sum(x .* x, dims=2))
    
    ϕ = atan.(view(x, :, 2), view(x, :, 1))
    θ = acos.(view(x, :, 3)./ r)
    ψ = ϕ .- ω.*t
    ωp = (69.2e-6 .* sqrt.( abs.(3 .* cos.(θ).*( cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ) ) .- cos.(θm))).*
            sqrt.(B0./1e14 .* (rNS./r).^3 .* ω ./ (2 .* π )))
    return ωp
end


end

function solve_vel_CS(θ, ϕ, r, NS_vel; guess=[0.1 0.1 0.1])
    ff = sum(NS_vel.^2); # unitless
    G = 132698000000.0 # km M_odot^-1 * (km/s)^2
    GMr = G ./ r ./ (2.998e5 .^ 2); # unitless
    rhat = [sin.(θ) .* cos.(ϕ) sin.(θ) .* sin.(ϕ) cos.(θ)]
    
    function f!(F, x)
        vx, vy, vz = x
        denom = ff .+ GMr .- sqrt.(ff) .* sum(x .* rhat);
        
        F[1] = (ff .* vx .+ sqrt.(ff) .* GMr .* rhat[1] .- sqrt.(ff) .* vx .* sum(x .* rhat)) ./ denom .- NS_vel[1]
        F[2] = (ff .* vy .+ sqrt.(ff) .* GMr .* rhat[2] .- sqrt.(ff) .* vy .* sum(x .* rhat)) ./ denom .- NS_vel[2]
        F[3] = (ff .* vz .+ sqrt.(ff) .* GMr .* rhat[3] .- sqrt.(ff) .* vz .* sum(x .* rhat)) ./ denom .- NS_vel[3]
        
    end
    
    soln = nlsolve(f!, guess, autodiff = :forward, ftol=1e-24, iterations=10000)
    # FF = zeros(3)
    # FF2 = zeros(3)
    # f!(FF, soln.zero)
    # f!(FF2, -soln.zero)
    # print(FF,"\t",FF2,"\n")
    return soln.zero
end

function test_vs_soln(θ, ϕ, r, NS_vel, x)
    ff = sum(NS_vel.^2); # unitless
    G = 132698000000.0 # km M_odot^-1 * (km/s)^2
    GMr = G ./ r ./ (2.998e5 .^ 2); # unitless
    rhat = [sin.(θ) .* cos.(ϕ) sin.(θ) .* sin.(ϕ) cos.(θ)]
    vx, vy, vz = x
    denom = ff .+ GMr .- sqrt.(ff) .* sum(x .* rhat);
        
    H1 = (ff .* vx .+ sqrt.(ff) .* GMr .* rhat[1] .- sqrt.(ff) .* vx .* sum(x .* rhat)) ./ denom .- NS_vel[1]
    H2 = (ff .* vy .+ sqrt.(ff) .* GMr .* rhat[2] .- sqrt.(ff) .* vy .* sum(x .* rhat)) ./ denom .- NS_vel[2]
    H3 = (ff .* vz .+ sqrt.(ff) .* GMr .* rhat[3] .- sqrt.(ff) .* vz .* sum(x .* rhat)) ./ denom .- NS_vel[3]
    return [H1 H2 H3]
end

function main_runner(Mass_a, θm, ωPul, B0, rNS, t_in, NS_vel; nsteps=10000, ln_tstart=-4, ln_tend=5, ode_err=1e-8, phiVs=100, thetaVs=100, threshold=0.05, sve=False)

    
    RT = RayTracer; # define ray tracer module
    ConvR = RT.Find_Conversion_Surface(Mass_a, t_in, θm, ωPul, B0, rNS, thetaVs=thetaVs, phiVs=phiVs)
    NumerP = [ln_tstart, ln_tend, ode_err]
    vGu=0.2
    
    finalX = zeros(2 * length(ConvR[:,1]), 3)
    finalV = zeros(2 * length(ConvR[:,1]), 3)
    
    SurfaceX = zeros(2 * length(ConvR[:,1]), 3)
    SurfaceV = zeros(2 * length(ConvR[:,1]), 3)
    cnt = 1;
    
    MagnetoVars = [θm, ωPul, B0, rNS, [1.0 1.0], t_in]
    
    for i in 1:length(ConvR[:,1])
        
        θ, ϕ, r = ConvR[i,:]

        
        velV = solve_vel_CS(θ, ϕ, r, NS_vel, guess=[vGu vGu vGu])
        velV2 = solve_vel_CS(θ, ϕ, r, NS_vel, guess=[-vGu -vGu -vGu])
        vel_init = [velV; velV2];
        # print(vel_init, "\n")
        # vel_init = [velV; -1.0 .* velV];
        # print(vel_init,"\n")
        x_init = [r .* sin.(θ) .* cos.(ϕ) r .* sin.(θ) .* sin.(ϕ) r .* cos.(θ); r .* sin.(θ) .* cos.(ϕ) r .* sin.(θ) .* sin.(ϕ) r .* cos.(θ)]
        xF, vF = RT.propagateAxion(x_init, vel_init, nsteps, NumerP);
        
   
        if sqrt.(sum((hcat(vF[1, :, end]...) .- NS_vel) .^ 2)) ./ sqrt.(sum(NS_vel.^2)) < threshold
            finalX[cnt, :] = xF[1, :, end];
            finalV[cnt, :] = vF[1, :, end];
            SurfaceX[cnt, :] = xF[1, :, 1];
            SurfaceV[cnt, :] = vF[1, :, 1];
            
            
            cnt += 1;
        # else
            # hold1 = test_vs_soln(θ, ϕ, r, NS_vel, velV)
            # print(hold1,"\t");
            # print(sqrt.(sum((hcat(vF[1, :, end]...) .- NS_vel) .^ 2)) ./ sqrt.(sum(NS_vel.^2)), "\t")
        end
        
        if sqrt.(sum((hcat(vF[2, :, end]...) .- NS_vel) .^ 2)) ./ sqrt.(sum(NS_vel.^2)) < threshold
            finalX[cnt, :] = xF[2, :, end];
            finalV[cnt, :] = vF[2, :, end];
            SurfaceX[cnt, :] = xF[2, :, 1];
            SurfaceV[cnt, :] = vF[2, :, 1];
            
            
            cnt += 1;
        # else
            # hold2 = test_vs_soln(θ, ϕ, r, NS_vel, velV2)
            # print(hold2,"\n");
            # print(sqrt.(sum((hcat(vF[2, :, end]...) .- NS_vel) .^ 2)) ./ sqrt.(sum(NS_vel.^2)), "\n")
        end
        
    end
    

    dkdl = RT.dk_dl(SurfaceX[1:(cnt-1),:], SurfaceV[1:(cnt-1),:], [RT.ωNR_e, MagnetoVars]) ./ 6.58e-16;
    ctheta = RT.Get_Normal_CosTheta(SurfaceX[1:(cnt-1),:], SurfaceV[1:(cnt-1),:], Mass_a, t_in, θm, ωPul, B0, rNS)
    if sve
        dirN = "temp_storage/"
        fileTail = "PhaseSpace_Map_AxionM_"*string(Mass_a)*"_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0)*"_rNS_";
        fileTail *= "Time_"*string(t_in)*"_sec_"
        if NS_vel[1] != 0
            fileTail *= "NS_velX_"*string(round(NS_vel[1], digits=4))
        end
        if NS_vel[2] != 0
            fileTail *= "NS_velY_"*string(round(NS_vel[2], digits=4))
        end
        if NS_vel[3] != 0
            fileTail *= "NS_velZ_"*string(round(NS_vel[3], digits=4))
        end
        fileTail *= "_.npz"
        npzwrite(dirN*"SurfaceX_"*fileTail, SurfaceX[1:(cnt-1),:])
        npzwrite(dirN*"SurfaceV_"*fileTail, SurfaceV[1:(cnt-1),:])
        npzwrite(dirN*"FinalX_"*fileTail, finalX[1:(cnt-1),:])
        npzwrite(dirN*"FinalV_"*fileTail, finalV[1:(cnt-1),:])
        npzwrite(dirN*"dkdz_"*fileTail, dkdl)
        npzwrite(dirN*"ctheta_"*fileTail, ctheta)
    else
        return SurfaceX[1:(cnt-1),:], SurfaceV[1:(cnt-1),:], finalX[1:(cnt-1),:], finalV[1:(cnt-1),:], dkdl, ctheta
    end

end

function test_runner(Mass_a, θm, ωPul, B0, rNS, t_in, NS_vel; indx=3000, nsteps=10000, ln_tstart=-4, ln_tend=5, ode_err=1e-8, phiVs=100, thetaVs=100)

    
    RT = RayTracer; # define ray tracer module
    ConvR = RT.Find_Conversion_Surface(Mass_a, t_in, θm, ωPul, B0, rNS, thetaVs=thetaVs, phiVs=phiVs)
    NumerP = [ln_tstart, ln_tend, ode_err]
    
    finalX = zeros(2 * length(ConvR[:,1]), 3)
    SurfaceX = zeros(2 * length(ConvR[:,1]), 3)
    SurfaceV = zeros(2 * length(ConvR[:,1]), 3)
    cnt = 1;
    
    θ, ϕ, r = ConvR[indx,:]
    velV = solve_vel_CS(θ, ϕ, r, NS_vel)
    vel_init = [velV; -velV];
    x_init = [r .* sin.(θ) .* cos.(ϕ) r .* sin.(θ) .* sin.(ϕ) r .* cos.(θ); r .* sin.(θ) .* cos.(ϕ) r .* sin.(θ) .* sin.(ϕ) r .* cos.(θ)]
    xF, vF = RT.propagateAxion(x_init, vel_init, nsteps, NumerP);
  
    return xF, vF

end
