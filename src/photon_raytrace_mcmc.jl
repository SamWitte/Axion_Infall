__precompile__()

# Simple ray-tracer


include("Constants.jl")
import .Constants: c_km, hbar, GNew

module RayTracer
import ..Constants: c_km, hbar, GNew
using ForwardDiff: gradient, derivative, Dual, Partials, hessian
using OrdinaryDiffEq
# using CuArrays
using DifferentialEquations
# using Plots
using NLsolve
using LSODA


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


### Parallelized, GPU-backed auto-differentiation ray-tracer

# compute photon trajectories
function func!(du, u, Mvars, lnt)
    @inbounds begin
        t = exp.(lnt);

        ω, Mvars2 = Mvars;
        θm, ωPul, B0, rNS, gammaF, time0 = Mvars2;
        time = time0 .+  t;
        
        dωdx = grad(ω(seed(view(u, :, 1:3)), view(u, :, 4:6), time0 .+ t, θm, ωPul, B0, rNS, gammaF));
        dωdk = grad(ω(view(u, :, 1:3), seed(view(u, :, 4:6)), time0 .+ t, θm, ωPul, B0, rNS, gammaF));
        dωdt = derivative(tI -> ω(view(u, :, 1:3),view(u, :, 4:6), tI, θm, ωPul, B0, rNS, gammaF), time[1]);
        du[:,1:3] = dωdk .* t .* 2.998e5;
        du[:,4:6] = -1.0 .* dωdx .* t  .* 2.998e5;
        du[:, 7] = dωdt .* t;
        
    end
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



# propogate photon module
function propagate(ω, x0::Matrix, k0::Matrix,  nsteps::Int, Mvars::Array, NumerP::Array)
    ln_tstart, ln_tend, ode_err = NumerP
    tspan = (ln_tstart, ln_tend)
    saveat = (tspan[2] .- tspan[1]) ./ (nsteps-1)

    # u0 = cu([x0 k0])
    u0 = ([x0 k0 zeros(length(x0[:, 1]))])

    prob = ODEProblem(func!, u0, tspan, [ω, Mvars], reltol=ode_err*1e-1, abstol=ode_err, maxiters=1e5)
    sol = solve(prob, Vern6(), saveat=saveat)
    # sol = solve(prob, Tsit5(), saveat=saveat)
    x = cat([Array(u)[:, 1:3] for u in sol.u]..., dims = 3);
    k = cat([Array(u)[:, 4:6] for u in sol.u]..., dims = 3);
    dt = cat([Array(u)[:, 7] for u in sol.u]..., dims = 2);
    
    return x, k, dt
    # return x, k
end

function propagateAxion(x0::Matrix, k0::Matrix, nsteps::Int, NumerP::Array)
    ln_tstart, ln_tend, ode_err = NumerP
    tspan = (ln_tstart, ln_tend)
    saveat = (tspan[2] - tspan[1]) / (nsteps-1)

    # u0 = cu([x0 k0])
    u0 = ([x0 k0])

    probAx = ODEProblem(func_axion!, u0, tspan, [ln_tstart], reltol=ode_err*1e-3, abstol=ode_err, maxiters=1e7);
    # sol = solve(probAx, Tsit5(), saveat=saveat);
    # sol = solve(probAx, Vern6(), saveat=saveat)
    sol = solve(probAx, lsoda(), saveat=saveat)


    x = cat([u[:, 1:3] for u in sol.u]...,dims=3);
    v = cat([u[:, 4:6] for u in sol.u]...,dims=3);

    return x, v

end

function solve_vel_CS(θ, ϕ, r, NS_vel; guess=[0.1 0.1 0.1], errV=1e-24)
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
        # print(F[1], "\t",F[2], "\t", F[3],"\n")
        # print(θ, "\t", ϕ,"\t", r, "\n")
    end

    soln = nlsolve(f!, guess, autodiff = :forward, ftol=errV, iterations=10000)

    # FF = zeros(3)
    # FF2 = zeros(3)
    # f!(FF, soln.zero)
    # f!(FF2, -soln.zero)
    # print(FF,"\t",FF2,"\n")
    return soln.zero
end

function jacobian_Lville(θ, ϕ, r, NS_vel, v_surf)
    ff = sum(NS_vel.^2); # unitless
    GM = 132698000000.0 ./ (2.998e5 .^ 2) # km
    GMr = GM ./ r ; # unitless
    rhat = [sin.(θ) .* cos.(ϕ) sin.(θ) .* sin.(ϕ) cos.(θ)]
    vx, vy, vz = v_surf
    dvx_vx = 1.0 .+ GM .* (-GM .- ff .* r .+ sqrt.(ff) .* r .* (vz .* sin.(θ) .* cos.(ϕ).^2 .+ vy .* sin.(ϕ))) ./ (GM .+ ff.*r .- sqrt.(ff).*r .* (vz .* cos.(θ) .+ sin.(θ) .* (vx .* cos.(ϕ) .+ vy .* sin.(ϕ)) )).^2
    dvy_vy_numer_1 = ((sqrt.(ff) .- vz .* cos.(θ) .- vx.*sin.(θ).*cos.(ϕ)) .* (GM .+ ff .* r .- sqrt.(ff).*r .* (vz .* cos.(θ) .+ vx .* sin.(θ) .* cos.(ϕ))) );
    dvy_vy_numer_2 = 2 .* vy .* sin.(θ) .* (-GM .- ff .* r .+ sqrt.(ff) .* r .* (vz .* cos.(θ) .+ vx .* sin.(θ) .* cos.(ϕ))) .* sin.(ϕ) .+ sqrt.(ff) .* (GM .+ r .* vy.^2).*sin.(ϕ).^2 .* sin.(θ).^2;
    dvy_vy_numer = sqrt.(ff).*r.*(dvy_vy_numer_1 .+ dvy_vy_numer_2);
    dvy_vy_denom = (GM .+ ff.*r .- sqrt.(ff).*r .* (vz .* cos.(θ) .+ sin.(θ) .* (vx .* cos.(ϕ) .+ vy .* sin.(ϕ)) )).^2
    dvy_vy = dvy_vy_numer / dvy_vy_denom;
    dvz_vz = 1.0 .+ GM .* (-GM .- ff .* r .+ sqrt.(ff) .* r .* (vz .* sin.(θ) .* cos.(ϕ).^2 .+ vy .* sin.(ϕ))) ./ (GM .+ ff.*r .- sqrt.(ff).*r .* (vz .* cos.(θ) .+ sin.(θ) .* (vx .* cos.(ϕ) .+ vy .* sin.(ϕ)) )).^2
    # print(abs.(dvx_vx).^(-1.0),"\t", abs.(dvy_vy).^(-1.0), "\t", abs.(dvz_vz).^(-1.0), "\t")
    return abs.(dvx_vx).^(-1.0) .* abs.(dvy_vy).^(-1.0) .* abs.(dvz_vz).^(-1.0)
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


function test_runner_surface_solver(Mass_a, θm, ωPul, B0, rNS, t_in, NS_vel_M, NS_vel_T; indx=3000, nsteps=10000, ln_tstart=-4, ln_tend=5, ode_err=1e-8, phiVs=100, thetaVs=100)

    NS_vel = [sin.(NS_vel_T) 0.0 cos.(NS_vel_T)] .* NS_vel_M;

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


function dwdt_vec(x0, k0, tarr, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    nphotons = size(x0)[1]
    delW = zeros(nphotons);
    
    erg = zeros(nphotons, length(tarr));
    t0 = tarr .+ t_start[1];
    for i in 1:length(tarr)
        erg[:, i] = ω(x0[:, :, i], k0[:, :, i], t0[i], θm, ωPul, B0, rNS, gammaF)
    end
    
    for k in 1:nphotons
        t0 = tarr .+ t_start[k];
        
        for i in 2:length(tarr)
            dωdt = derivative(t -> ω(transpose(x0[k, :, i]), transpose(k0[k, :, i]), t, θm, ωPul, B0, rNS, gammaF), t0[i]);
            delW[k] += dωdt[1] .* (t0[i] .- t0[i-1]);
        end
    end
    return delW
end

function cyclotronF(x0, t0, θm, ωPul, B0, rNS)
    Bvec, ωp = GJ_Model_scalar(x0, t0, θm, ωPul, B0, rNS)
    omegaC = sqrt.(sum(Bvec.^2, dims=2)) * 0.3 / 5.11e5 * (1.95e-20 * 1e18) # eV
    return omegaC
end

function cyclotronF_vec(x0, t0, θm, ωPul, B0, rNS)
    Bvec, ωp = GJ_Model_vec(x0, t0, θm, ωPul, B0, rNS)
    omegaC = sqrt.(sum(Bvec.^2, dims=2)) * 0.3 / 5.11e5 * (1.95e-20 * 1e18) # eV
    return omegaC
end

function tau_cyc(x0, k0, tarr, Mvars, Mass_a)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    nphotons = size(x0)[1]
    cxing_indx = zeros(Int16, nphotons)
    tau = zeros(nphotons)
    xpoints = zeros(nphotons, 3)
    kpoints = zeros(nphotons, 3)
    tpoints = zeros(nphotons)
    # cyclFOut = zeros(nphotons)
    
    for k in 1:nphotons
        t0 = tarr .+ t_start[k];
        cyclF = zeros(length(t0));
        for i in 1:length(tarr)
            cyclF[i] = cyclotronF(x0[k, :, i], t0[i], θm, ωPul, B0, rNS)[1];
        end
        cxing_st = get_crossings(log.(cyclF) .- log.(Mass_a));
        if length(cxing_st.i1) == 0
            tpoints[k] = t0[1]
            xpoints[k, :] = x0[k, :, 1]
            kpoints[k, :] = [0 0 0]
            
        else
            
            tpoints[k] = t0[cxing_st.i1[1]] .* cxing_st.weight[1] .+ (1.0 - cxing_st.weight[1]) .* t0[cxing_st.i2[1]];
            xpoints[k, :] = (x0[k, :, cxing_st.i1[1]]  .* cxing_st.weight[1] .+  (1.0 - cxing_st.weight[1]) .* x0[k, :, cxing_st.i2[1]])
            kpoints[k, :] = (k0[k, :, cxing_st.i1[1]]  .* cxing_st.weight[1] .+  (1.0 - cxing_st.weight[1]) .* k0[k, :, cxing_st.i2[1]])
        end
        # cyclFOut[k] = cyclotronF(xpoints[k, :], tpoints[k], θm, ωPul, B0, rNS)[1]
        
    end
    
    ωp = GJ_Model_ωp_vec(xpoints, tpoints, θm, ωPul, B0, rNS)
    dOc_grd = grad(cyclotronF_vec(seed(xpoints), tpoints, θm, ωPul, B0, rNS))
    kmag = sqrt.(sum(kpoints .^ 2, dims=2))
    dOc_dl = abs.(sum(kpoints .* dOc_grd, dims=2))
    dOc_dl[kmag .> 0] ./= kmag[kmag .> 0]
    tau = π * ωp .^2 ./ dOc_dl ./ (2.998e5 .* 6.58e-16);
    
    if sum(kmag .== 0) > 0
        tau[kmag .== 0] .= 0.0
        
    end
    
    
    return tau
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

# computing dωp/dr along axion traj in conversion prob
function dwdr_abs_vec(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    dωdr_grd = grad(GJ_Model_ωp_vec(seed(x0), t_start, θm, ωPul, B0, rNS))
    dωdr_proj = abs.(sum(k0 .* dωdr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    return dωdr_proj
end

function surfNorm(x0, k0, Mvars; return_cos=true)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    dωdr_grd = grad(GJ_Model_ωp_vec(seed(x0), t_start, θm, ωPul, B0, rNS))
    snorm = dωdr_grd ./ sqrt.(sum(dωdr_grd .^ 2, dims=2))
    ctheta = (sum(k0 .* snorm, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    # print(k0, "\n", snorm,"\n\n")
    if return_cos
        return ctheta
    else
        return ctheta, snorm
    end
end


function d2wdr2_abs_vec(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    dωdr2_grd = grad(dwdr_abs_vec(seed(x0), k0, Mvars))
    dωdr2_proj = abs.(sum(k0 .* dωdr2_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
  
    dwdr = dwdr_abs_vec(x0, k0, Mvars)
    θ = theta_B(x0, k0, Mvars)
    
    d0dr_grd = grad(theta_B(seed(x0), k0, Mvars))
    d0dr_proj = abs.(sum(k0 .* d0dr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    
    return (2 ./ tan.(θ) .* d0dr_proj .* dwdr .-  dωdr2_proj) ./ sin.(θ) .^ 2
end

function theta_B(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    Bvec, ωpL = GJ_Model_vec(x0, t_start, θm, ωPul, B0, rNS)
    return acos.(sum(k0 .* Bvec, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2) .* sum(Bvec.^2, dims=2)) )
end

function dθdr_proj(x0, k0, Mvars)
    d0dr_grd = grad(theta_B(seed(x0), k0, Mvars))
    return abs.(sum(k0 .* d0dr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
end

# just return net plasma freq
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

function GJ_Model_ωp_scalar(x, t, θm, ω, B0, rNS)
    # For GJ model, return \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]

    r = sqrt.(sum(x .* x))
    ϕ = atan.(x[2], x[1])
    θ = acos.( x[3]./ r)
    ψ = ϕ .- ω.*t
    ωp = (69.2e-6 .* sqrt.( abs.(3 .* cos.(θ).*( cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ) ) .- cos.(θm))).*
            sqrt.(B0./1e14 .* (rNS./r).^3 .* ω ./ (2 .* π )))
    return ωp
end

function GJ_Model_scalar(x, t, θm, ω, B0, rNS)
    # For GJ model, return \vec{B} and \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]

    r = sqrt.(sum(x .* x))
    ϕ = atan.(x[2], x[1])
    θ = acos.( x[3]./ r)
    ψ = ϕ .- ω.*t
    ωp = (69.2e-6 .* sqrt.( abs.(3 .* cos.(θ).*( cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ) ) .- cos.(θm))).*
            sqrt.(B0./1e14 .* (rNS./r).^3 .* ω ./ (2 .* π )))
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    
    Br = 2 .* Bnorm .* (cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θm) .* sin.(θ) .- sin.(θm) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θm) .* sin.(ψ)
    
    Bx = Br .* sin.(θ) .* cos.(ϕ) .+ Btheta .* cos.(θ) .* cos.(ϕ) .- Bphi .* sin.(ϕ)
    By = Br .* sin.(θ) .* sin.(ϕ) .+ Btheta .* cos.(θ) .* sin.(ϕ) .+ Bphi .* cos.(ϕ)
    Bz = Br .* cos.(θ) .- Btheta .* sin.(θ)


    # format: [e-, e+] last two -- plasma mass and gamma factor
    return [Bx By Bz], ωp
end

function rotate_alpha(x, v, Mvars)
    ctheta, surfNorm_alpha = surfNorm(x, v, Mvars, return_cos=false); # alpha
    surfNorm_alpha ./= sqrt.(sum(surfNorm_alpha.^2,  dims=2))
    # print(ctheta,"\t",acos.(ctheta), "\t", surfNorm_alpha, "\n" )
    rotAng = zeros(length(x[:,1]), 3)
    ang_new = sqrt.(rand(length(x[:, 1]))) .* π ./ 2
    ang_shift = zeros(length(x[:,1]))
    
    for i in 1:length(x[:,1])
        rotAng[i,:] .= [surfNorm_alpha[i,2].* v[i,3] .- surfNorm_alpha[i,3].* v[i,2], surfNorm_alpha[i,3].* v[i,1] .- surfNorm_alpha[i,1].* v[i,3], surfNorm_alpha[i,1].* v[i,2] .- surfNorm_alpha[i,2].* v[i,1] ]
        
        ang_shift[i] = (ang_new[i] .- acos.(ctheta[i]))
        # print(ang_shift[i], "\t", ang_new[i], "\t",acos.(ctheta[i]), "\n")
    end
    rotAng ./= sqrt.(sum(rotAng.^2,  dims=2))
    newV = zeros(length(x[:,1]), 3)
    newV[:, 1] .= (rotAng[:,1].^2 .+ cos.(ang_shift).*(rotAng[:,2].^2 .+ rotAng[:,3].^2)).*v[:,1] .+ (rotAng[:,1].*rotAng[:,2].*(1 .- cos.(ang_shift)) .- rotAng[:,3].*sin.(ang_shift)).*v[:,2] .+ (rotAng[:,1].*rotAng[:,3].*(1 .- cos.(ang_shift)) .+ rotAng[:,2].*sin.(ang_shift)).*v[:,3]
    newV[:, 2] .= (rotAng[:,1].*rotAng[:,2].*(1 .- cos.(ang_shift)) .+ sin.(ang_shift).*rotAng[:,3] ).*v[:,1] .+ (rotAng[:,2].^2 .+ (rotAng[:,3].^2 .+ rotAng[:,1].^2).*cos.(ang_shift)).*v[:,2] .+ (rotAng[:,2].*rotAng[:,3].*(1 .- cos.(ang_shift)) .- rotAng[:,1].*sin.(ang_shift)).*v[:,3]
    newV[:, 3] .= (rotAng[:,1].*rotAng[:,3].*(1 .- cos.(ang_shift)) .- sin.(ang_shift).* rotAng[:,2]).*v[:,1] .+ (rotAng[:,2].*rotAng[:,3].*(1 .- cos.(ang_shift)) .+ rotAng[:,1].*sin.(ang_shift)).*v[:,2] .+ (rotAng[:,3].^2 .+ (rotAng[:,2].^2 .+ rotAng[:,1].^2 ).*cos.(ang_shift)).*v[:,3]
  
    return newV
end

# roughly compute conversion surface, so we know where to efficiently sample
function Find_Conversion_Surface(Ax_mass, t_in, θm, ω, B0, rNS, gammaL, relativ)

    rayT = RayTracer;
    if θm < (π ./ 2.0)
        θmEV = θm ./ 2.0
    else
        θmEV = (θm .+ π) ./ 2.0
    end
    # estimate max dist
    om_test = GJ_Model_ωp_scalar(rNS .* [sin.(θmEV) 0.0 cos.(θmEV)], t_in, θm, ω, B0, rNS);
    rc_guess = rNS .* (om_test ./ Ax_mass) .^ (2.0 ./ 3.0);

    return rc_guess .* 1.01 # add a bit just for buffer
end

###
# ~~~ Energy as function of phase space parameters
###
function ωFree(x, k, t, θm, ωPul, B0, rNS, gammaF)
    # assume simple case where ωp proportional to r^{-3/2}, no time dependence, no magnetic field
    return sqrt.(sum(k .* k, dims = 2) .+ 1e-40 .* sqrt.(sum(x .* x, dims=2)) ./ (rNS.^ 2) )
end

function ωFixedp(x, k, t, θm, ωPul, B0, rNS, gammaF)
    # assume simple case where ωp proportional to r^{-3/2}, no time dependence, no magnetic field

    r = sqrt(sum(x .* x), dims = 2)
    ωp = 1e-6 * (rNS / r)^(3/2)
    k2 = sum(k.*k, dims = 2)

    return sqrt.(k2 .+ ωp^2)
end

function ωSimple(x, k, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, but no magnetic field

    ωpL = GJ_Model_ωp_vec(x, t, θm, ωPul, B0, rNS)
    return sqrt.(sum(k.*k, dims=2) .+ ωpL .^2)
end

function ωFixedp_BZ(x, k, t, θm, ωPul, B0, rNS, gammaF)
    #  ωp proportional to r^{-3/2}, Magnetic field in z direction (1/r^3 dependence), non-relativistic single species plasma

    rmag = sqrt.(sum(x .* x, dims=2))
    kmag = sqrt.(sum(k .* k, dims=2))
    Bvec = [0.0 0.0 B0 * (rNS / rmag)^3]
    ωp = 1e-6 .* (rNS ./ rmag) .^(3/2)
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    cθ = sum(k .* Bvec, dims=2) ./ (kmag .* Bmag)
    return sqrt(0.5 * (kmag .^2 .+ ωp .^2 .+ sqrt.(kmag .^4 .+ ωp .^4 .+ 2.0 .*(1.0 .- 2.0 .*cθ .^2) .*kmag .^2 .*ωp .^2)))
end

function ωNR_BZ(x, k, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, Magnetic field in z direction (1/r^3 dependence), non-relativistic single species plasma

    Bvec, ωpL = GJ_Model_vec(x, t, θm, ωPul, B0, rNS)
    rmag = sqrt.(sum(x .* x, dims=2))
    kmag = sqrt.(sum(k .* k, dims=2))
    Bvec = [0.0 0.0 B0 * (rNS ./ rmag).^3]
    Bmag = sqrt(sum(Bvec .* Bvec), dims=2)
    ωp = sqrt.(sum(ωpL .* ωpL, dims=2))
    cθ = sum(k .* Bvec, dims=2) ./ (kmag .* Bmag)
    return sqrt(0.5 .* (kmag .^2 .+ ωp .^2 .+ sqrt.(kmag .^4 .+ ωp .^4 .+ 2.0 .*(1.0 .- 2.0 .*cθ .^2) .*kmag .^2 .*ωp .^2)))
end



function ωNR_e(x, k, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, Magnetic field, non-relativstic e only
    Bvec, ωpL = GJ_Model_vec(x, t, θm, ωPul, B0, rNS)

    kmag = sqrt.(sum(k .* k, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    ωp = sqrt.(sum(ωpL .* ωpL, dims=2))

    cθ = sum(k .* Bvec, dims=2) ./ (kmag .* Bmag)
    
    cθTerm = (1.0 .- 2.0 .* cθ.^2)
    # print(cθTerm)
    # abs not necessary, but can make calculation easier
    return sqrt.(abs.(0.5 .* (kmag .^2 + ωp .^2 + sqrt.(abs.(kmag .^4 + ωp .^4 + 2.0 .* cθTerm .*kmag .^2 .* ωp .^2 )))))

end


function dk_dl(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    ωErg = ω(x0, k0, t_start, θm, ωPul, B0, rNS, gammaF)
    dkdr_grd = grad(kNR_e(seed(x0), k0, ωErg, t_start, θm, ωPul, B0, rNS, gammaF))
    
    dkdr_proj = abs.(sum(k0 .* dkdr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    return dkdr_proj
end

function dk_ds(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    ωErg = ω(x0, k0, t_start, θm, ωPul, B0, rNS, gammaF)
    dkdr_grd = grad(kNR_e(seed(x0), k0, ωErg, t_start, θm, ωPul, B0, rNS, gammaF))
    Bvec, ωpL = GJ_Model_vec(x0, t_start, θm, ωPul, B0, rNS)
    
    kmag = sqrt.(sum(k0 .* k0, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    cθ = sum(k0 .* Bvec, dims=2) ./ (kmag .* Bmag)
    khat = k0 ./ kmag
    Bhat = Bvec ./ Bmag
    
    dkdr_proj_s = zeros(length(k0[:, 1]));
    for i in 1:length(k0[:,1])
        uvec = [k0[i,2] .* Bvec[i,3] - Bvec[i,2] .* k0[i,3],  k0[i,3] .* Bvec[i,1] - Bvec[i,3] .* k0[i,1], k0[i,1] .* Bvec[i,2] - Bvec[i,1] .* k0[i,2]] ./ Bmag[i] ./ kmag[i]
        uhat = uvec ./ sqrt.(sum(uvec .^ 2));
        R = [uhat[1].^2 uhat[1] .* uhat[2] .+ uhat[3] uhat[1] .* uhat[3] .- uhat[2]; uhat[1] .* uhat[2] .- uhat[3] uhat[2].^2 uhat[2] .* uhat[3] .+ uhat[1]; uhat[1].*uhat[3] .+ uhat[2] uhat[2].*uhat[3] .- uhat[1] uhat[3].^2];
        shat = R * Bhat[i, :];
        dkdr_proj_s[i] = abs.(sum(shat .* dkdr_grd[i]));
        
        # print("test 1 \t", sum(shat .* Bhat[i,:]), "\t",   sum(shat .* uhat), "\t",  sum(shat .* khat[i,:]) , "\t",  sum(shat2 .* Bhat[i,:]), "\t",   sum(shat2 .* uhat),"\t", sum(shat2 .* khat[i,:]), "\n")
    end
    # dkdr_proj = abs.(sum(k0 .* dkdr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    return dkdr_proj_s
end

function dk_dlS(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    ωErg = ω(x0, k0, t_start, θm, ωPul, B0, rNS, gammaF)
    dkdr_grd = grad(ksimple(seed(x0), k0, ωErg, t_start, θm, ωPul, B0, rNS, gammaF))
    
    dkdr_proj = abs.(sum(k0 .* dkdr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    return dkdr_proj
end


function kNR_e(x, k, ω, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, Magnetic field, non-relativstic e only
    Bvec, ωpL = GJ_Model_vec(x, t, θm, ωPul, B0, rNS)

    kmag = sqrt.(sum(k .* k, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    ωp = sqrt.(sum(ωpL .* ωpL, dims=2))

    cθ = sum(k .* Bvec, dims=2) ./ (kmag .* Bmag)
    
    #ω = ωNR_e(x, k, t, θm, ωPul, B0, rNS, gammaF)
    
    # abs not necessary, but can make calculation easier (numerical error can creap at lvl of 1e-16 for ctheta)
    return sqrt.(((ω.^2 .-  ωp.^2) ./ (1 .- cθ.^2 .* ωp.^2 ./ ω.^2)))
end

function ksimple(x, k, ω, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, Magnetic field, non-relativstic e only
    Bvec, ωpL = GJ_Model_vec(x, t, θm, ωPul, B0, rNS)

    ωp = sqrt.(sum(ωpL .* ωpL, dims=2))
    # abs not necessary, but can make calculation easier (numerical error can creap at lvl of 1e-16 for ctheta)
    return sqrt.((ω.^2 .-  ωp.^2))
end

function ωGam(x, k, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, Magnetic field, thermal single species plasma (assume thermal is first species!)

    Bvec, ωpL = GJ_Model_vec(x, t, θm, ωPul, B0, rNS)
    gam = gammaF[1]

    kmag = sqrt.(sum(k .* k, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    ωp = sqrt.(sum(ωpL .* ωpL, dims=2))

    cθ = sum(k .* Bvec, dims=2) ./ (kmag .* Bmag)
    sqr_fct = sqrt.(kmag.^4 .* (gam .^2 .- cθ.^2 .* (gam.^2 .- 1)).^2 .-
        2 .* kmag .^2 .* gam .* (cθ.^2 .+ (cθ.^2 .- 1) .* gam.^2) .* ωp.^2 .+ gam.^2 .* ωp.^4)
    ω_final = sqrt.((kmag.^2 .*(gam.^2 .+ cθ.^2 .*(gam.^2 .- 1)) .+ gam.*ωp.^2 .+ sqr_fct) ./ (2 .* gam.^2))

    return ω_final
end

function find_samples(maxR, ntimes_ax, θm, ωPul, B0, rNS, Mass_a, Mass_NS; period_average=false)
    batchsize = 2;
    if period_average
        times_eval = rand(batchsize) .* (2π ./ ωPul)
    else
        times_eval = zeros(batchsize)
    end


    tt_ax = LinRange(-2*maxR, 2*maxR, ntimes_ax); # Not a real physical time -- just used to get trajectory crossing

    # randomly sample angles θ, ϕ
    θi = acos.(1.0 .- 2.0 .* rand(batchsize));
    ϕi = rand(batchsize) .* 2π;

    vvec_all = [sin.(θi) .* cos.(ϕi) sin.(θi) .* sin.(ϕi) cos.(θi)];
    # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
    ϕRND = rand(batchsize) .* 2π;
    # rRND = sqrt.(rand(batchsize)) .* maxR; standard flat sampling
    rRND = rand(batchsize) .* maxR; # New 1/r sampling
    x1 = rRND .* cos.(ϕRND);
    x2 = rRND .* sin.(ϕRND);
    # rotate using Inv[EurlerMatrix(ϕi, θi, 0)] on vector (x1, x2, 0)
    x0_all= [x1 .* cos.(-ϕi) .* cos.(-θi) .+ x2 .* sin.(-ϕi) x2 .* cos.(-ϕi) .- x1 .* sin.(-ϕi) .* cos.(-θi) x1 .* sin.(-θi)];
    x_axion = [transpose(x0_all[i,:]) .+ transpose(vvec_all[i,:]) .* tt_ax[:] for i in 1:batchsize];

    cxing_st = [get_crossings(log.(GJ_Model_ωp_vec(x_axion[i], times_eval[i], θm, ωPul, B0, rNS)) .- log.(Mass_a)) for i in 1:batchsize];
    cxing = [apply(cxing_st[i], tt_ax) for i in 1:batchsize];
    # see if any crossings
    indx_cx = [if length(cxing[i]) .> 0 i else -1 end for i in 1:batchsize];
    # remove those which dont
    indx_cx_cut = indx_cx[indx_cx .> 0];
    # assign index for random point selection
    randInx = [rand(1:length(cxing[indx_cx_cut][i])) for i in 1:length(indx_cx_cut)];
    cxing_short = [cxing[indx_cx_cut][i][randInx[i]] for i in 1:length(indx_cx_cut)];
    weights = [length(cxing[indx_cx_cut][i]) for i in 1:length(indx_cx_cut)];

    times_eval = times_eval[indx_cx .> 0]

    numX = length(cxing_short);
    # print(cxing, "\t", cxing_short, "\t", indx_cx_cut, "\t", randInx, "\n")

    if numX != 0
        # print(x0_all, "\t", cxing_short, "\t", indx_cx_cut, "\t", randInx,"\n")
        xpos = [transpose(x0_all[indx_cx_cut[i], :]) .+ transpose(vvec_all[indx_cx_cut[i], :]) .* cxing_short[i] for i in 1:numX];
        vvec_full = [transpose(vvec_all[indx_cx_cut[i],:]) .* ones(1, 3) for i in 1:numX];

        R_sample = vcat([rRND[indx_cx_cut][i] for i in 1:numX]...);
        # print(x0_all, "\t", xpos, "\t", vvec_full, "\t", R_sample, "\n")
        t_new_arr = LinRange(- abs.(tt_ax[3] - tt_ax[1]), abs.(tt_ax[3] - tt_ax[1]), 100);
        xpos_proj = [xpos[i] .+ vvec_full[i] .* t_new_arr[:] for i in 1:numX];

        cxing_st = [get_crossings(log.(GJ_Model_ωp_vec(xpos_proj[i], times_eval[i], θm, ωPul, B0, rNS)) .- log.(Mass_a)) for i in 1:numX];
        cxing = [apply(cxing_st[i], t_new_arr) for i in 1:numX];
        indx_cx = [if length(cxing[i]) .> 0 i else -1 end for i in 1:numX];
        indx_cx_cut = indx_cx[indx_cx .> 0];
        R_sample = R_sample[indx_cx_cut];
        times_eval = times_eval[indx_cx .> 0]

        numX = length(indx_cx_cut);

        randInx = [rand(1:length(cxing[indx_cx_cut][i])) for i in 1:numX];
        cxing = [cxing[indx_cx_cut][i][randInx[i]] for i in 1:numX];

        xpos = [xpos[indx_cx_cut,:][i] .+ vvec_full[indx_cx_cut,:][i] .* cxing[indx_cx_cut][i] for i in 1:numX];

        xpos_flat = reduce(vcat, xpos);
        vvec_flat = reduce(vcat, vvec_full);

        rmag = sqrt.(sum(xpos_flat .^ 2, dims=2));
        indx_r_cut = rmag .> rNS;
        if sum(indx_r_cut) - length(xpos_flat[:,1 ]) < 0
            xpos_flat = xpos_flat[indx_r_cut[:], :]
            vvec_flat = vvec_flat[indx_r_cut[:], :]
            R_sample = R_sample[indx_r_cut[:]]
            numX = length(xpos_flat);
            rmag = sqrt.(sum(xpos_flat .^ 2, dims=2));
            times_eval = times_eval[indx_r_cut[:]]
        end

        ntrajs = length(R_sample)

        if ntrajs == 0
            return 0.0, 0.0, 0, 0.0, times_eval
        end

        # print("here...\t", xpos_flat, "\t", times_eval, "\n")
        ωpL = GJ_Model_ωp_vec(xpos_flat, times_eval, θm, ωPul, B0, rNS)
        vmag = sqrt.(2 * 132698000000.0 .* Mass_NS ./ rmag) ; # km/s
        erg_ax = sqrt.( Mass_a^2 .+ (Mass_a .* vmag / 2.998e5) .^2 );

        # make sure not in forbidden region....
        fails = ωpL .> erg_ax;
        n_fails = sum(fails);
        if n_fails > 0

            ωpLi2 = [if fails[i] == 1 Mass_a .- GJ_Model_ωp_vec(transpose(xpos_flat[i,:]) .+ transpose(vvec_flat[i,:]) .* t_new_arr[:], times_eval[i], θm, ωPul, B0, rNS) else -1 end for i in 1:ntrajs];


            t_new = [if length(ωpLi2[i]) .> 1 t_new_arr[findall(x->x==ωpLi2[i][ωpLi2[i] .> 0][argmin(ωpLi2[i][ωpLi2[i] .> 0])], ωpLi2[i])][1] else -1e6 end for i in 1:length(ωpLi2)];
            t_new = t_new[t_new .> -1e6];
            xpos_flat[fails[:],:] .+= vvec_flat[fails[:], :] .* t_new;

        end

        return xpos_flat, R_sample, ntrajs, weights, times_eval
    else
        return 0.0, 0.0, 0, 0.0, times_eval
    end

end



end



function main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, Ntajs, gammaF, batchsize; ode_err=1e-5, maxR=Nothing, cutT=10, fix_time=Nothing, CLen_Scale=true, file_tag="", ntimes=1000, v_NS=[0 0 0], errSlve=1e-10, period_average=false, M_MC=1e-12, R_MC=1.0e9,  save_more=true, vmean_ax=220.0, ntimes_ax=10000, dir_tag="results", trace_trajs=false)

    # pass parameters
    # axion mass [eV], axion-photon coupling [1/GeV], misalignment angle (rot-B field) [rad], rotational freq pulars [1/s]
    # magnetic field strengh at surface [G], radius NS [km], mass NS [solar mass], dispersion relations
    # number of axion trajectories to generate
    # maxR should be nuumber < 1 used for efficient sampling...
    # vmean_ax = 220.0; # km/s mean velocity of axion in rest frame galaxy

    # rhoDM -- this is asymptotic density. currently scaled to 1 GeV / cm^3


    RT = RayTracer; # define ray tracer module

    func_use = RT.ωNR_e

    NS_vel = [sin.(NS_vel_T) 0.0 cos.(NS_vel_T)] .* NS_vel_M;


    # one time run --- time can be reduced in RayTracer.jl file by reducing sampling
    # cannot run this with a non-GJ model for time being (but works for relativistic plasmas as well)....
    maxR = RT.Find_Conversion_Surface(Mass_a, fix_time, θm, ωPul, B0, rNS, 1, false)

    maxR_tag = "";

    if maxR < rNS
        print("Too small Max R.... quitting.... \n")
        omegaP_test = RT.GJ_Model_ωp_scalar(rNS .* [sin.(θm) 0.0 cos.(θm)], 0.0, θm, ωPul, B0, rNS);
        print("Max omegaP found... \t", omegaP_test, "Max radius found...\t", maxR, "\n")
        return
    end

    photon_trajs = 1
    desired_trajs = Ntajs
    # assumes desired_trajs large!
    save_more=true;
    SaveAll = zeros(desired_trajs * 2, 21);
    f_inx = 0;


    tt_ax = LinRange(-2*maxR, 2*maxR, ntimes_ax); # Not a real physical time -- just used to get trajectory crossing
    t_diff = tt_ax[2] - tt_ax[1];
    tt_ax_zoom = LinRange(-2*t_diff, 2*t_diff, ntimes_ax);

    ln_t_start = -22;
    ln_t_end = log.(1 ./ ωPul);

    NumerPass = [ln_t_start, ln_t_end, ode_err];
    ttΔω = exp.(LinRange(ln_t_start, ln_t_end, ntimes));

    if fix_time != Nothing
        file_tag *= "_fixed_time_"*string(fix_time);
    end

    file_tag *= "_odeErr_"*string(ode_err);

    file_tag *= "_vxNS_"*string(v_NS[1]);
    file_tag *= "_vyNS_"*string(v_NS[2]);
    file_tag *= "_vzNS_"*string(v_NS[3]);
    if (v_NS[1] == 0)&&(v_NS[1] == 0)&&(v_NS[1] == 0)
        phaseApprox = true;
    else
        phaseApprox = false;
    end
    vNS_mag = sqrt.(sum(v_NS.^2));
    if vNS_mag .> 0
        vNS_theta = acos.(v_NS[3] ./ vNS_mag);
        vNS_phi = atan.(v_NS[2], v_NS[1]);
    end

    vGu = 0.3 # guess point

    # define time at which we find crossings
    t0_ax = zeros(batchsize);
    xpos_flat = zeros(batchsize, 3);
    R_sample = zeros(batchsize);
    mcmc_weights = zeros(batchsize);
    times_pts = zeros(batchsize);
    filled_positions = false;
    fill_indx = 1;


    while photon_trajs < desired_trajs

        while !filled_positions

            xv, Rv, numV, weights, times_eval = RT.find_samples(maxR, ntimes_ax, θm, ωPul, B0, rNS, Mass_a, Mass_NS, period_average=period_average)
            # print(xv, "\t", times_eval, "\n")
            # count sample
            f_inx += 1;

            if numV == 0
                continue
            end

            # for i in 1:1
            for i in 1:numV # Keep more?
                if fill_indx <= batchsize

                    xpos_flat[fill_indx, :] .= xv[i, :];
                    R_sample[fill_indx] = Rv[i];
                    mcmc_weights[fill_indx] = weights[i];
                    times_pts[fill_indx] = times_eval[i];
                    fill_indx += 1
                end
            end

            if fill_indx > batchsize
                filled_positions = true
                fill_indx = 1
            end
        end
        filled_positions = false;

        rmag = sqrt.(sum(xpos_flat .^ 2, dims=2));
        ϕ = atan.(view(xpos_flat, :, 2), view(xpos_flat, :, 1))
        θ = acos.(view(xpos_flat, :, 3)./ rmag)

        # add random vel dispersion to NS_vel

        vel_disp = sqrt.(GNew .* M_MC ./ R_MC) ./ c_km # km
        vel = zeros(length(rmag)*2, 3)
        for i in 1:length(rmag)

            NS_vel_guess = NS_vel .+ erfinv.(2 .* rand(1, 3) .- 1.0) .* vel_disp
            velV = RT.solve_vel_CS(θ[i], ϕ[i], rmag[i], NS_vel_guess, guess=[vGu vGu vGu], errV=errSlve)
            velV2 = RT.solve_vel_CS(θ[i], ϕ[i], rmag[i], NS_vel_guess, guess=[-vGu -vGu -vGu], errV=errSlve)
            vel[i, :] = velV
            vel[i+length(rmag), :] = velV2
        end
        # stack results
        xpos_stacked = cat(xpos_flat, xpos_flat, dims=1)
        rmag = cat(rmag, rmag, dims=1)

        R_sampleFull = cat(R_sample, R_sample, dims=1)
        t0_full = cat(times_pts, times_pts, dims=1)
        mcmc_weightsFull = cat(mcmc_weights, mcmc_weights, dims=1)


        vmag = sqrt.(2 * 132698000000.0 .* Mass_NS ./ rmag) ; # km/s
        erg_ax = sqrt.( Mass_a^2 .+ (Mass_a .* vmag / 2.998e5) .^2 );
        ωpL = RT.GJ_Model_ωp_vec(xpos_stacked, t0_full, θm, ωPul, B0, rNS)
        newV = vel ./ sqrt.(sum(vel.^2, dims=2))

        calpha = RT.surfNorm(xpos_stacked, newV, [func_use, [θm, ωPul, B0, rNS, gammaF, t0_full, Mass_NS]], return_cos=true); # alpha
        weight_angle = abs.(calpha);

        Bvec, ωp = RT.GJ_Model_vec(xpos_stacked, t0_full, θm, ωPul, B0, rNS);
        Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))

       
        cθ = sum(newV .* Bvec, dims=2) ./ Bmag
        vIfty = erfinv.(2 .* rand(length(vmag), 3) .- 1.0) .* vmean_ax .+ v_NS # km /s
        
        vel_eng = sum((vIfty ./ 2.998e5).^ 2, dims = 2) ./ 2;
        vmag_tot = sqrt.(vmag .^ 2 .+ sum(vIfty .^ 2, dims = 2) ); # km/s
        k_init_ax = Mass_a .* newV .* (vmag_tot ./ 2.998e5);

        erg_ax = sqrt.(sum(k_init_ax .^2, dims=2) .+ Mass_a .^2);

        k_init = sqrt.(erg_ax .^2 .-  ωp .^ 2) .* newV ./ sqrt.(1.0 .- cθ .^ 2 .* ωp .^ 2 ./ erg_ax .^2);


        B_tot = sqrt.(sum(Bvec .^ 2, dims=2)) .* (1.95e-20) ; # GeV^2
        MagnetoVars =  [θm, ωPul, B0, rNS, gammaF, t0_full]
        

        sln_δk = RT.dk_ds(xpos_stacked, k_init, [func_use, MagnetoVars]);
        conversion_F = sln_δk ./  (6.58e-16 .* 2.998e5) # 1/km^2;

        # pref1 = 1 .+ (ωp.^4 .* cθ.^2 .* sin.(acos.(cθ)) .^2) ./ (Mass_a.^2 .* (1 .+ vmag_tot.^2) .- ωp.^2 .* cθ.^2).^2;
        # pref2 = (ωp.^2 .* cθ.^2) ./ (Mass_a.^2 .*  (1 .+ vmag_tot.^2)) .- 1;
        # Prob = π ./ 2.0 .* (1e-12 .* B_tot).^2 .* (1 .+ vmag_tot.^2) .* sin.(acos.(cθ)) .^2 .* pref1  .* cLen ./ (vmag_tot .^2 .* pref2.^2) ./ (2.998e5 .* 6.58e-16 ./ 1e9).^2 ; # g [1e-12 GeV^-1], unitless
        
        Prob = π ./ 2 .* (Ax_g .* B_tot ./ sin.(acos.(cθ))) .^2 ./ conversion_F .* (1e9 .^2) ./ (vmag_tot ./ 2.998e5) .^2 ./ ((2.998e5 .* 6.58e-16) .^2) ./ sin.(acos.(cθ)).^2; #unitless

        phaseS = (π .* maxR .* R_sampleFull .* 2) .* 1.0 .* Prob ./ Mass_a .* 1e9 .* (1e5).^3  # 1 / km
        if trace_trajs
            nsteps = 1000;
            ln_tstart=-15;
            ln_tend=5;
            ode_err=1e-10;
            NumerP = [ln_tstart, ln_tend, ode_err]
            xF_AX, vF_AX = RT.propagateAxion(xpos_stacked, newV, nsteps, NumerP);
        end
        density_enhancement = 2 ./ sqrt.(π) .* vmag ./ vel_disp # unitless
        
        sln_prob = weight_angle .* phaseS .* density_enhancement .* 2.998e5 ; # photons / second

        
        sln_k = k_init;
        sln_x = xpos_stacked;
        sln_vInf = vel_eng ;
        sln_t = t0_full;
        sln_ConVL = sqrt.(π ./ conversion_F);

        #--------------------------------
        # First we batch together some axion trajectories
        # Now we loop compute the photon trajectories for all loops
        #--------------------------------
        # with axion trajectory (and crossing points) in hand, follow photon trajectory

        MagnetoVars = [θm, ωPul, B0, rNS, gammaF, sln_t] # θm, ωPul, B0, rNS, gamma factors, Time = 0
        xF, kF, tF = RT.propagate(func_use, sln_x, sln_k, ntimes, MagnetoVars, NumerPass);

        
        
        ϕf = atan.(view(kF, :, 2, ntimes), view(kF, :, 1, ntimes));
        ϕfX = atan.(view(xF, :, 2, ntimes), view(xF, :, 1, ntimes));
        θf = acos.(view(kF, :, 3, ntimes) ./ sqrt.(sum(view(kF, :, :, ntimes) .^2, dims=2)));
        θfX = acos.(view(xF, :, 3, ntimes) ./ sqrt.(sum(view(xF, :, :, ntimes) .^2, dims=2)));

        # compute energy dispersion (ωf - ωi) / ωi
        passA = [func_use, MagnetoVars];
        # Δω = (RT.dwdt_vec(xF, kF, ttΔω, passA) ) ./ Mass_a .+ sln_vInf[:];
        Δω = tF[:, end] ./ Mass_a .+ sln_vInf[:];
        
        
        opticalDepth = RT.tau_cyc(xF, kF, ttΔω, passA, Mass_a);
        
        num_photons = length(ϕf)
        passA2 = [func_use, MagnetoVars, Mass_a];
        if CLen_Scale
            weightC = ConvL_weights(xF, kF, vmag_tot ./ 2.998e5, ttΔω, sln_ConVL, passA2)
        else
            weightC = ones(num_photons)
        end
        
        # cut out high prob conversions
        weightC[weightC[:] .> 1.0] .= 0.0;
        
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 1] .= view(θf, :);
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 2] .= view(ϕf,:);
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 3] .= view(θfX, :);
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 4] .= view(ϕfX, :);
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 5] .= sqrt.(sum(xF[:, :, end] .^2, dims=2))[:]; # r final
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 6] .= sln_prob[:] .* weightC .^ 2 .* exp.(-opticalDepth[:]); #  num photons / second
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 7] .= Δω[:];
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 8] .= sln_ConVL[:];
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 9] .= sln_x[:, 1]; # initial x
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 10] .= sln_x[:, 2]; # initial y
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 11] .= sln_x[:, 3]; # initial z
        
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 12] .= sln_k[:, 1]; # initial kx
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 13] .= sln_k[:, 2]; # initial ky
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 14] .= sln_k[:, 3]; # initial kz
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 15] .= opticalDepth[:]; # optical depth
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 16] .= weightC[:]; # optical depth
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 17] .= Prob[:]; # optical depth
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 18] .= calpha[:]; # surf norm

        if trace_trajs
            SaveAll[photon_trajs:photon_trajs + num_photons - 1, 19] .= xF_AX[:, 1, end]; #
            SaveAll[photon_trajs:photon_trajs + num_photons - 1, 20] .= xF_AX[:, 2, end]; #
            SaveAll[photon_trajs:photon_trajs + num_photons - 1, 21] .= xF_AX[:, 3, end]; #
        end

        photon_trajs += num_photons;

    end

    # cut out unused elements
    SaveAll = SaveAll[SaveAll[:,6] .> 0, :];
    SaveAll[:,6] ./= float(f_inx); # divide off by N trajectories sampled

    fileN = dir_tag*"/Minicluster__MassAx_"*string(Mass_a);
    fileN *= "_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0)*"_rNS_";
    fileN *= string(rNS)*"_MassNS_"*string(Mass_NS);
    fileN *= "_NS_Mag_"*string(round(NS_vel_M, digits=5))*"_NS_Theta_"*string(round(NS_vel_T, digits=3))
    fileN *= "_Mmc_"*string(M_MC)*"_Rmc_"*string(R_MC)*"_"
    fileN *= "_"*file_tag*"_.npz";
    npzwrite(fileN, SaveAll)
end

function dist_diff(xfin)
    b = zeros(size(xfin[:,1,:]))
    b[:, 1:end-1] = abs.((sqrt.(sum(xfin[:, :, 2:end] .^ 2, dims=2)) .- sqrt.(sum(xfin[:, :, 1:end-1] .^ 2, dims=2)) )) ./ 2.998e5 ./ 6.58e-16 # 1 / eV
    b[end] = b[end-2]
    return b
end

function ConvL_weights(xfin, kfin, v_sur, tt, conL, Mvars)
    # xfin and kfin from runner
    # tt from time list, conL [km], and vel at surface (unitless)
    func_use, MagnetoVars, Mass_a = Mvars
    RT = RayTracer
    
    dR = dist_diff(xfin)
    ntimes = length(tt)
    nph = length(xfin[:,1,1])
    phaseS = zeros(nph, ntimes)
    # phaseS_old = zeros(nph, ntimes)
    for i in 1:ntimes
        
        Bvec, ωpL = RT.GJ_Model_vec(xfin[:,:,i], tt[i], MagnetoVars[1], MagnetoVars[2], MagnetoVars[3], MagnetoVars[4])
        thetaB_hold = sum(kfin[:,:,i] .* Bvec, dims=2) ./ sqrt.(sum(Bvec .^ 2, dims=2) .* sum(kfin[:,:,i] .^ 2, dims=2))
        if sum(thetaB_hold .> 1) > 0
            thetaB_hold[thetaB_hold .> 1] .= 1.0;
        end
        thetaB = acos.(thetaB_hold)
        thetaZ_hold = sqrt.( sum(kfin[:,:,i] .* kfin[:,:,1], dims=2) .^2 ./ sum(kfin[:,:,1] .^ 2, dims=2) ./ sum(kfin[:,:,i] .^ 2, dims=2))
        if sum(thetaZ_hold .> 1) > 0
            thetaZ_hold[thetaZ_hold .> 1] .= 1.0;
        end
        thetaZ = acos.(thetaZ_hold)
        ωfull = func_use(xfin[:, :, i], kfin[:, :, i], tt[i], MagnetoVars[1], MagnetoVars[2], MagnetoVars[3], MagnetoVars[4], MagnetoVars[5])
        # dePhase_Factor = cos.(thetaZ) .- ωpL .^2 ./ ωfull .^2 .* sin.(thetaZ) .*  sin.(thetaB).^2 ./ tan.(thetaB) ./ (1 .- cos.(thetaB).^2 .* ωpL .^2 ./ ωfull .^2);
        # phaseS[:, i] = dR[:, i] .* (Mass_a .* v_sur .- dePhase_Factor .* sqrt.( abs.((ωfull .^2 .-  ωpL .^ 2 ) ./ (1 .- cos.(thetaB) .^2 .* ωpL .^2 ./ ωfull .^2)))  )
        phaseS[:, i] = dR[:, i]  .* (Mass_a .* v_sur .-  cos.(thetaZ) .* sqrt.( abs.((ωfull .^2 .-  ωpL .^ 2 ) ./ (1 .- cos.(thetaB) .^2 .* ωpL .^2 ./ ωfull .^2)))  )
        # if i == 1
        #     print(thetaB,"\t", ωpL, "\t", ωfull, "\n")
        # end
        # print("Here...\t", i, "\t", phaseS[:, i], "\t", phaseS_old[:, i], "\n")
    end
    # print("\n\n")
    δphase = cumsum(phaseS, dims=2)
    weights = zeros(nph)
    for i in 1:nph
        if abs.(δphase[i,1]) .> (π ./ 2)
            weights[i] = 0.0
        else
            cx_list = RT.get_crossings(abs.(δphase[i,:]) .- π ./ 2 )
            convL_real = sum(dR[i, 1:cx_list.i2[1]]) .* 6.58e-16 .* 2.998e5

            weights[i] = convL_real[1] ./ conL[i]
        end
        # print(weights[i], "\n")
    end
    return weights
end