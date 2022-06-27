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
using LinearAlgebra: cross, det
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
        du[:,1:3] = dωdk .* t .* c_km;
        du[:,4:6] = -1.0 .* dωdx .* t  .* c_km;
        du[:, 7] = dωdt .* t;
        
    end
end

function func_axion!(du, u, Mvars, lnt)
    @inbounds begin
        t = exp.(lnt)
        x = u[:,1:3]
        v = u[:,4:6]

        r = sqrt.(sum(x .* x, dims=2))

        xhat = x  ./ r
        
        mass_eff = ones(length(r))
#        mass_eff[r .< 10] .= 1.0 .* (r[r .< 10] ./ 10).^3
        du[:,1:3] = -v .* t ;  # v is km/s, x in km, t [s]
        du[:,4:6] = GNew .* mass_eff ./ r.^2 .* xhat .* t; # units km/s/s, assume 1M NS

    end
end

function func_axionBACK!(du, u, Mvars, t)
    @inbounds begin
        x = u[:,1:3]
        v = u[:,4:6]

        r = sqrt.(sum(x .* x, dims=2))

        xhat = x  ./ r
        
        mass_eff = ones(length(r))
#        mass_eff[r .< 10] .= 1.0 .* (r[r .< 10] ./ 10).^3
        du[:,1:3] = v ;  # v is km/s, x in km, t [s]
        du[:,4:6] = -GNew .* mass_eff ./ r.^2 .* xhat; # units km/s/s, assume 1M NS

        
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
    # tstart, tend, ode_err = NumerP
    # tspan = (tstart, tend)
    saveat = (tspan[2] - tspan[1]) / (nsteps-1)

    # u0 = cu([x0 k0])
    u0 = ([x0 k0])

    function condition(u, t, integrator) # Event when event_f(u,t) == 0
        hold = sqrt.(sum(u[:, 1:3].^2, dims=2)) .- 10.0
        # print(hold)
        return minimum(hold)
    end
    function affect!(int)
        if (int.t-int.tprev) .> 1e-12
            set_proposed_dt!(int,(int.t-int.tprev)/100)
        end
    end
    cb = ContinuousCallback(condition, affect!)
    probAx = ODEProblem(func_axion!, u0, tspan, [ln_tstart], reltol=ode_err, abstol=ode_err, callback=cb, maxiters=1e7, dtmin=1e-13, force_dtmin=true);
    # probAx = ODEProblem(func_axion!, u0, tspan, [tstart], reltol=ode_err, abstol=1e-20, maxiters=1e7);
    # sol = solve(probAx, Tsit5(), saveat=saveat);
    sol = solve(probAx, Vern7(), saveat=saveat)
    # sol = solve(probAx, lsoda(), saveat=saveat)

    x = cat([u[:, 1:3] for u in sol.u]...,dims=3);
    v = cat([u[:, 4:6] for u in sol.u]...,dims=3);

    return x, v

end

function propagateAxion_backwards(x0::Matrix, k0::Matrix, nsteps::Int, saveT::Array, NumerP::Array)
    
    tstart, tend, ode_err = NumerP
    tspan = (tstart, tend)
    saveat = saveT

    # u0 = cu([x0 k0])
    u0 = ([x0 k0])

    
    probAx = ODEProblem(func_axionBACK!, u0, tspan, [tstart], reltol=ode_err, abstol=1e-13, maxiters=1e7);
    # sol = solve(probAx, Tsit5(), saveat=saveat);
    # sol = solve(probAx, Vern7(), saveat=saveat, dtmax=)
    sol = solve(probAx, lsoda(), saveat=saveat)

    
    
    x = cat([u[:, 1:3] for u in sol.u]..., dims=3)
    v = cat([u[:, 4:6] for u in sol.u]..., dims=3);

    return x, v

end

function solve_vel_CS(θ, ϕ, r, NS_vel; guess=[0.1 0.1 0.1], errV=1e-24, Mass_NS=1)
    ff = sum(NS_vel.^2); # unitless
    
    GMr = GNew .* Mass_NS ./ r ./ (c_km .^ 2); # unitless
    rhat = [sin.(θ) .* cos.(ϕ) sin.(θ) .* sin.(ϕ) cos.(θ)]

    function f!(F, x)
        vx, vy, vz = x

        denom = ff .+ GMr .- sqrt.(ff) .* sum(x .* rhat);


        F[1] = (ff .* vx .+ sqrt.(ff) .* GMr .* rhat[1] .- sqrt.(ff) .* vx .* sum(x .* rhat)) ./ (NS_vel[1] .* denom) .- 1.0
        F[2] = (ff .* vy .+ sqrt.(ff) .* GMr .* rhat[2] .- sqrt.(ff) .* vy .* sum(x .* rhat)) ./ (NS_vel[2] .* denom) .- 1.0
        F[3] = (ff .* vz .+ sqrt.(ff) .* GMr .* rhat[3] .- sqrt.(ff) .* vz .* sum(x .* rhat)) ./ (NS_vel[3] .* denom) .- 1.0
        # print(F[1], "\t",F[2], "\t", F[3],"\n")
        # print(θ, "\t", ϕ,"\t", r, "\n")
    end


    soln = nlsolve(f!, guess, autodiff = :forward, ftol=errV, iterations=10000)

    FF = zeros(3)
    f!(FF, soln.zero)
    accur = sqrt.(sum(FF.^2))
    # print("accuracy... ", FF,"\n")
    return soln.zero, accur
end

function jacobian_fv(x_in, vel_loc)

    rmag = sqrt.(sum(x_in.^2));
    ϕ = atan.(x_in[2], x_in[1])
    θ = acos.(x_in[3] ./ rmag)
    
    
    dvXi_dV = grad(v_infinity(θ, ϕ, rmag, seed(vel_loc), v_comp=1));
    dvYi_dV = grad(v_infinity(θ, ϕ, rmag, seed(vel_loc), v_comp=2));
    dvZi_dV = grad(v_infinity(θ, ϕ, rmag, seed(vel_loc), v_comp=3));
    
    JJ = det([dvXi_dV; dvYi_dV; dvZi_dV])
    
    return abs.(JJ).^(-1)
end

function v_infinity(θ, ϕ, r, vel_loc; v_comp=1, Mass_NS=1)
    vx, vy, vz = vel_loc
    vel_loc_mag = sqrt.(sum(vel_loc.^2))
    GMr = GNew .* Mass_NS ./ r ./ (c_km .^ 2); # unitless

    v_inf = sqrt.(vel_loc_mag.^2 .- (2 .* GMr)); # unitless
    rhat = [sin.(θ) .* cos.(ϕ) sin.(θ) .* sin.(ϕ) cos.(θ)]
    
    
    denom = v_inf.^2 .+ GMr .- v_inf .* sum(vel_loc .* rhat);
    if v_comp == 1
        v_inf_comp = (v_inf.^2 .* vx .+ v_inf .* GMr .* rhat[1] .- v_inf .* vx .* sum(vel_loc .* rhat)) ./ denom
    elseif v_comp == 2
        v_inf_comp = (v_inf.^2 .* vy .+ v_inf .* GMr .* rhat[2] .- v_inf .* vy .* sum(vel_loc .* rhat)) ./ denom
    else
        v_inf_comp = (v_inf.^2 .* vz .+ v_inf .* GMr .* rhat[3] .- v_inf .* vz .* sum(vel_loc .* rhat)) ./ denom
    end
    return v_inf_comp
end

function solve_Rinit(X_surf, NS_vel, vel_surf; guess=[0.1 0.1 0.1], errV=1e-10, Roche_R=1e13)
    

    L_surf = [X_surf[2] .* vel_surf[3] .- vel_surf[2] .* X_surf[3] X_surf[3] .* vel_surf[1] .- vel_surf[3] .* X_surf[1] X_surf[1] .* vel_surf[2] .- vel_surf[1] .* X_surf[2]]
    L_surf_mag = sqrt.(sum(L_surf.^2))
    function f!(F, x)
        L_far = [x[2] .* NS_vel[3] .- NS_vel[2] .* x[3] x[3] .* NS_vel[1] .- NS_vel[3] .* x[1] x[1] .* NS_vel[2] .- NS_vel[1] .* x[2]]
        # x_mag = sqrt.(sum(x .^2))
        x_proj = sum(x .* (- NS_vel) ./ sqrt.(sum(NS_vel.^2)))
        # F[1] = L_far[1] ./ L_surf[1] .- 1.0
        # F[2] = L_far[2] ./ L_surf[2] .- 1.0
        # F[3] = L_far[3] ./ L_surf[3] .- 1.0
        
        F[1] = sqrt.(sum(L_far.^2)) ./ L_surf_mag .- 1.0
        F[2] = (sum(L_surf .* L_far) ./ (L_surf_mag .* sqrt.(sum(L_far.^2)))) .- 1.0
        F[3] = x_proj ./ Roche_R .- 1.0
    end

    soln = nlsolve(f!, guess, autodiff = :forward, ftol=errV, iterations=10000)

    # FF = zeros(3)
    # f!(FF, soln.zero)
    L_far = [soln.zero[2] .* NS_vel[3] .- NS_vel[2] .* soln.zero[3] soln.zero[3] .* NS_vel[1] .- NS_vel[3] .* soln.zero[1] soln.zero[1] .* NS_vel[2] .- NS_vel[1] .* soln.zero[2]]
    # print("TEST>>>> ", L_far ./ L_surf .- 1.0, "\n")
    # dist = sqrt.(sum(soln.zero.^2))
    # print("TEST>>>> ", dist, "\n")
    # print("accuracy... ", FF,"\n")
    return soln.zero
end


function test_vs_soln(θ, ϕ, r, NS_vel, x)
    ff = sum(NS_vel.^2); # unitless
   
    GMr = GNew ./ r ./ (c_km .^ 2); # unitless
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
    velV, accur = solve_vel_CS(θ, ϕ, r, NS_vel)
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
    tau = π * ωp .^2 ./ dOc_dl ./ (c_km .* hbar);
    
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


function dk_dl(x0, k0, Mvars; flat=true)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start, erg = Mvars2
    
    ωErg = ω(x0, k0, t_start, θm, ωPul, B0, rNS, gammaF);
    dkdr_grd = grad(kNR_e(seed(x0), k0, erg, t_start, θm, ωPul, B0, rNS, gammaF))
    
    #dkdr_proj = abs.(sum(k0 .* dkdr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    dkdr_proj = (sum(k0 .* dkdr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    return abs.(dkdr_proj)
end

function dk_ds(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start, ωErg = Mvars2
    
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
        # shat = R * Bhat[i, :];
        yhat = R * khat[i, :];
        
        # dkdr_proj_s[i] = abs.(sum(shat .* dkdr_grd[i, :]));
        dkdz = (sum(khat[i, :] .* dkdr_grd[i, :]));
        dkdy = (sum(yhat .* dkdr_grd[i, :]));
        dkdr_proj_s[i] = (dkdz .+ dkdy .* sin.(acos.(cθ[i])).^2 .* ωpL[i].^2 ./ (ωErg[i].^2 .- ωpL[i].^2 .* cθ[i].^2) ./ tan.(acos.(cθ[i])))
        
    end
    
    return abs.(dkdr_proj_s)
end

function test_dw_millar(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start, ωErg = Mvars2
    Bvec, ωp = GJ_Model_vec(x0, t_start, θm, ωPul, B0, rNS)
    
    
    kmag = sqrt.(sum(k0 .* k0, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    cθ = sum(k0 .* Bvec, dims=2) ./ (kmag .* Bmag)
    kgam = kNR_e(x0, k0, ωErg, t_start, θm, ωPul, B0, rNS, gammaF)
    xi = sin.(acos.(cθ)).^2 ./ (1.0 .- ωp.^2 ./ ωErg.^2 .* cθ.^2)
    khat = k0 ./ kmag
    Bhat = Bvec ./ Bmag
    
    Mass_NS = 1.0
    vel = sqrt.(2 .* GNew .* Mass_NS ./ sqrt.(sum(x0.^2, dims=2))) ./ c_km
    
    grad1 = grad(mass_mixingTerm(seed(x0), k0, Mvars, k_is_kgam=true))
    grad2 = grad(mass_mixingTerm(seed(x0), k0, Mvars, k_is_kgam=false)) #
    
    dkdr_proj_s = zeros(length(k0[:, 1]));
    dkdr_proj_s2 = zeros(length(k0[:, 1]));
    for i in 1:length(k0[:,1])
        uvec = [k0[i,2] .* Bvec[i,3] - Bvec[i,2] .* k0[i,3],  k0[i,3] .* Bvec[i,1] - Bvec[i,3] .* k0[i,1], k0[i,1] .* Bvec[i,2] - Bvec[i,1] .* k0[i,2]] ./ Bmag[i] ./ kmag[i]
        uhat = uvec ./ sqrt.(sum(uvec .^ 2));
        R = [uhat[1].^2 uhat[1] .* uhat[2] .+ uhat[3] uhat[1] .* uhat[3] .- uhat[2]; uhat[1] .* uhat[2] .- uhat[3] uhat[2].^2 uhat[2] .* uhat[3] .+ uhat[1]; uhat[1].*uhat[3] .+ uhat[2] uhat[2].*uhat[3] .- uhat[1] uhat[3].^2];
        # shat = R * Bhat[i, :];
        yhat = R * khat[i, :];
        
        # dkdr_proj_s[i] = abs.(sum(shat .* dkdr_grd[i, :]));
        dkdz = (sum(khat[i, :] .* grad1[i, :]));
        dkdy = (sum(yhat .* grad1[i, :]));
        dkdr_proj_s[i] = (dkdz .+ dkdy .* sin.(acos.(cθ[i])).^2 .* ωp[i].^2 ./ (ωErg[i].^2 .- ωp[i].^2 .* cθ[i].^2) ./ tan.(acos.(cθ[i])))
        
        dkdz2 = (sum(khat[i, :] .* grad2[i, :]));
        dkdy2 = (sum(yhat .* grad2[i, :]));
        dkdr_proj_s2[i] = (dkdz2 .+ dkdy2 .* sin.(acos.(cθ[i])).^2 .* ωp[i].^2 ./ (ωErg[i].^2 .- ωp[i].^2 .* cθ[i].^2) ./ tan.(acos.(cθ[i])))
        
    end
    # print(dkdr_proj_s ./ dkdr_proj_s2, "\n")
    return abs.(dkdr_proj_s2)
    

end

function mass_mixingTerm(x0, k0, Mvars; k_is_kgam=true)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start, ωErg = Mvars2
    Bvec, ωp = GJ_Model_vec(x0, t_start, θm, ωPul, B0, rNS)
    
    
    kmag = sqrt.(sum(k0 .* k0, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    cθ = sum(k0 .* Bvec, dims=2) ./ (kmag .* Bmag)
    kgam = kNR_e(x0, k0, ωErg, t_start, θm, ωPul, B0, rNS, gammaF)
    xi = sin.(acos.(cθ)).^2 ./ (1.0 .- ωp.^2 ./ ωErg.^2 .* cθ.^2)
    Mass_NS = 1.0
    vel = sqrt.(2 .* GNew .* Mass_NS ./ sqrt.(sum(x0.^2, dims=2))) ./ c_km
    
    if k_is_kgam
        factor = (ωp.^2 .- xi .* ωp.^2) ./ (2 .* kgam)
    else
        factor = (ωp.^2 .- xi .* ωp.^2) ./ (2 .* ωp .* vel)
        # factor = (ωp.^2 .- xi .* ωp.^2)
    end
    
    return factor
end


function dwds_abs_vec(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start, ωErg = Mvars2
    dωdr_grd = grad(GJ_Model_ωp_vec(seed(x0), t_start, θm, ωPul, B0, rNS))
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
        yhat = R * khat[i, :];
        
        
        dkdz = sum(khat[i,:] .* dωdr_grd[i, :]);
        dkdy = sum(yhat .* dωdr_grd[i, :]);
    
        dkdr_proj_s[i] = (dkdz .+ dkdy .* sin.(acos.(cθ[i])).^2 .* ωpL[i].^2 ./ (ωErg[i].^2 .- ωpL[i].^2 .* cθ[i].^2) ./ tan.(acos.(cθ[i])))

    end

    return abs.(dkdr_proj_s)

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

function find_samples(maxR, ntimes_ax, θm, ωPul, B0, rNS, Mass_a, Mass_NS; n_max=8, fix_time=0.0)
    batchsize = 2;
    over_cnt = zeros(batchsize)

    tt_ax = LinRange(-2*maxR, 2*maxR, ntimes_ax); # Not a real physical time -- just used to get trajectory crossing
    
    # randomly sample angles θ, ϕ
    θi = acos.(1.0 .- 2.0 .* rand(batchsize));
    ϕi = rand(batchsize) .* 2π;
    
    vvec_all = [sin.(θi) .* cos.(ϕi) sin.(θi) .* sin.(ϕi) cos.(θi)];
    # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
    ϕRND = rand(batchsize) .* 2π;
    rRND = sqrt.(rand(batchsize)) .* maxR; # standard flat sampling
    # rRND = rand(batchsize) .* maxR; # New 1/r sampling
    x1 = rRND .* cos.(ϕRND);
    x2 = rRND .* sin.(ϕRND);
    # rotate using Inv[EurlerMatrix(ϕi, θi, 0)] on vector (x1, x2, 0)
    x0_all= [x1 .* cos.(-ϕi) .* cos.(-θi) .+ x2 .* sin.(-ϕi) x2 .* cos.(-ϕi) .- x1 .* sin.(-ϕi) .* cos.(-θi) x1 .* sin.(-θi)];
    x_axion = [transpose(x0_all[i,:]) .+ transpose(vvec_all[i,:]) .* tt_ax[:] for i in 1:batchsize];

    cxing_st = [get_crossings(log.(GJ_Model_ωp_vec(x_axion[i], fix_time, θm, ωPul, B0, rNS)) .- log.(Mass_a)) for i in 1:batchsize];
    cxing = [apply(cxing_st[i], tt_ax) for i in 1:batchsize];
    
    randInx = [rand(1:n_max) for i in 1:batchsize];
    
    # see if keep any crossings
    indx_cx = [if length(cxing[i]) .>= randInx[i] i else -1 end for i in 1:batchsize];
    
    # indx_cx = [if length(cxing[i]) .> 0 i else -1 end for i in 1:batchsize];

    # remove those which dont
    randInx = randInx[indx_cx .> 0];
    indx_cx_cut = indx_cx[indx_cx .> 0];

    # randInx = [rand(1:length(cxing[indx_cx_cut][i])) for i in 1:length(indx_cx_cut)];
    
    cxing_short = [cxing[indx_cx_cut][i][randInx[i]] for i in 1:length(indx_cx_cut)];
    weights = [length(cxing[indx_cx_cut][i]) for i in 1:length(indx_cx_cut)];


    numX = length(cxing_short);
    R_sample = vcat([rRND[indx_cx_cut][i] for i in 1:numX]...);

    # print(cxing, "\t", cxing_short, "\t", indx_cx_cut, "\t", randInx, "\n")
    
    if numX != 0
        
        # print(x0_all, "\t", cxing_short, "\t", indx_cx_cut, "\t", randInx,"\n")
        xpos = [transpose(x0_all[indx_cx_cut[i], :]) .+ transpose(vvec_all[indx_cx_cut[i], :]) .* cxing_short[i] for i in 1:numX];
        vvec_full = [transpose(vvec_all[indx_cx_cut[i],:]) .* ones(1, 3) for i in 1:numX];
        

        # print(x0_all, "\t", xpos, "\t", vvec_full, "\t", R_sample, "\n")
        t_new_arr = LinRange(- abs.(tt_ax[3] - tt_ax[1]), abs.(tt_ax[3] - tt_ax[1]), 100);
        xpos_proj = [xpos[i] .+ vvec_full[i] .* t_new_arr[:] for i in 1:numX];

        cxing_st = [get_crossings(log.(GJ_Model_ωp_vec(xpos_proj[i], fix_time, θm, ωPul, B0, rNS)) .- log.(Mass_a)) for i in 1:numX];
        cxing = [apply(cxing_st[i], t_new_arr) for i in 1:numX];
        indx_cx = [if length(cxing[i]) .> 0 i else -1 end for i in 1:numX];
        indx_cx_cut = indx_cx[indx_cx .> 0];
        R_sample = R_sample[indx_cx_cut];
        numX = length(indx_cx_cut);
        if numX == 0
            return 0.0, 0.0, 0, 0.0, 0.0
        end



        randInx = [rand(1:length(cxing[indx_cx_cut][i])) for i in 1:numX];
        cxing = [cxing[indx_cx_cut][i][randInx[i]] for i in 1:numX];
        vvec_flat = reduce(vcat, vvec_full);
        # print(xpos, "\t", vvec_full, "\t", cxing, "\n")
        xpos = [xpos[indx_cx_cut[i],:] .+ vvec_full[indx_cx_cut[i],:] .* cxing[i] for i in 1:numX];
        vvec_full = [vvec_full[indx_cx_cut[i],:] for i in 1:numX];

        try
            xpos_flat = reduce(vcat, xpos);
        catch
            print("why is this a rare fail? \t", xpos, "\n")
        end
        try
            xpos_flat = reduce(vcat, xpos_flat);
            vvec_flat = reduce(vcat, vvec_full);
        catch
            print("for some reason reduce fail... ", vvec_full, "\t", xpos_flat, "\n")
            vvec_flat = vvec_full;
        end

       
        rmag = sqrt.(sum(xpos_flat .^ 2, dims=2));
        indx_r_cut = rmag .> (rNS + 0.0); #
        # print(xpos_flat, "\t", vvec_flat,"\t", R_sample, "\t", indx_r_cut, "\n")
        if sum(indx_r_cut) - length(xpos_flat[:,1 ]) < 0
            xpos_flat = xpos_flat[indx_r_cut[:], :]
            vvec_flat = vvec_flat[indx_r_cut[:], :]
            R_sample = R_sample[indx_r_cut[:]]
            numX = length(xpos_flat);
            rmag = sqrt.(sum(xpos_flat .^ 2, dims=2));
        end
        
        ntrajs = length(R_sample)
        if ntrajs == 0
            return 0.0, 0.0, 0, 0.0
        end
        
        # print("here...\t", xpos_flat, "\t", R_sample,"\n")
        ωpL = GJ_Model_ωp_vec(xpos_flat, zeros(ntrajs), θm, ωPul, B0, rNS)
        vmag = sqrt.(2 * 132698000000.0 .* Mass_NS ./ rmag) ; # km/s
        erg_ax = sqrt.( Mass_a^2 .+ (Mass_a .* vmag / 2.998e5) .^2 );
        
        # make sure not in forbidden region....
        fails = ωpL .> erg_ax;
        n_fails = sum(fails);
        if n_fails > 0
            
            ωpLi2 = [if fails[i] == 1 Mass_a .- GJ_Model_ωp_vec(transpose(xpos_flat[i,:]) .+ transpose(vvec_flat[i,:]) .* t_new_arr[:], fix_time, θm, ωPul, B0, rNS) else -1 end for i in 1:ntrajs];
            # ωpLi2 = [if fails[i] == 1 Mass_a .- GJ_Model_ωp_vec(xpos_flat[i,:] .+ vvec_flat[i,:] .* t_new_arr[:], [fix_time], θm, ωPul, B0, rNS) else -1 end for i in 1:ntrajs];

            t_new = [if length(ωpLi2[i]) .> 1 t_new_arr[findall(x->x==ωpLi2[i][ωpLi2[i] .> 0][argmin(ωpLi2[i][ωpLi2[i] .> 0])], ωpLi2[i])][1] else -1e6 end for i in 1:length(ωpLi2)];
            t_new = t_new[t_new .> -1e6];
            xpos_flat[fails[:],:] .+= vvec_flat[fails[:], :] .* t_new;

        end
        # print(xpos_flat, "\t")
        return xpos_flat, R_sample, ntrajs, weights
    else
        return 0.0, 0.0, 0, 0.0
    end
    
end


end



function main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, Ntajs, gammaF, batchsize; ode_err=1e-5, fix_time=Nothing, CLen_Scale=true, file_tag="", ntimes=1000, errSlve=1e-10, M_MC=1e-12, R_MC=1.0e9,  save_more=true, ntimes_ax=10000, dir_tag="results", trace_trajs=true, n_maxSample=8, axion_star_moddisp=false, theta_cut_trajs=true)

    # pass parameters
    # axion mass [eV], axion-photon coupling [1/GeV], misalignment angle (rot-B field) [rad], rotational freq pulars [1/s]
    # magnetic field strengh at surface [G], radius NS [km], mass NS [solar mass], dispersion relations
    # number of axion trajectories to generate
    
    # rhoDM -- this is asymptotic density. currently scaled to 1 GeV / cm^3

    accur_threshold = 1e-4;
    RT = RayTracer; # define ray tracer module

    func_use = RT.ωNR_e
    Roche_R = R_MC .* (2 .* Mass_NS ./ M_MC).^(1.0 ./ 3.0) # km
    NS_vel = [0.0 sin.(NS_vel_T) cos.(NS_vel_T)] .* NS_vel_M; # unitless
    NS_vel_norm = [0.0 sin.(NS_vel_T) cos.(NS_vel_T)] ; # unitless
    R_guessVal = -[0.0 sin.(NS_vel_T) cos.(NS_vel_T)] .* Roche_R
    


    # one time run --- time can be reduced in RayTracer.jl file by reducing sampling
    # cannot run this with a non-GJ model for time being (but works for relativistic plasmas as well)....
    maxR = RT.Find_Conversion_Surface(Mass_a, fix_time, θm, ωPul, B0, rNS, 1, false)
    
    # br_max = sqrt.(2 * GNew .* Mass_NS .* maxR) ./ (NS_vel_M .* c_km) # km
    br_max = 2 * R_MC;
    
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
    SaveAll = zeros(desired_trajs * 2, 24);
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

    # define time at which we find crossings
    t0_ax = zeros(batchsize);
    xpos_flat = zeros(batchsize, 3);
    R_sample = zeros(batchsize);
    mcmc_weights = zeros(batchsize);
    times_pts = ones(batchsize) .* fix_time;
    filled_positions = false;
    fill_indx = 1;
    Ncx_max = 1;
    count_success = 0;


    while photon_trajs < desired_trajs

        while !filled_positions
            
            xv, Rv, numV, weights = RT.find_samples(maxR, ntimes_ax, θm, ωPul, B0, rNS, Mass_a, Mass_NS, n_max=n_maxSample, fix_time=fix_time)
            f_inx += 2
            
            if numV == 0
                continue
            end
          

            for i in 1:numV # Keep more?
                f_inx -= 1
                if fill_indx <= batchsize
                    xpos_flat[fill_indx, :] .= xv[i, :];
                    R_sample[fill_indx] = Rv[i];
                    mcmc_weights[fill_indx] = n_maxSample;
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
        vel_disp = sqrt.(2 .* GNew .* M_MC ./ R_MC) ./ c_km  # dimensionless
        
        
        vel = zeros(length(rmag)*2, 3)
        vF_AX = zeros(length(rmag)*2, 3)
        xF_AX = zeros(length(rmag)*2, 3)
        mcmc_weightsFull = cat(mcmc_weights, mcmc_weights, dims=1)
        
        for i in 1:length(rmag)
            if !axion_star_moddisp
                v_perturb = erfinv.(2 .* rand(1, 3) .- 1.0) .* vel_disp
            else
                v_perturb = rand(1, 3) .* vel_disp ./ sqrt.(3)
            end
            
            NS_vel_p = NS_vel .+ v_perturb

            

            fail_first = false
            fail_second = false
            
            found = false
            cnt_careful= 0
            while !found
                randARR = float(rand(-1:2:1, 3))
                vGuess = [rand() .* randARR[1] rand() .* randARR[2] rand() .* randARR[3]]
                velV, accur = RT.solve_vel_CS(θ[i], ϕ[i], rmag[i], NS_vel_p, guess=vGuess, errV=errSlve)
                test_func = (sum(velV.^2) .- (2 .* GNew .* Mass_NS ./ rmag[i] ./ (c_km .^ 2))); # unitless
                if (accur .< accur_threshold)&&(test_func .> 0)
                    vel[i, :] = velV
                    mcmc_weightsFull[i] *= RT.jacobian_fv(xpos_flat[i, :], velV)
                    found = true
                end
                cnt_careful += 1
                if cnt_careful > 50
                    print("failing here at pt 1....")
                    mcmc_weightsFull[i] *= 0.0
                    fail_first = true
                    break;
                end
            end
            
            
            found = false
            cnt_careful= 0
            while !found
                randARR = float(rand(-1:2:1, 3))
                vGuess = [rand() .* randARR[1] rand() .* randARR[2] rand() .* randARR[3]]
                
                velV2, accur = RT.solve_vel_CS(θ[i], ϕ[i], rmag[i], NS_vel_p, guess=vGuess, errV=errSlve)
                

                test_func = (sum(velV2.^2) .- (2 .* GNew .* Mass_NS ./ rmag[i] ./ (c_km .^ 2))); # unitless
                if (velV2 != vel[i, :])&&(accur .< accur_threshold)&&(test_func .> 0)
                    found = true
                    vel[i+length(rmag), :] = velV2
                    mcmc_weightsFull[i+length(rmag)] *= RT.jacobian_fv(xpos_flat[i, :], velV2)
                end
                cnt_careful += 1
                if cnt_careful > 50
                    print("failing here at pt 2....")
                    mcmc_weightsFull[i+length(rmag)] *= 0.0
                    fail_second = true
                    break;
                end
            end
            
            if !fail_first
                if !trace_trajs
                    xF_AX[i, :] = RT.solve_Rinit(xpos_flat[i, :], NS_vel_p, vel[i, :]; guess=R_guessVal, errV=1e-10, Roche_R=Roche_R)
                end
                vF_AX[i, :] = NS_vel_p;
            end
            if !fail_second
                if !trace_trajs
                    xF_AX[i+length(rmag), :] = RT.solve_Rinit(xpos_flat[i, :], NS_vel_p, vel[i+length(rmag), :]; guess=R_guessVal, errV=1e-10, Roche_R=Roche_R)
                end
                vF_AX[i, :] = NS_vel_p;
            end
            
        end
        
        # stack results
        xpos_stacked = cat(xpos_flat, xpos_flat, dims=1)
        rmag = cat(rmag, rmag, dims=1)
        R_sampleFull = cat(R_sample, R_sample, dims=1)
        t0_full = cat(times_pts, times_pts, dims=1)
        
        
        # b = sqrt.(sum(vel.^2, dims=2)) .* rmag ./ (NS_vel_M)
        
        
        if trace_trajs
            nsteps = 1000;
            ln_tstart=-15;
            ln_tend=log.(Roche_R ./ (NS_vel_M .* c_km));
            ode_err=1e-12;
            NumerP = [ln_tstart, ln_tend, ode_err]
            tempX, vempV = RT.propagateAxion(xpos_stacked, vel .* c_km, nsteps, NumerP);
            xF_AX = tempX[:, :, end]
            vF_AX = vempV[:, :, end]
            # print(xF_tempt, "\t",  xF_AX, "\t", vF_tempt, "\t", vF_AX, "\n")
        end
        
        if theta_cut_trajs
            xFNorm = xF_AX ./ sqrt.(sum(xF_AX .^2, dims=2));
            theta_real = zeros(length(xF_AX[:,1]))
            try
                theta_real = acos.(abs.(sum(xFNorm .* NS_vel_norm, dims=2)));
            catch
                theta_real = 0.0 # sometimes failure to due precision, doesnt matter though
            end
            theta_max = asin.(br_max ./ Roche_R)
            cut_trajs = [if abs.(theta_real[i]) .<= abs.(theta_max) i else -1 end for i in 1:length(theta_real)];
            f_inx += sum(cut_trajs .<= 0);
            cut_trajs = cut_trajs[cut_trajs .> 0];
        else
            cut_trajs = [i for i in 1:length(xF_AX[:, 1])];
        end
        
        
        if length(cut_trajs) == 0
            continue
        end
        
        count_success += length(cut_trajs)
        # print("success: ", count_success, " total \t", f_inx, "\n")
        
        xpos_stacked = xpos_stacked[cut_trajs, :];
        vel = vel[cut_trajs, :];
        rmag = rmag[cut_trajs]
        R_sampleFull = R_sampleFull[cut_trajs]
        t0_full = t0_full[cut_trajs]
        mcmc_weightsFull = mcmc_weightsFull[cut_trajs]
        xF_AX = xF_AX[cut_trajs, :];
        vF_AX = vF_AX[cut_trajs, :];
        
        

        # print(xpos_stacked, "\t", t0_full, "\n")

        vmag = sqrt.(sum(vel.^2, dims=2)) .* c_km ; # km/s
        erg_ax = sqrt.( Mass_a^2 .+ (Mass_a .* vmag / c_km) .^2 );
        ωpL = RT.GJ_Model_ωp_vec(xpos_stacked, t0_full, θm, ωPul, B0, rNS)
        newV = vel ./ sqrt.(sum(vel.^2, dims=2))

        calpha = RT.surfNorm(xpos_stacked, newV, [func_use, [θm, ωPul, B0, rNS, gammaF, t0_full, Mass_NS]], return_cos=true); # alpha
        weight_angle = abs.(calpha);

        Bvec, ωp = RT.GJ_Model_vec(xpos_stacked, t0_full, θm, ωPul, B0, rNS);
        Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))

       
        cθ = sum(newV .* Bvec, dims=2) ./ Bmag
        
        
        vmag_tot = vmag; # km/s
        k_init_ax = Mass_a .* newV .* (vmag_tot ./ c_km);

        erg_ax = sqrt.(sum(k_init_ax .^2, dims=2) .+ Mass_a .^2);

        k_init = sqrt.(erg_ax .^2 .-  ωp .^ 2) .* newV ./ sqrt.(1.0 .- cθ .^ 2 .* ωp .^ 2 ./ erg_ax .^2);


        B_tot = sqrt.(sum(Bvec .^ 2, dims=2)) .* (1.95e-20) ; # GeV^2
        
        MagnetoVars =  [θm, ωPul, B0, rNS, [1.0 1.0], t0_full, erg_ax]
        sln_δk = RT.dk_ds(xpos_stacked, k_init, [func_use, MagnetoVars]);
        conversion_F = sln_δk ./  (hbar .* c_km) # 1/km^2;
        
        # sln_δk_a = RT.dwds_abs_vec(xpos_stacked, k_init, [func_use, MagnetoVars]);
        # conversion_F_a = sln_δk_a ./  (hbar .* c_km) # 1/km^2;
        
        # sln_δk_millar = RT.test_dw_millar(xpos_stacked, k_init, [func_use, MagnetoVars]);
        # conversion_F_Millar = sln_δk_millar ./  (hbar .* c_km) # 1/km^2;
        # print(conversion_F_a ./ conversion_F_Millar, "\n\n")
        
        MagnetoVars =  [θm, ωPul, B0, rNS, [1.0 1.0], times_pts]
        mass_factor = sin.(acos.(cθ)).^2 ./ (sin.(acos.(cθ)).^2 .+ (vmag_tot ./ 2.998e5).^2).^2
        mass_factorR = 1 ./ (sin.(acos.(cθ)).^2 .+ (vmag_tot ./ 2.998e5).^2).^2
        # prob_alx = π ./ 2 .* (Ax_g .* B_tot) .^2 ./ abs.(conversion_F_a) .* (1e9 .^2) ./ (vmag_tot ./ c_km)  ./ ((hbar .* c_km) .^2) .* mass_factor; #unitless
        
        
        # kk = Mass_a .* (vmag_tot ./ c_km)
        # xi = sin.(acos.(cθ)).^2 ./ (1.0 .- ωp.^2 ./ erg_ax.^2 .* cθ.^2)
        # prob_millar =  π ./ 2 .* (Ax_g .* B_tot) .^2 ./ abs.(conversion_F_Millar) .* (1e9 .^2) ./ (vmag_tot ./ 2.998e5)  ./ ((2.998e5 .* 6.58e-16) .^2) .* mass_factor .* (xi .* ωp ./ kk); #unitless
        
        Prob = π ./ 2 .* (Ax_g .* B_tot) .^2 ./ conversion_F .* (1e9 .^2) ./ (vmag_tot ./ 2.998e5) .^2 ./ ((2.998e5 .* 6.58e-16) .^2) .* mass_factorR; #unitless
        
        # prob_alx = π ./ 2 .* (Ax_g .* B_tot) .^2 ./ conversion_F_a .* (1e9 .^2) ./ (vmag_tot ./ 2.998e5)  ./ ((2.998e5 .* 6.58e-16) .^2) ./ sin.(acos.(cθ)).^2; #unitless
        # print(prob ./ prob_alx, "\t", prob2 ./ prob_alx, "\n")
        # print(abs.(conversion_F_a) ./ abs.(conversion_F_Millar), "\n")
        # print(prob_alx ./ prob_millar, "\t", Prob ./ prob_millar, "\n")
        
        # phaseS = (2 .* π .* maxR .* R_sampleFull .* 2) .* 1.0 .* Prob ./ Mass_a .* 1e9 .* (1e5).^3  # 1 / km
        phaseS = (2 .* π .* maxR.^2) .* 1.0 .* Prob ./ Mass_a .* 1e9 .* (1e5).^3  # 1 / km

        
        sln_prob = weight_angle .* (vmag ./ c_km) .* phaseS .* c_km .* mcmc_weightsFull ; # photons / second
        

        
        sln_k = k_init;
        sln_x = xpos_stacked;
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
        # Δω = (RT.dwdt_vec(xF, kF, ttΔω, passA) ) ./ Mass_a .+ vF_AX.^2 ./ 2.0;
        Δω = tF[:, end] ./ Mass_a .+ sqrt.(sum(vF_AX.^2, dims=2)) ./ 2.0;
        
        
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
        
        f_inx += num_photons
        
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 1] .= view(θf, :);
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 2] .= view(ϕf,:);
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 3] .= view(θfX, :);
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 4] .= view(ϕfX, :);
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 5] .= sqrt.(sum(xF[:, :, end] .^2, dims=2))[:]; # r final
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 6] .= sln_prob[:] .* weightC .^ 2 .* exp.(-opticalDepth[:]); #  num photons / second
        # SaveAll[photon_trajs:photon_trajs + num_photons - 1, 6] .= sln_probS[:]; #  num photons / second
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

        
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 19] .= xF_AX[:, 1]; #
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 20] .= xF_AX[:, 2]; #
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 21] .= xF_AX[:, 3]; #
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 22] .= vF_AX[:, 1]; #
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 23] .= vF_AX[:, 2]; #
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 24] .= vF_AX[:, 3]; #
        

        photon_trajs += num_photons;

    end

    # cut out unused elements
    SaveAll = SaveAll[SaveAll[:,6] .> 0, :];
    # print("FINX \t", f_inx, "\n")
    SaveAll[:,6] ./= (float(f_inx)); # divide off by N trajectories sampled

    fileN = dir_tag*"/Minicluster__MassAx_"*string(Mass_a)*"_AxG_"*string(Ax_g);
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
