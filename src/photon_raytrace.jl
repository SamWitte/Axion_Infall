__precompile__()


module RayTracer

using ForwardDiff: gradient, derivative, Dual, Partials, hessian
using OrdinaryDiffEq
# using CuArrays
using DifferentialEquations
using Plots
using LSODA


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

#     u0 = cu([x0 k0])
    u0 = ([x0 k0 zeros(length(x0[:, 1]))])

    prob = ODEProblem(func!, u0, tspan, [ω, Mvars], reltol=ode_err*1e0, abstol=ode_err, maxiters=1e5)
    sol = solve(prob, Vern6(), saveat=saveat)
    # sol = solve(prob, Tsit5(), saveat=saveat, batch_size=10)
    x = cat([Array(u)[:, 1:3] for u in sol.u]..., dims = 3);
    k = cat([Array(u)[:, 4:6] for u in sol.u]..., dims = 3);
    dt = cat([Array(u)[:, 7] for u in sol.u]..., dims = 2);
    
    return x, k, dt
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


function Find_Conversion_Surface(Ax_mass, t_in, θm, ω, B0, rNS; thetaVs=100, phiVs=100)

    # rayT = RayTracer;
    om_test = GJ_Model_ωp_scalar(rNS .* [sin.(θm) 0.0 cos.(θm)], t_in, θm, ω, B0, rNS);
    rc_guess = rNS .* (om_test ./ Ax_mass) .^ (2.0 ./ 3.0);
    
    rL = 10 .^ LinRange(log10.(rNS), log10.(2 .* rc_guess), 5000)
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


function surfNorm(x0, k0, Mvars; return_cos=true)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    dωdr_grd = grad(GJ_Model_ωp_vec(seed(x0), t_start, θm, ωPul, B0, rNS))
    snorm = dωdr_grd ./ sqrt.(sum(dωdr_grd .^ 2, dims=2))
    ctheta = (sum(k0 .* snorm, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    if return_cos
        return ctheta
    else
        return ctheta, snorm
    end
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


function dwdt_vec(x0, k0, tarr, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    nphotons = size(x0)[1]
    delW = zeros(nphotons);
    
    for k in 1:nphotons
        t0 = tarr .+ t_start[k];
        for i in 2:length(tarr)
            dωdt = derivative(t -> ω(transpose(x0[k, :, i]), transpose(k0[k, :, i]), t, θm, ωPul, B0, rNS, gammaF), t0[i]);
            delW[k] += dωdt[1] .* sqrt.(sum((x0[k, :, i] .- x0[k, :, i-1]) .^2)) / 2.998e5
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
    
    ψ = ϕ .- ω .* t
    
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
    ϕ = atan.(view(x, :, 2),  view(x, :, 1))
    θ = acos.(view(x, :, 3) ./ r)
    # ϕ = atan.(x[:, 2], x[:, 1])
    # θ = acos.(x[:, 3] ./ r)
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
    ϕ = atan.(x[2] , x[1])
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

function ωFree(x, k, t, θm, ωPul, B0, rNS, gammaF)
    # assume simple case where ωp proportional to r^{-3/2}, no time dependence, no magnetic field
    return sqrt.(sum(k .* k, dims = 2) .+ 1e-40 .* sqrt.(sum(x .* x, dims=2)) ./ (rNS.^ 2) )
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
        # print(F[1], "\t",F[2], "\t", F[3],"\n")
        # print(θ, "\t", ϕ,"\t", r, "\n")
    end
    
    soln = nlsolve(f!, guess, autodiff = :forward, ftol=1e-24, iterations=10000)
   
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


function surface_solver(Mass_a, θm, ωPul, B0, rNS, t_in, NS_vel_M, NS_vel_T; nsteps=10000, ln_tstart=-4, ln_tend=5, ode_err=1e-8, phiVs=100, thetaVs=100, threshold=0.05, single_density_field=true, sve=False)

    NS_vel = [sin.(NS_vel_T) 0.0 cos.(NS_vel_T)] .* NS_vel_M;
    
    RT = RayTracer; # define ray tracer module
    func_use = RT.ωNR_e
    ConvR = RT.Find_Conversion_Surface(Mass_a, t_in, θm, ωPul, B0, rNS, thetaVs=thetaVs, phiVs=phiVs)
    ConvR_Cut = ConvR[ConvR[:,3] .>= rNS, :]
    NumerP = [ln_tstart, ln_tend, ode_err]
    vGu=0.3 # guess point
    
    if !single_density_field
        finalX = zeros(2 * length(ConvR_Cut[:,1]), 3)
        finalV = zeros(2 * length(ConvR_Cut[:,1]), 3)
    end
    
    SurfaceX = zeros(2 * length(ConvR_Cut[:,1]), 3)
    SurfaceV = zeros(2 * length(ConvR_Cut[:,1]), 3)
    cnt = 1;
    MagnetoVars = [θm, ωPul, B0, rNS, [1.0 1.0], t_in]
    
    for i in 1:length(ConvR_Cut[:,1])
        θ, ϕ, r = ConvR_Cut[i,:]

        
        velV = solve_vel_CS(θ, ϕ, r, NS_vel, guess=[vGu vGu vGu])
        velV2 = solve_vel_CS(θ, ϕ, r, NS_vel, guess=[-vGu -vGu -vGu])
        xPos = [r .* sin.(θ) .* cos.(ϕ) r .* sin.(θ) .* sin.(ϕ) r .* cos.(θ)]
        
        if single_density_field
            SurfaceX[cnt, :] = xPos;
            SurfaceV[cnt, :] = velV;
            cnt += 1;
            SurfaceX[cnt, :] = xPos;
            SurfaceV[cnt, :] = velV2;
            cnt += 1;
        
        else
            vel_init = [velV; velV2];
            x_init = [r .* sin.(θ) .* cos.(ϕ) r .* sin.(θ) .* sin.(ϕ) r .* cos.(θ); r .* sin.(θ) .* cos.(ϕ) r .* sin.(θ) .* sin.(ϕ) r .* cos.(θ)]
            xF, vF = RT.propagateAxion(x_init, vel_init, nsteps, NumerP);
   
            if sqrt.(sum((hcat(vF[1, :, end]...) .- NS_vel) .^ 2)) < threshold
                finalX[cnt, :] = xF[1, :, end];
                finalV[cnt, :] = vF[1, :, end];
                SurfaceX[cnt, :] = xF[1, :, 1];
                SurfaceV[cnt, :] = vF[1, :, 1];
                cnt += 1;
            end
        
            if sqrt.(sum((hcat(vF[2, :, end]...) .- NS_vel) .^ 2)) ./ sqrt.(sum(NS_vel.^2)) < threshold
                finalX[cnt, :] = xF[2, :, end];
                finalV[cnt, :] = vF[2, :, end];
                SurfaceX[cnt, :] = xF[2, :, 1];
                SurfaceV[cnt, :] = vF[2, :, 1];
                cnt += 1;
            end
            
        end
        
    end
    
    
    dkdl = RT.dk_ds(SurfaceX[1:(cnt-1),:], Mass_a .* SurfaceV[1:(cnt-1),:], [RT.ωNR_e, MagnetoVars]) ./ (6.58e-16 .* 2.998e5) ; 1 / km^2
    ctheta = surfNorm(SurfaceX[1:(cnt-1),:], SurfaceV[1:(cnt-1),:], [func_use, MagnetoVars]; return_cos=true)
    
    if sve
        dirN = "/scratch/work/"
        fileTail = "PhaseSpace_Map_AxionM_"*string(Mass_a)*"_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0)*"_rNS_";
        fileTail *= "Time_"*string(t_in)*"_sec_"
        fileTail *= "_NS_Mag_"*string(round(NS_vel_M, digits=5))*"_NS_Theta_"*string(round(NS_vel_T, digits=3))
        fileTail *= "_.npz"
        npzwrite(dirN*"SurfaceX_"*fileTail, SurfaceX[1:(cnt-1),:])
        npzwrite(dirN*"SurfaceV_"*fileTail, SurfaceV[1:(cnt-1),:])
        npzwrite(dirN*"dkdz_"*fileTail, dkdl)
        npzwrite(dirN*"ctheta_"*fileTail, ctheta)
        if !single_density_field
            npzwrite(dirN*"FinalX_"*fileTail, finalX[1:(cnt-1),:])
            npzwrite(dirN*"FinalV_"*fileTail, finalV[1:(cnt-1),:])
        end
        
            
    else
        return SurfaceX, SurfaceV, dkdl, ctheta
    end

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

function main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, t_list; ode_err=1e-5, maxR=Nothing, cutT=10, fix_time=Nothing, CLen_Scale=true, file_tag="", ntimes=1000, NS_vel_M=0.0, NS_vel_T=0.0, RadApprox=false, phiVs=100, thetaVs=100, vel_disp=1.0, single_density_field=true)

    NS_vel = [sin.(NS_vel_T) 0.0 cos.(NS_vel_T)] .* NS_vel_M;
    RT = RayTracer; # define ray tracer module
    if !RadApprox
        func_use = RT.ωNR_e
    else
        func_use = RT.ωFree
        CLen_Scale = false
        file_tag *= "_RadApprox_"
    end
    
    ln_t_start = -20;
    ln_t_end = log.(1 ./ ωPul);
    NumerPass = [ln_t_start, ln_t_end, ode_err];
    ttΔω = exp.(LinRange(ln_t_start, ln_t_end, ntimes));
    
    θL = LinRange(0.01 , π - 0.01, thetaVs);
    ϕL = LinRange(0.01, 2 * π - 0.01, phiVs);
    dθ = θL[2] - θL[1];
    dϕ = ϕL[2] - ϕL[1];
    
    
    dirN = "/scratch/work/"

    for i in 1:length(t_list)
        t_in = t_list[i]
        
        SurfaceX, SurfaceV, dkdl, ctheta = surface_solver(Mass_a, θm, ωPul, B0, rNS, t_in, NS_vel_M, NS_vel_T; phiVs=phiVs,thetaVs=thetaVs,single_density_field=single_density_field, sve=false)
        SaveAll = zeros(length(SurfaceX[:, 1]), 18);
        
        sln_t = ones(length(SurfaceX[:,1])) .* t_list[i]
        vmag_tot = sqrt.(sum(SurfaceV.^2, dims=2))
        kini = SurfaceV .* Mass_a
        MagnetoVars = [θm, ωPul, B0, rNS, [1.0 1.0], sln_t] # θm, ωPul, B0, rNS, gamma factors, Time = 0
        
        rr = sqrt.(sum(SurfaceX .^2, dims=2));
        
        xF, kF, tF = RT.propagate(func_use, SurfaceX, kini, ntimes, MagnetoVars, NumerPass);
        ϕf = atan.(view(kF, :, 2, ntimes), view(kF, :, 1, ntimes));
        ϕfX = atan.(view(xF, :, 2, ntimes), view(xF, :, 1, ntimes));
        θf = acos.(view(kF, :, 3, ntimes) ./ sqrt.(sum(view(kF, :, :, ntimes) .^2, dims=2)));
        θfX = acos.(view(xF, :, 3, ntimes) ./ sqrt.(sum(view(xF, :, :, ntimes) .^2, dims=2)));

        # compute energy dispersion (ωf - ωi) / ωi
        passA = [func_use, MagnetoVars];
        
        Δω = tF[:, end] ./ Mass_a;
        opticalDepth = RT.tau_cyc(xF, kF, ttΔω, passA, Mass_a);
    
         
        num_photons = length(SurfaceX[:,1])
        sln_ConVL = sqrt.(π ./ dkdl); # km
        passA2 = [func_use, MagnetoVars, Mass_a];
        if CLen_Scale
            weightC = ConvL_weights(xF, kF, vmag_tot, ttΔω, sln_ConVL, passA2)
        else
            weightC = ones(num_photons)
        end
        
        Bvec, ωp = RT.GJ_Model_vec(SurfaceX, sln_t, θm, ωPul, B0, rNS);
        cθ = sum(kini .* Bvec, dims=2) ./ sqrt.(sum(Bvec.^2, dims=2)) ./ sqrt.(sum(kini.^2, dims=2));
        B_tot = sqrt.(sum(Bvec .^ 2, dims=2)) .* (1.95e-20); # GeV^2
        
        
        if !RadApprox
            cLen = 1.0 ./ dkdl[:]
        else
            cLen = 2.0 .* rr .* vmag_tot ./ (3.0 .* Mass_a) .* 6.56e-16 .* 2.998e5;
        end
        probab = π ./ 2.0 .* vmag_tot.^2 .* (1e-12 .* B_tot ./  sin.(acos.(cθ))).^2 .* cLen ./ (2.998e5 .* 6.58e-16 ./ 1e9).^2 ./  sin.(acos.(cθ)).^2 ; # g [1e-12 GeV^-1], unitless
        
        dS = rr.^2 .* sin.(acos.(SurfaceX[:, 3] ./ rr)) .* dθ .* dϕ;
        # assume number density at each point 1 / cm^3! Arbitrary but we re-scale after.
        SaveAll[:, 1] .= view(θf, :);
        SaveAll[:, 2] .= view(ϕf,:);
        SaveAll[:, 3] .= view(θfX, :);
        SaveAll[:, 4] .= view(ϕfX, :);
        SaveAll[:, 5] .= sqrt.(sum(xF[:, :, end] .^2, dims=2))[:]; # r final
        SaveAll[:, 7] .= Δω[:];
        SaveAll[:, 6] .= 1.0 .* dS[:] .* ctheta[:] .* vmag_tot[:].^3  .* probab[:] .* weightC[:] .^ 2 .* exp.(-opticalDepth[:]) .* (1e5).^2 .* 2.998e10; # num photons cm^3 / s. multiply by rho to get L in eV/s
        SaveAll[:, 6] .*= 2 ./ sqrt.(π) .* sqrt.(132698000000.0 ./ (2.998e5 .^ 2) ./ rr[:]) ./ (vel_disp ./ 2.998e5) ./ (4 .* π); # note 1 / (4 pi) corrects for non isotropic distribution.
        
        SaveAll[:, 8] .= sln_ConVL[:];
        SaveAll[:, 9] .= SurfaceX[:, 1]; # initial x
        SaveAll[:, 10] .= SurfaceX[:, 2]; # initial y
        SaveAll[:, 11] .= SurfaceX[:, 3]; # initial z
        SaveAll[:, 12] .= kini[:, 1]; # initial kx
        SaveAll[:, 13] .= kini[:, 2]; # initial ky
        SaveAll[:, 14] .= kini[:, 3]; # initial kz
        SaveAll[:, 15] .= opticalDepth[:]; # optical depth
        SaveAll[:, 16] .= weightC[:]; #
        SaveAll[:, 17] .= probab[:]; # optical depth
        SaveAll[:, 18] .= ctheta[:]; # surf norm
            
        fileN = "/scratch/work/Minicluster_Time_"*string(t_list[i])*"_MassAx_"*string(Mass_a);
        fileN *= "_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0)*"_rNS_";
        fileN *= string(rNS)*"_MassNS_"*string(Mass_NS);
        
        fileN *= "_NS_Mag_"*string(round(NS_vel_M, digits=5))*"_NS_Theta_"*string(round(NS_vel_T, digits=3))
        fileN *= "_"*file_tag*"_.npz";
        npzwrite(fileN, SaveAll)
    end

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
        dePhase_Factor = cos.(thetaZ) .- ωpL .^2 ./ ωfull .^2 .* sin.(thetaZ) .*  sin.(thetaB).^2 ./ tan.(thetaB) ./ (1 .- cos.(thetaB).^2 .* ωpL .^2 ./ ωfull .^2);
        phaseS[:, i] = dR[:, i] .* (Mass_a .* v_sur .- dePhase_Factor .* sqrt.( abs.((ωfull .^2 .-  ωpL .^ 2 ) ./ (1 .- cos.(thetaB) .^2 .* ωpL .^2 ./ ωfull .^2)))  )
        
    end
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

function period_average(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, t_list; ode_err=1e-5, maxR=Nothing, cutT=10, fix_time=Nothing, CLen_Scale=true, file_tag="", ntimes=1000, NS_vel_M=0.0, NS_vel_T=0.0, RadApprox=false)

    NS_vel = [sin.(NS_vel_T) 0.0 cos.(NS_vel_T)] .* NS_vel_M;

    if RadApprox
        CLen_Scale = false
        file_tag *= "_RadApprox_"
    end

    
    table_started = false
    dt = (t_list[2] - t_list[1])
    for i in 1:length(t_list)
        t_in = t_list[i]
        
        fileN = "/scratch/work/Minicluster_Time_"*string(t_list[i])*"_MassAx_"*string(Mass_a);
        fileN *= "_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0)*"_rNS_";
        fileN *= string(rNS)*"_MassNS_"*string(Mass_NS);
        
        fileN *= "_NS_Mag_"*string(round(NS_vel_M, digits=5))*"_NS_Theta_"*string(round(NS_vel_T, digits=3))
        fileN *= "_"*file_tag*"_.npz";
        
        if !isfile(fileN)
            continue
        end
        
        hold = npzread(fileN)
        if !table_started
            sve_info = hold
            table_started = true
        else
            sve_info = vcat((sve_info, hold)...)
        end
        
    end
    
    sve_info = sve_info[sve_info[:,6] .> 0, :];
    if length(sve_info[:,6]) > 0
        period = 2 .* π ./ ωPul;
        sve_info[:, 6] ./= period;
        sve_info[:, 6] .*= dt;
        
        fileS = fileN = "results/Minicluster_PeriodAvg_MassAx_"*string(Mass_a);
        fileS *= "_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0)*"_rNS_";
        fileS *= string(rNS)*"_MassNS_"*string(Mass_NS);
            
        fileS *= "_NS_Mag_"*string(round(NS_vel_M, digits=5))*"_NS_Theta_"*string(round(NS_vel_T, digits=3))
        fileS *= "_"*file_tag*"_.npz";
        npzwrite(fileS, sve_info)
    end
    
end
