__precompile__()


module RayTracer

using ForwardDiff: gradient, derivative, Dual, Partials, hessian
using OrdinaryDiffEq
# using CuArrays
using DifferentialEquations
using Plots


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


# propogate photon module
function propagate(ω, x0::Matrix, k0::Matrix,  nsteps::Int, Mvars::Array, NumerP::Array)
    ln_tstart, ln_tend, ode_err = NumerP
    tspan = (ln_tstart, ln_tend)
    saveat = (tspan[2] .- tspan[1]) ./ (nsteps-1)

#     u0 = cu([x0 k0])
    u0 = ([x0 k0 zeros(length(x0[:, 1]))])

    prob = ODEProblem(func!, u0, tspan, [ω, Mvars], reltol=ode_err*1e-2, abstol=ode_err, maxiters=1e5)
    sol = solve(prob, Tsit5(), saveat=saveat)
    x = cat([Array(u)[:, 1:3] for u in sol.u]..., dims = 3);
    k = cat([Array(u)[:, 4:6] for u in sol.u]..., dims = 3);
    dt = cat([Array(u)[:, 7] for u in sol.u]..., dims = 2);
    
    return x, k, dt
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
    
    if sum(kmag .= 0) > 0
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
    ϕ = atan.(view(x, :, 2) ./  view(x, :, 1))
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
    ϕ = atan.(x[2] ./ x[1])
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
    ϕ = atan.(x[2] ./ x[1])
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

end

function main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, t_list; ode_err=1e-5, maxR=Nothing, cutT=10, fix_time=Nothing, CLen_Scale=true, file_tag="", ntimes=1000, v_NS=[0 0 0], RadApprox=false)

    RT = RayTracer; # define ray tracer module
    func_use = RT.ωNR_e
    
    ln_t_start = -20;
    ln_t_end = log.(1 ./ ωPul);
    NumerPass = [ln_t_start, ln_t_end, ode_err];
    ttΔω = exp.(LinRange(ln_t_start, ln_t_end, ntimes));
    
    
    dirN = "temp_storage/"

    for i in 1:length(t_list)
        t_in = t_list[i]
        ## FIX
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
        
        sfx_fnme = dirN*"SurfaceX_"*fileTail
        sfv_fnme = dirN*"SurfaceV_"*fileTail
        sfdk_fnme = dirN*"dkdz_"*fileTail
        sfct_fnme = dirN*"ctheta_"*fileTail
        
        if isfile(sfx_fnme)&&isfile(sfv_fnme)&&isfile(sfdk_fnme)&&isfile(sfct_fnme)
            SurfaceX = npzread(sfx_fnme)
            SurfaceV = npzread(sfv_fnme)
            dkdl = npzread(sfdk_fnme)
            ctheta = npzread(sfct_fnme)
        else
            print(sfx_fnme,"\n")
            print("files not generated....\n\n")
            return
        end
        
        SaveAll = zeros(length(SurfaceX[:, 1]), 7);
        
        sln_t = ones(length(SurfaceX[:,1])) .* t_list[i]
        vmag_tot = sqrt.(sum(SurfaceV.^2, dims=2))
        kini = SurfaceV .* Mass_a
        MagnetoVars = [θm, ωPul, B0, rNS, [1.0 1.0], sln_t] # θm, ωPul, B0, rNS, gamma factors, Time = 0
        
        xF, kF, tF = RT.propagate(func_use, SurfaceX, kini, ntimes, MagnetoVars, NumerPass);
        ϕf = atan.(view(kF, :, 2, ntimes), view(kF, :, 1, ntimes));
        ϕfX = atan.(view(xF, :, 2, ntimes), view(xF, :, 1, ntimes));
        θf = acos.(view(kF, :, 3, ntimes) ./ sqrt.(sum(view(kF, :, :, ntimes) .^2, dims=2)));
        θfX = acos.(view(xF, :, 3, ntimes) ./ sqrt.(sum(view(xF, :, :, ntimes) .^2, dims=2)));

        # compute energy dispersion (ωf - ωi) / ωi
        passA = [func_use, MagnetoVars];
        Δω = tF[:, end] ./ Mass_a;
        
        opticalDepth = RT.tau_cyc(xF, kF, ttΔω, passA, Mass_a);
        
        num_photons = length(ϕf)
        sln_ConVL = sqrt.(π ./ dkdl.^2); # km
        passA2 = [func_use, MagnetoVars, Mass_a];
        if CLen_Scale
            weightC = ConvL_weights(xF, kF, vmag_tot, ttΔω, sln_ConVL, passA2)
        else
            weightC = ones(num_photons)
        end
        
        SaveAll[:, 1] .= view(θf, :);
        SaveAll[:, 2] .= view(ϕf,:);
        SaveAll[:, 3] .= view(θfX, :);
        SaveAll[:, 4] .= view(ϕfX, :);
        SaveAll[:, 5] .= sqrt.(sum(xF[:, :, end] .^2, dims=2))[:]; # r final
        
        
        Bvec, ωp = RT.GJ_Model_vec(SurfaceX, sln_t, θm, ωPul, B0, rNS);
        BperpV = Bvec .- kini .* sum(Bvec .* kini, dims=2) ./ sum(kini .^ 2, dims=2);
        Bperp = sqrt.(sum(BperpV .^ 2, dims=2)) .* (1.95e-20); # GeV^2
        
        rr = sqrt.(sum(SurfaceX .^2, dims=2));
        if !RadApprox
            cLen = 1.0 ./ dkdl[:]
        else
            cLen = 2.0 .* rr .* vmag_tot ./ (3.0 .* Mass_a) .* 6.56e-16 .* 2.998e5;
        end
        probab = π ./ vmag_tot.^2 .* (1e-12 .* Bperp).^2 .* cLen ./ (2.998e5 .* 6.58e-16 ./ 1e9).^2; # g [1e-12 GeV^-1], unitless
        
        dS = rr.^2 .* sin.(acos.(SurfaceX[:, 3] ./ rr));
        # assume number density at each point 1 / cm^3
        SaveAll[:, 6] .= 1.0 .* dS[:] .* vmag_tot[:].^3  .* probab[:] .* weightC[:] .^ 2 .* exp.(-opticalDepth[:]) .* (1e5).^2 .* 2.998e10; # num photons / second
        if !RadApprox
            SaveAll[:, 6] .*= ctheta[:];
        end
        SaveAll[:, 7] .= Δω[:];
    
    
        fileN = "results/Minicluster_Time_"*string(t_list[i])*"_MassAx_"*string(Mass_a);
        fileN *= "_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0)*"_rNS_";
        fileN *= string(rNS)*"_MassNS_"*string(Mass_NS);
        
        if NS_vel[1] != 0
            fileN *= "NS_velX_"*string(round(NS_vel[1], digits=4))
        end
        if NS_vel[2] != 0
            fileN *= "NS_velY_"*string(round(NS_vel[2], digits=4))
        end
        if NS_vel[3] != 0
            fileN *= "NS_velZ_"*string(round(NS_vel[3], digits=4))
        end
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
        phaseS[:, i] = dR[:, i] .* (Mass_a .* v_sur .- cos.(thetaZ) .* sqrt.( abs.((ωfull .^2 .-  ωpL .^ 2 ) ./ (1 .- cos.(thetaB) .^2 .* ωpL .^2 ./ ωfull .^2)))  )
        
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

