__precompile__()
using NPZ

function find_init_conds(finalX, b, M, v_NS)
    # b is impact param, M is mass axion minicluster solar mass
    # for the time being, i will assume -v_NS in zhat direction and impact parameter in x hat -- can be generalized in future

    R_amc = 1.4e8 .* (M / 1e-10) .^ (1.0 ./ 3.0) # km
    # assume misaligned in xhat direction
    finalX[:, 1] .-= b
    rho_sf = sqrt.(finalX[:, 1].^2 .+ finalX[:, 2].^2)
    subpts = finalX[rho_sf .< R_amc, :]
    first_pt = subpts[argmin(subpts[:, 3]), :]
    if (R_amc.^2 >(first_pt[1].^2 .+ first_pt[2].^2))
        z0 = -sqrt.(R_amc.^2 .- (first_pt[1].^2 .+ first_pt[2].^2)) .+ first_pt[3]
    else

        z0 = -sqrt.(-R_amc.^2 .+ (first_pt[1].^2 .+ first_pt[2].^2)) .+ first_pt[3]
    end
    return [b 0 -z0]
end

function max_time(finalX, b, M, v_NS)
    if M < 1e-20
        R_amc = 1.4e8 .* (M / 1e-10) .^ (1.0 ./ 3.0) # km
        # assume misaligned in xhat direction
        finalX[:, 1] .-= b
        rho_sf = sqrt.(finalX[:, 1].^2 .+ finalX[:, 2].^2)
        subpts = finalX[rho_sf .< R_amc, :]
        first_pt = subpts[argmin(subpts[:, 3]), :]
        last_pt = subpts[argmax(subpts[:, 3]), :]
        time = (last_pt[3] .- first_pt[3]) ./ v_NS[3]
    else
        R_amc = 1.4e8 .* (M / 1e-10) .^ (1.0 ./ 3.0) # km
        yy = sqrt.(R_amc .^ 2 .- b.^2)
        time = 2 * yy /  v_NS[3]
    end
    return time # seconds
end

function density_eval(finalX, init_conds, M, rho_amc, v_NS, t)
    # assumes powerlaw profile
    # rho_amc = M_dot / pc^3
	 
    central_amc = -init_conds .+ v_NS .* t
    
    dist = sqrt.(sum( (finalX .- central_amc) .^ 2 , dims=2))
     
    R_amc = 1.4e8 .* (M / 1e-10) .^ (1.0 ./ 3.0) #	km
    density = zeros(length(finalX[:, 1]))
    rho_0 = R_amc.^(9.0/4.0) ./ 4.0 .* rho_amc

    density[dist[:] .< R_amc] .= rho_0 ./ dist[dist[:] .< R_amc].^(9.0./4.0)
    return density # units: Solar mass / pc^3
end

function scan_mc_evolution(fileTail, b, M, rho_amc, v_NS, tlist; sve=sve)
    fileName = "temp_storage/FinalX_"*fileTail
    finalX = npzread(fileName)
    init_Cs = find_init_conds(finalX, b, M, v_NS)
    density_full = zeros(length(finalX[:,1]), length(tlist))
    for i in 1:length(tlist)
        density_full[:,i] .= density_eval(finalX, init_Cs, M, rho_amc, v_NS, tlist[i])
    end
    
    if sve
        fileN = "temp_storage/TimeEvol_Rho_Mmc_"*string(M)*"_rhomc_"*string(rho_amc)*"_impactP_"*string(b)*"_"
        fileN*=fileTail
        npzwrite(fileN, vcat(tlist, transpose(density_full[1,:])))
        
    else
        return vcat(tlist, transpose(density_full[1,:]))
    end
end
