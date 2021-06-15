__precompile__()
using NPZ


function max_time(b, M, v_NS)
    # functions under assumption that R_amc >> target surface (should be true for M > 1e-20 Solar Mass)
    R_amc = 1.4e8 .* (M / 1e-10) .^ (1.0 ./ 3.0) # km
    yy = sqrt.(R_amc .^ 2 .- b.^2)
    
    time = 2 * yy / sqrt.( sum(v_NS .^2))
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

