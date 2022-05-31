import numpy as np

nfw = True
M_MC = 1.0e-13
c=100

axion_star = False
m_ax = 2.6e-5

rho_MC = 1e5 # solar mass / pc^3

if nfw and not axion_star:
    f_c = np.log(1+c) - c / (1+c)
    r_s = (M_MC / (4*np.pi * rho_MC * f_c))**(1/3)
    r_t = c * r_s * 3.086e+13
    
elif not nfw and not axion_star:
    r_t = (3 * M_MC / (4*np.pi * rho_MC))**(1/3) * 3.086e+13

elif axion_star:
    # gustavo calculated, ma [1e-5 eV units], M_MC in 1e-10 solar mass
    r_t = 26.4 / ((m_ax/ 1e-5)**2 * (M_MC / 1e-10))
        
print("Trunc radius [km]: ", r_t )

