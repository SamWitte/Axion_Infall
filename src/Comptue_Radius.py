import numpy as np

nfw = True
M_MC = 1e-10
c=100

rho_MC = 1e5 # solar mass / pc^3

if nfw:
    f_c = np.log(1+c) - c / (1+c)
    r_s = (M_MC / (4*np.pi * rho_MC * f_c))**(1/3)
    r_t = c * r_s * 3.086e+13
    
else:
    r_t = (3 * M_MC / (4*np.pi * rho_MC))**(1/3) * 3.086e+13

print("Trunc radius [km]: ", r_t )

