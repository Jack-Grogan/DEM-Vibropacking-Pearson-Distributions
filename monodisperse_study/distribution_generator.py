# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:33:46 2024

@author: JACK GROGAN

      ___  ________  ________  ___  __            ________     
     |\  \|\   __  \|\   ____\|\  \|\  \         |\   ____\    
     \ \  \ \  \|\  \ \  \___|\ \  \/  /|_       \ \  \___|    
   __ \ \  \ \   __  \ \  \    \ \   ___  \       \ \  \  ___  
  |\  \\_\  \ \  \ \  \ \  \____\ \  \\ \  \       \ \  \|\  \ 
  \ \________\ \__\ \__\ \_______\ \__\\ \__\       \ \_______\
   \|________|\|__|\|__|\|_______|\|__| \|__|        \|_______|
   
"""

# imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.special import beta
import scipy.stats as stats
from natsort import natsorted

import gmsh
if gmsh.isInitialized():
    gmsh.finalize()
gmsh.initialize()

from jinja2 import Template
import os
import sympy
import glob
import toml
import re


# Total particle volume check

def Distribution_Sphere_Volume(x, p, N):
    if len(x) != len(p):
        return print('len(x) != len(p)')
    else: 
        p_frac = p/sum(p)
        volume = [0]*len(x)
        for i in range(len(x)):
            volume[i] = p_frac[i]*4/3*np.pi*(x[i]/2)**3*N
        volume_total = sum(volume)
        return volume_total
            
# Total particle volume calculator

def Distribution_Set_Volume_N(x, p, set_volume):
    if len(x) != len(p):
        return print('len(x) != len(p)')
    else: 
        p_frac = p/sum(p)
        volume = [0]*len(x)
        for i in range(len(x)):
            volume[i] = p_frac[i]*4/3*np.pi*(x[i]/2)**3
            
        volume_total = sum(volume)
        N = round(set_volume/volume_total)       
            
        return N

#
def Distribution_Discritiser(N_particles, height_data):
    particle_split = [0]*len(height_data)
    for i in range(len(height_data)):
        particle_split[i] = round(height_data[i]*N_particles)
    return particle_split

            
    
#
def pearson_pdf(x, mu, sigma, skew, kurt):
    
    # Huge help from the book Continuous Univariate Distributions Volume 1
    # Pearson's original paper https://sci-hub.hkvisa.net/10.1098/rsta.1895.0010
    # Huge help from Github https://github.com/mrsoltys/2C-PLIF/blob/master/Downloaded%20Matlab%20Scripts/pearspdf/pearspdf.m
    # Help from the Matlab page https://uk.mathworks.com/help/stats/pearson-distribution.html#mw_2399a80f-2773-430f-af91-ee6fa1e2c297
    
    if skew**2 >= kurt - 1:
        return 'Error skew^2 must be less than kurt - 1'

    # normalise the x values
    X0 = (x - mu)/sigma
    
    # calculate values for varience, beta 1 and beta 2
    #mu_2 = sigma**2
    
    beta1 = skew**2
    beta2 = kurt

    # coefficient values for calculating descriminant
    # data has been normalised so mu_2 = 1 and can ignore common denominator

    c0 = (4*beta2 - 3*beta1)  # /(10*beta2 - 12*beta1 - 18)*mu_2
    c1 = skew*(beta2+3)  # /(10*beta2 - 12*beta1 - 18)*np.sqrt(mu_2)
    c2 = (2*beta2 - 3*beta1 - 6)  # /(10*beta2 - 12*beta1 - 18)
    a = c1
    
    # determining the pearson distribution

    discrim = c1**2 - 4*c2*c0
    
    # Continuous Univariate Distributions Volume 1 conditions

    if c1 == c2 == 0:

        dist_type = 0
            
    elif beta1 == 0 and beta2 < 3:
        
        dist_type = 2

    elif (10*beta2 - 12*beta1 - 18) == 0:

        dist_type = 3

    elif beta1 == 0 and beta2 > 3:

        dist_type = 7
        
    else:
        
        kappa = c1**2/ (4*c0*c2)
        
        if kappa < 0:

            dist_type = 1
    
        elif 0 < kappa < 1:

            dist_type = 4
    
        elif kappa == 1:

            dist_type = 5
    
        elif kappa > 1:

            dist_type = 6
        
        
    # determining roots of Pearson denominator

    if discrim > 0:

        temp_a1 = (- c1 - np.sqrt(c1**2 - 4*c0*c2))/(2*c2)
        temp_a2 = (- c1 + np.sqrt(c1**2 - 4*c0*c2))/(2*c2)

        if temp_a1 <= temp_a2:
            a1 = temp_a1
            a2 = temp_a2
        else:
            a1 = temp_a2
            a2 = temp_a1

    elif discrim < 0:

        real1 = (- c1)/(2*c2)
        im1 = - np.sqrt(-(c1**2 - 4*c0*c2))/(2*c2)
        real2 = (- c1)/(2*c2)
        im2 = np.sqrt(-(c1**2 - 4*c0*c2))/(2*c2)

        a1 = complex(real1, im1)
        a2 = complex(real2, im2)


    # True values of pearson quadratic coefficients
    # denominator C0 + C1*x + C2*x**2
    
    denominator = 10*beta2 - 12*beta1 - 18

    if abs(denominator) > np.finfo(float).tiny:
        C0 = (4*beta2 - 3*beta1)/(denominator)#*mu_2
        C1 = skew*(beta2+3)/(denominator)#*np.sqrt(mu_2)
        C2 = (2*beta2 - 3*beta1 - 6)/(denominator)
        A = C1

    else:
        dist_type = 1


    # generating the distribution

    if dist_type == 0:
        p = 1/(np.sqrt(2*np.pi*C0))*np.exp(-(X0+A)**2/(2*C0))
        p = p/sigma

    elif dist_type == 1:
        
        if abs(denominator) > np.finfo(float).tiny:
            m1 = (A + a1)/(C2*(a2 - a1))
            m2 = -(A + a2)/(C2*(a2 - a1))
            
        else:
            m1 = a/(c2*(a2 - a1))
            m2 = -a/(c2*(a2 - a1))

        X = (X0-a1)/(a2-a1)
        p = stats.beta.pdf(X, m1+1, m2+1)
        p = p*(max(X) - min(X))/(max(x) - min(x))

    elif dist_type == 2:
        m = (A + a1)/(C2*2*abs(a1))
        X = (X0-a1)/(2*abs(a1))
        p = stats.beta.pdf(X, m+1, m+1)
        p = p*(max(X) - min(X))/(max(x) - min(x))

    elif dist_type == 3:
        p = stats.pearson3.pdf(X0, skew)

    elif dist_type == 4:
        
        m = 1/(2*C2)
        v = 2*C1*(1 - m)/(np.sqrt(4*C0*C2-C1**2))
        b = 2*(m-1)
        a = np.sqrt(b**2*(b-1)/(b**2 + v**2))
        lam = a*v/b
        
        X0 = (x - mu)/sigma
        X = X0 - lam/a
        
        p = abs(gamma(complex(m, v/2))/gamma(m))**2/(a*beta(m-1/2, 1/2))*(1+X**2)**(-m)*np.exp(-v*np.arctan(X))
        p = p*a/sigma


    elif dist_type == 5:
        root = C1/(2*C2)
        shape_param_a = 1/C2-1
        X = -((C1-root)/C2)/(X0 + root)
        p = stats.gamma.pdf(X, shape_param_a)
        p = p*(max(X) - min(X))/(max(x) - min(x))

    elif dist_type == 6:
        m1 = (a1 + C1)/(C2*(a2 - a1))
        m2 = -(a2 + C1)/(C2*(a2 - a1))
        if a2 < 0:
            nu1 = 2*(m2 + 1)
            nu2 = -2*(m1 + m2 + 1)
            X = (X0-a2)/(a2-a1)*(nu2/nu1)
            p = stats.f.pdf(X, nu1, nu2)
            p = p*(max(X) - min(X))/(max(x) - min(x))

        else:
            nu1 = 2*(m1 + 1)
            nu2 = -2*(m1 + m2 + 1)
            X = (X0-a1)/(a1-a2)*(nu2/nu1)
        p = stats.f.pdf(X, nu1, nu2)
        p = p*(max(X) - min(X))/(max(x) - min(x))

    elif dist_type == 7:
        nu = 1/C2 - 1
        X = X0/(np.sqrt(C0/(1-C2)))
        p = stats.t.pdf(X, nu)
        p = p*(max(X) - min(X))/(max(x) - min(x))

    else:
        print('--------------ERROR--------------')
    return p, dist_type


def pearson_array_intorfloat(x, mu, sigma, skew, kurt):
    
    if isinstance(skew, int):
        skew = float(skew)
        
    if isinstance(kurt, int):
        kurt = float(kurt)
        
    if isinstance(sigma, int):
        sigma = float(sigma)

    if isinstance(skew, list) and isinstance(kurt, float) and isinstance(sigma, float):
        skew = np.array(skew)
    elif isinstance(kurt, list) and isinstance(skew, float) and isinstance(sigma, float):  
        kurt = np.array(kurt)
    elif isinstance(sigma, list) and isinstance(skew, float) and isinstance(kurt, float):  
        sigma = np.array(sigma)

    if isinstance(skew, np.ndarray) and isinstance(kurt, float) and isinstance(sigma, float):
        p = [0]*len(skew)
        dist_type = [0]*len(skew)
        for i in range(len(skew)):
            p[i], dist_type[i] = pearson_pdf(x, mu, sigma, skew[i], kurt)            
    elif isinstance(kurt, np.ndarray) and isinstance(skew, float) and isinstance(sigma, float):  
        p = [0]*len(kurt)
        dist_type = [0]*len(kurt)
        for i in range(len(kurt)):
            p[i], dist_type[i] = pearson_pdf(x, mu, sigma, skew, kurt[i])
    elif isinstance(sigma, np.ndarray) and isinstance(skew, float) and isinstance(kurt, float):  
        p = [0]*len(sigma)
        dist_type = [0]*len(sigma)
        for i in range(len(sigma)):
            p[i], dist_type[i] = pearson_pdf(x, mu, sigma[i], skew, kurt)
            
    return p, dist_type


def constant_N_distribution(p, N_particles):
    
    if isinstance(p, list):
        N_particle_frac_raw = [0]*len(p)
        p_N = [0]*len(p)
        
        for i in range(len(p)):
            N_particle_frac_raw[i] = p[i]/sum(p[i])
            p_N[i] = Distribution_Discritiser(N_particles, N_particle_frac_raw[i])
            p_N[i] = np.array(p_N[i])
            
    elif isinstance(p, np.ndarray):
        
        N_particle_frac_raw = p/sum(p)
        p_N = Distribution_Discritiser(N_particles, N_particle_frac_raw)
        p_N = np.array(p_N)

    return p_N

        
def constant_V_distribution(x, p, mu, sigma, N_particles):
    
    p_norm, _ =  pearson_pdf(x, mu, sigma, 0, 3)
    norm_V = Distribution_Sphere_Volume(x, p_norm, N_particles)
    
    const_vol_N = [0]*len(p)
    N_particle_frac_raw = [0]*len(p)

    for i in range(len(p)):
        N_particle_frac_raw[i] = p[i]/sum(p[i])
        const_vol_N[i] = Distribution_Set_Volume_N(x, N_particle_frac_raw[i], norm_V)
            
    p_V = [0]*len(const_vol_N)

    for i in range(len(const_vol_N)):
        p_V[i] = Distribution_Discritiser(const_vol_N[i], N_particle_frac_raw[i])
        p_V[i] = np.array(p_V[i])

    return p_V, const_vol_N, norm_V
        

#------------------------------------------------------------------------------------------
# Setting Distribution Parameters 
#------------------------------------------------------------------------------------------

# N particles from https://asmedigitalcollection.asme.org/manufacturingscience/article/144/12/121013/1143462/Effect-of-Horizontal-Vibrations-and-Particle-Size?casa_token=ngvwpp9GbTsAAAAA:XboodU-Q70_6CGyNC6uAjDN9EZK1RT_YVnv_ZOiMAewufcqj4NXpsQ8rw9V1XGW_g0aW-u8
N_particles = 22000
mu = 0.01 #m diameter

n_bars = 21

sigma = 1/330

x_range = 3*1/330 # This ensures that changes in sigma do not result in too small particle sizes

x_bar = np.linspace(mu-x_range, mu+x_range, n_bars)
x_cont = np.linspace(mu-x_range, mu+x_range, 200)

p_bar = [0]*len(x_bar)
p_bar[int((len(x_bar) - 1)/2)] = N_particles
p_bar = [np.asarray(p_bar)]
#------------------------------------------------------------------------------------------
# Generating Normal Distribution for standardising particle volume
#------------------------------------------------------------------------------------------

norm_cont, norm_type_cont =  pearson_pdf(x_cont, mu, 1/330, 0, 3)
norm_bar, norm_type_bar =  pearson_pdf(x_bar, mu, 1/330, 0, 3)

#------------------------------------------------------------------------------------------
#  Distribution of particle fractions for constant number of particles
#------------------------------------------------------------------------------------------

N_norm_particle_split = constant_N_distribution(norm_bar, N_particles)

#------------------------------------------------------------------------------------------
#  Distribution of particle fractions for constant volume of equivelent normal distibution
#------------------------------------------------------------------------------------------

V_particle_split_bar, const_vol_N, volume = constant_V_distribution(x_bar, p_bar, mu, sigma, N_particles)

#------------------------------------------------------------------------------------------
# LIGGGHTS script generator
#------------------------------------------------------------------------------------------

particle_radii = x_bar/2

particle_split = V_particle_split_bar[0]/sum(V_particle_split_bar[0])

# cylinder dimensions as found in https://asmedigitalcollection.asme.org/manufacturingscience/article/144/12/121013/1143462/Effect-of-Horizontal-Vibrations-and-Particle-Size?casa_token=ngvwpp9GbTsAAAAA:XboodU-Q70_6CGyNC6uAjDN9EZK1RT_YVnv_ZOiMAewufcqj4NXpsQ8rw9V1XGW_g0aW-u8
z0_cylinder         = 0                                     #m
x0_cylinder         = 0                                     #m
y0_cylinder         = 0                                     #m
lc_cylinder         = 1e-2                                  #-
cylinder_height     = 90*max(x_bar)                         #m
cylinder_radii      = 6*max(x_bar)                         #m

# insertion volume
z0_ins_vol = z0_cylinder + max(x_bar)/2                     #m
zh_ins_vol = cylinder_height - max(x_bar)/2                 #m
x0_ins_vol = x0_cylinder                                    #m                       
y0_ins_vol = y0_cylinder                                    #m
radii_ins_vol = cylinder_radii - max(x_bar)/2               #m

# Domain Dimensions
domain_x_min = -2*cylinder_radii
domain_x_max =  2*cylinder_radii
domain_y_min = -2*cylinder_radii
domain_y_max =  2*cylinder_radii
domain_z_min = -0.5*cylinder_height
domain_z_max = 1.5*cylinder_height


# Mesh Configuration
cylinder_mesh_max   = 0.02                                  #-
cylinder_mesh_min   = 0.0                                   #-


#------------------------------------------------------------------------------------------
# Simulation Specifications
#------------------------------------------------------------------------------------------


timestep            = 79e-7                                 #s
dumptime            = 0.079                                 #s
ontime              = 12.5                                  #s
filltime            = 2                                     #s
settletime          = 1                                     #s
w                   = 200                                   #rad/s

x_amplitude         = 0.1*mu                                #m 
y_amplitude         = 0                                     #m 
z_amplitude         = 0                                     #m


# set volume equal constant
N                   = const_vol_N                           #-
density             = 2500                                  #kg/m^3
youngs_modulus      = 1e7                                   #Pa
poisson_ratio       = 0.29                                  #-

sliding_pw          = 0.3                                   #-
sliding_pp          = 0.3                                   #-
rolling_pw          = 0.002                                 #-
rolling_pp          = 0.002                                 #-
restitution_pw      = 0.922                                 #-
restitution_pp      = 0.922                                 #-
cohesion_pw         = 0                                     #-
cohesion_pp         = 0                                     #-

number_of_seeds     = 5

#------------------------------------------------------------------------------------------
# Slurm Job Launch Specifications
#------------------------------------------------------------------------------------------

job_runtime         = "10:00:00"        # hr:min:sec
ntasks              = int(8)

#------------------------------------------------------------------------------------------
# Preliminary Calculations
#------------------------------------------------------------------------------------------

input_ontime        = np.round(np.ceil(ontime/timestep)*timestep,8)
input_filltime      = np.round(np.ceil(filltime/timestep)*timestep,8)
input_settletime    = np.round(np.ceil(settletime/timestep)*timestep,8)
oscillation_period  = 2*np.pi/w

    
seeds = [0]*number_of_seeds

for k in range(number_of_seeds):
    seeds[k] = sympy.prime(2000 + k)

#------------------------------------------------------------------------------------------
# Open Files
#------------------------------------------------------------------------------------------

with open(os.path.join("template", "cylinder_template.geo"), 'r') as f:
    cylinder_template = f.read()

with open(os.path.join("template","shake_template.sim"), 'r') as f:
    simulation_template = f.read()

with open(os.path.join("template","batch_launch_template.sh"), 'r') as f:
    batch_launch_template = f.read()

#------------------------------------------------------------------------------------------
# Study File Generation
#------------------------------------------------------------------------------------------
for seed in seeds:

    # setting up jinja dictionaries for text replacement of tempalte files 

    cylinder_mesh_data      = { "cylinder_height": cylinder_height,
                                "cylinder_radius": cylinder_radii,
                                "z_0": z0_cylinder,
                                "y_0": y0_cylinder,
                                "x_0": x0_cylinder,
                                "lc": lc_cylinder,
                                "mesh_max": cylinder_mesh_max,
                                "mesh_min": cylinder_mesh_min,
                                "Algor": 6
                                }

    simulation_data         = { "timestep": timestep,
                                "dumptime": dumptime,
                                "filltime": input_filltime,
                                "ontime": input_ontime,
                                "settletime": input_settletime,
                                "number_particles": N[0],
                                "density": density,
                                "youngs_modulus": youngs_modulus,
                                "poisson_ratio": poisson_ratio,
                                "sliding_pp": sliding_pp,
                                "sliding_pw": sliding_pw,
                                "rolling_pp": rolling_pp,
                                "rolling_pw": rolling_pw,
                                "restitution_pp": restitution_pp,
                                "restitution_pw": restitution_pw,
                                "cohesion_pp": cohesion_pp,
                                "cohesion_pw": cohesion_pw,
                                "seed": seed,
                                "x_amplitude": x_amplitude,
                                "y_amplitude": y_amplitude,
                                "z_amplitude": z_amplitude,
                                "period": oscillation_period,
                                "z0_ins_vol": z0_ins_vol,
                                "zh_ins_vol": zh_ins_vol,
                                "x0_ins_vol": x0_ins_vol,
                                "y0_ins_vol": y0_ins_vol,
                                "radii_ins_vol": radii_ins_vol,
                                "domain_x_min": domain_x_min,
                                "domain_x_max": domain_x_max,
                                "domain_y_min": domain_y_min,
                                "domain_y_max": domain_y_max,
                                "domain_z_min": domain_z_min,
                                "domain_z_max": domain_z_max
                                }

    radii_data              = {f"R_{int(j)}": round(particle_radii[j],8) for j in range(len(particle_radii))}
    fractional_data         = {f"F_{int(k)}": round(particle_split[k],11) for k in range(len(particle_split))}  

    simulation_data.update(radii_data)
    simulation_data.update(fractional_data)

    batch_launch_data       = { "runtime": job_runtime,
                                "ntasks": ntasks
                                }

    post_processing_data  = simulation_data
    post_processing_data.update(cylinder_mesh_data)
    
    j2_cylinder_template         = Template(cylinder_template)
    j2_simulation_template       = Template(simulation_template)
    j2_batch_launch_template     = Template(batch_launch_template)

    # Creating file paths

    study_directory = os.path.join(f"mean_particle_diameter_{mu*1000:.0f}_mm", f"seed_{seed:.0f}")
    if not os.path.exists(study_directory):
        os.makedirs(study_directory)

    simulation_newpath = os.path.join(study_directory, f"diameter_{mu:.2f}")
    if not os.path.exists(simulation_newpath):
        os.makedirs(simulation_newpath)

    batch_launch_newpath = os.path.join(study_directory, f"diameter_{mu:.2f}")
    if not os.path.exists(batch_launch_newpath):
        os.makedirs(batch_launch_newpath)

    mesh_newpath = os.path.join(study_directory, f"diameter_{mu:.2f}", "mesh")
    if not os.path.exists(mesh_newpath):
        os.makedirs(mesh_newpath)

    toml_path = os.path.join(study_directory, f"diameter_{mu:.2f}")
    if not os.path.exists(toml_path):
        os.makedirs(toml_path)

    # Creating new simulation files

    cylinder_geo_file_new = os.path.join(mesh_newpath, "shake_cylinder.geo")
    simulation_file_new = os.path.join(simulation_newpath, "shake.sim")
    batch_launch_file_new = os.path.join(batch_launch_newpath, "batch_launch.sh")
    toml_file = os.path.join(toml_path, "simulation_data.toml")

    # Writing parameters to new files

    with open(cylinder_geo_file_new, 'w') as f:
        f.write(j2_cylinder_template.render(cylinder_mesh_data))

    with open(simulation_file_new, 'w') as f:
        f.write(j2_simulation_template.render(simulation_data))

    with open(batch_launch_file_new, 'w') as f:
        f.write(j2_batch_launch_template.render(batch_launch_data))

    with open(toml_file, 'w') as f:
        toml.dump(post_processing_data, f)

    # saving plot of distribution studied

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 6))


    ax1.plot(x_bar, V_particle_split_bar[0])
    ax1.set_title('Constant Volume Distributon')
    ax1.grid()
    ax1.grid(which='minor', alpha=0.2)
    ax1.minorticks_on()

    ax2.plot(particle_radii, particle_split)
    ax2.grid()
    ax2.grid(which='minor', alpha=0.2)
    ax2.minorticks_on()
    ax2.set_title('LIGGGHTS Input Distributon')

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.savefig(os.path.join(simulation_newpath,"LIGGGHTS_input_distribution"))

    # Creating stl files from the gmesh geo files

    gmsh.open(cylinder_geo_file_new)

    #------------------------------------------------------------------------------------------
    # Launch Slurm Jobs
    #------------------------------------------------------------------------------------------
    
    glob_input = os.path.join(study_directory, "diameter_*")
    directories = natsorted([k for k in glob.glob(glob_input)])
    directories = sorted(directories, key=lambda x:float(re.findall("(-?\d+\.\d*)",x)[0]))
    
    for directory in directories:
        launch_file = os.path.join(directory, "batch_launch.sh")
        cmd = f"sbatch --output={directory}/slurm-%j.out {launch_file} {directory}"
        print(cmd)
        os.system(cmd)