"""
@author: Jack Richard Grogan

      ___  ________  ________  ___  __            ________
     |\  \|\   __  \|\   ____\|\  \|\  \         |\   ____\
     \ \  \ \  \|\  \ \  \___|\ \  \/  /|_       \ \  \___|
   __ \ \  \ \   __  \ \  \    \ \   ___  \       \ \  \  ___
  |\  \\_\  \ \  \ \  \ \  \____\ \  \\ \  \       \ \  \|\  \
  \ \________\ \__\ \__\ \_______\ \__\\ \__\       \ \_______\
   \|________|\|__|\|__|\|_______|\|__| \|__|        \|_______|

"""

import numpy as np
import pyvista as pv
import os
import glob
from natsort import natsorted
import re
from collections import Counter
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.special import beta
import scipy.stats as stats
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib import patches
from matplotlib.patches import Ellipse
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        
        X = X0 - lam/a
        
        p = abs(gamma(complex(m, v/2))/gamma(m))**2/(a*beta(m-1/2, 1/2))*(1+X**2)**(-m)*np.exp(-v*np.arctan(X))
        p = p*a/sigma#*(max(X) - min(X))/(max(x) - min(x))


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
        

def list_elements_equal(lst):
    repeated = list(np.ones(len(lst))*lst[0])
    return all(repeated == lst)


#------------------------------------------------------------------------------------------
# Setting Distribution Parameters 
#------------------------------------------------------------------------------------------

# N particles from https://asmedigitalcollection.asme.org/manufacturingscience/article/144/12/121013/1143462/Effect-of-Horizontal-Vibrations-and-Particle-Size?casa_token=ngvwpp9GbTsAAAAA:XboodU-Q70_6CGyNC6uAjDN9EZK1RT_YVnv_ZOiMAewufcqj4NXpsQ8rw9V1XGW_g0aW-u8
N_particles = 22000
mu = 0.01 #m diameter

N_curves_1 = 7
N_curves_2 = 14
N_curves = N_curves_1 + N_curves_2
n_bars = 21

# skew^2 must be less than kurt - 1
sigma = 1/330 #m
skew = 0
kurt = np.concatenate((np.linspace(1.8, 3, N_curves_1), np.linspace(4, 17, N_curves_2)))

x_range = 3*1/330 # This ensures that changes in sigma do not result in too small particle sizes

x_bar = np.linspace(mu-x_range, mu+x_range, n_bars)
x_cont = np.linspace(mu-x_range, mu+x_range, 200)

#------------------------------------------------------------------------------------------
# Graphing inputs 
#------------------------------------------------------------------------------------------

position_1  = 0
position_2  = 1
position_3  = 10
position_4  = 20

positions = [position_1, position_2, position_3, position_4]
colourmap = matplotlib.colors.LinearSegmentedColormap.from_list("",['#1e90ff', '#3a7edf', '#566cbf', '#725a9f', '#8f4880', '#ab3660', '#c72440', '#e31220', '#ff0000'])

#------------------------------------------------------------------------------------------
# Plotting Distribution for one of skew and kurt as array  
#------------------------------------------------------------------------------------------

p_bar, dist_type_bar =  pearson_array_intorfloat(x_bar, mu, sigma, skew, kurt)
p_cont, dist_type_cont =  pearson_array_intorfloat(x_cont, mu, sigma, skew, kurt)

#------------------------------------------------------------------------------------------
# Generating Normal Distribution for standardising particle volume
#------------------------------------------------------------------------------------------

norm_cont, norm_type_cont =  pearson_pdf(x_cont, mu, 1/330, 0, 3)
norm_bar, norm_type_bar =  pearson_pdf(x_bar, mu, 1/330, 0, 3)

#------------------------------------------------------------------------------------------
#  Distribution of particle fractions for constant number of particles
#------------------------------------------------------------------------------------------

N_particle_split_bar = constant_N_distribution(p_bar, N_particles)
N_norm_particle_split = constant_N_distribution(norm_bar, N_particles)

#------------------------------------------------------------------------------------------
#  Distribution of particle fractions for constant volume of equivelent normal distibution
#------------------------------------------------------------------------------------------

V_particle_split_bar, const_vol_N, volume = constant_V_distribution(x_bar, p_bar, mu, sigma, N_particles)

#------------------------------------------------------------------------------------------
#  Extracting data from simulations 
#------------------------------------------------------------------------------------------

study = os.path.join("mean_particle_diameter_10_mm")


glob_input_study = os.path.join(study, "seed_*")
seeds = natsorted([k for k in glob.glob(glob_input_study)])

kurtosis_seed = [0]*len(seeds)

for j in range(len(seeds)):
    
    glob_input_directories = os.path.join(seeds[j], "kurtosis_*")
    directories = natsorted([k for k in glob.glob(glob_input_directories)])
    directories = sorted(directories, key=lambda x:float(re.findall("(-?\d+\.\d*)",x)[0]))

    kurtosis_seed[j] = directories

kurtosis_seed = np.asarray(kurtosis_seed).T

counted_radii_data_bank = [0]*len(kurtosis_seed)
V_particle_split_bar_dict = [0]*len(kurtosis_seed)
theoretical_dict = [0]*len(kurtosis_seed)
theoretical_dict_values_array  = [0]*len(kurtosis_seed)
particle_radii_input  = [0]*len(kurtosis_seed)
max_percentage_difference_array = [0]*len(kurtosis_seed)
min_percentage_difference_array = [0]*len(kurtosis_seed)
max_difference_array = [0]*len(kurtosis_seed)
min_difference_array = [0]*len(kurtosis_seed)
mean_percentage_difference_array = [0]*len(kurtosis_seed)

for i in positions:
    
    columns = [0]*len(kurtosis_seed[i])
    counted_radii_data = [0]*len(kurtosis_seed[i])
    
    particle_percentage_difference = [0]*len(kurtosis_seed[i])
    particle_percentage_difference_array = [0]*len(kurtosis_seed[i])
    
    particle_difference_array = [0]*len(kurtosis_seed[i])

    particle_radii_array = [0]*len(kurtosis_seed[i])
    
    V_particle_split_bar_dict[i] = {}

    for o in range(len(x_bar)):
        V_particle_split_bar_dict[i][round(x_bar[o]/2, 8)] = V_particle_split_bar[i][o]
            
    for j in range(len(kurtosis_seed[i])):
    
        columns[j] = kurtosis_seed[i][j].split('\\')[1]
        
        glob_input = os.path.join(kurtosis_seed[i][j], "post", "particles_*")
        files = natsorted([f for f in glob.glob(glob_input) if "boundingBox" not in f])
        end_file = files[-1]
        
        data = pv.read(end_file)
        radii_data = data["radius"]
        
        counted_radii_data[j] = dict(Counter(radii_data))
        counted_radii_data[j] = {k: v for k, v in sorted(counted_radii_data[j].items(), key=lambda item: item[0])}
          
        for key in V_particle_split_bar_dict[i]:
            if key not in list(counted_radii_data[j].keys()):
                missed_data = {key: 0}
                counted_radii_data[j].update(missed_data)
                counted_radii_data[j] = dict(sorted(counted_radii_data[j].items()))

        particle_percentage_difference[j] = {key: counted_radii_data[j][key] - V_particle_split_bar_dict[i].get(key, 0) for key in counted_radii_data[j] }
        theoretical_dict[i] = {key: V_particle_split_bar_dict[i].get(key, 0) for key in counted_radii_data[j] }
        
        particle_radii_array[j] = list(particle_percentage_difference[j].keys())
        
        particle_percentage_difference_array[j] = list(particle_percentage_difference[j].values())
        
        theoretical_dict_values_array[i] = list(theoretical_dict[i].values())
        particle_difference_array[j] = list(counted_radii_data[j].values())

    particle_percentage_difference_array = np.asarray(particle_percentage_difference_array).T
    particle_difference_array = np.asarray(particle_difference_array).T
    
    particle_radii_array = np.asarray(particle_radii_array).T
    
    particle_radii_input[i] = [0]*len(particle_radii_array)
    for z in range(len(particle_radii_array)):
        if list_elements_equal(particle_radii_array[z]):
            particle_radii_input[i][z] = particle_radii_array[z][0]
        else:
            print('consistancy error')

    max_percentage_difference_array[i] = [max(t) for t in particle_percentage_difference_array]
    min_percentage_difference_array[i] = [min(t) for t in particle_percentage_difference_array]
    
    max_difference_array[i] = [max(t) for t in particle_difference_array]
    min_difference_array[i] = [min(t) for t in particle_difference_array]
    
    mean_percentage_difference_array[i] = [np.mean(t) for t in particle_percentage_difference_array]
    counted_radii_data_bank[i] = counted_radii_data
    
#------------------------------------------------------------------------------------------
# Plotting distributions
#------------------------------------------------------------------------------------------

# Setting out figure dimensions
fig, _ = plt.subplots(figsize =(14,10), dpi=600)

gs = gridspec.GridSpec(4, 2)
ax1 = plt.subplot(gs[0, 1])
ax2 = plt.subplot(gs[1, 1])
ax3 = plt.subplot(gs[2, 1])
ax4 = plt.subplot(gs[3, 1])
ax5 = plt.subplot(gs[:, 0])

small_axs = [ax1, ax2, ax3, ax4]

ax1a = ax1.twinx()
ax2a = ax2.twinx()
ax3a = ax3.twinx()
ax4a = ax4.twinx()

small_axs_a = [ax1a, ax2a, ax3a, ax4a]
ax1.get_shared_x_axes().join(ax1, ax2, ax3, ax4)

# Assigning x and y data values
y_data_1 = list(V_particle_split_bar_dict[position_1].values())
y_data_2 = list(V_particle_split_bar_dict[position_2].values())
y_data_3 = list(V_particle_split_bar_dict[position_3].values())
y_data_4 = list(V_particle_split_bar_dict[position_4].values())

x_data_1 = list(V_particle_split_bar_dict[position_1].keys())
x_data_2 = list(V_particle_split_bar_dict[position_2].keys())
x_data_3 = list(V_particle_split_bar_dict[position_3].keys())
x_data_4 = list(V_particle_split_bar_dict[position_4].keys())

# Asigning bar chart widths
width_1 = (max(x_data_1) - min(x_data_1))/len(x_data_1)*0.8
width_2 = (max(x_data_2) - min(x_data_2))/len(x_data_2)*0.8
width_3 = (max(x_data_3) - min(x_data_3))/len(x_data_3)*0.8
width_4 = (max(x_data_4) - min(x_data_4))/len(x_data_4)*0.8

ax1a.fill_between(particle_radii_input[position_1], min_percentage_difference_array[position_1], max_percentage_difference_array[position_1], color = 'k', alpha = 0.15)
ax1a.plot(particle_radii_input[position_1], mean_percentage_difference_array[position_1], color = 'k', alpha = 0.15)

ax2a.fill_between(particle_radii_input[position_2], min_percentage_difference_array[position_2], max_percentage_difference_array[position_2], color = 'k', alpha = 0.15)
ax2a.plot(particle_radii_input[position_2], mean_percentage_difference_array[position_2], color = 'k', alpha = 0.15)

ax3a.fill_between(particle_radii_input[position_3], min_percentage_difference_array[position_3], max_percentage_difference_array[position_3], color = 'k', alpha = 0.15)
ax3a.plot(particle_radii_input[position_3], mean_percentage_difference_array[position_3], color = 'k', alpha = 0.15)

ax4a.fill_between(particle_radii_input[position_4], min_percentage_difference_array[position_4], max_percentage_difference_array[position_4], color = 'k', alpha = 0.15)
ax4a.plot(particle_radii_input[position_4], mean_percentage_difference_array[position_4], color = 'k', alpha = 0.15)

# Subplot formatting 
for ax in small_axs:
    ax.set_ylim(0, 5000)
    ax.set_ylabel("Number of Particles (-)")
    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.set_xticks(particle_radii_input[position_4], labels = particle_radii_input[position_4], rotation = 70)
    ax.grid(color ='grey',
    		linestyle ='-.', linewidth = 0.5,
    		alpha = 0.4)
    for spine in ['top', 'bottom', 'left', 'right']:
    	ax.spines[spine].set_visible(False)
        
for axa in small_axs_a:
    axa.set_ylabel("Particle Insertion Error (-)")
    axa.set_ylim(-10, 10)
    axa.yaxis.set_major_locator(MultipleLocator(2))
    axa.grid(color ='grey',
    		linestyle ='--', linewidth = 0.5,
    		alpha = 0.4)
    for spine in ['top', 'bottom', 'left', 'right']:
    	axa.spines[spine].set_visible(False)
        

ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xlabel("Particle Radii (m)", fontsize = 11.5)

# -----------------------------------------------------------------------------
# ax5
# -----------------------------------------------------------------------------

# Reading in data

df = pd.read_csv(r"final_packing_results.csv")
df.rename({'Unnamed: 0': 'kurtosis'}, axis=1, inplace=True)
df_kurtosis = np.asarray(df['kurtosis'])

repeat_columns = [column for column in df.columns if column.startswith("seed")]
df_repeat_T = df[repeat_columns].T

# Converting data into arrays

repeats = [0]*len(df_kurtosis)

for i in range(len(df_kurtosis)):
    repeats[i] = np.asarray(df_repeat_T[i])
    
repeat_data = [0]*len(repeats[0])

for i in range(len(repeats[0])):
    repeat_data[i] = np.asarray(df[repeat_columns[i]])

scatter           = []
scatter_x_points  = []

for i in repeat_data:
    for j in i:
        scatter.append(j)

for i in range(len(repeat_data)):
    for j in df_kurtosis:
        scatter_x_points.append(j)
        
mean = []
std  = []

for i in repeats:
    mean.append(np.mean(i))
    std.append(np.std(i))
    
    
# Creating shade of +- 1 standard deviation either side of the mean

top_line = np.asarray(mean) + np.asarray(std)
bottom_line = np.asarray(mean) - np.asarray(std)

s = ax5.scatter(scatter_x_points, scatter, c=scatter, cmap=colourmap, marker = 'x')

intopolarion_runs = 50
interpolation_number = np.linspace(0,(1 - 1/intopolarion_runs), intopolarion_runs)


intoplolated_mean = []
interpolated_kurt = []
interpolated_topline = []
interpolated_bottomline = []


for m in range(len(mean)):
    if mean[m] != mean[-1]:
        for i in range(len(interpolation_number)):
            interpolation_mean = mean[m] + (mean[m+1] - mean[m])*interpolation_number[i]
            intoplolated_mean.append(interpolation_mean)
            
            interpolation_kurt = df_kurtosis[m] + (df_kurtosis[m+1] - df_kurtosis[m])*interpolation_number[i]
            interpolated_kurt.append(interpolation_kurt)
    
            interpolation_topline = top_line[m] + (top_line[m+1] - top_line[m])*interpolation_number[i]
            interpolated_topline.append(interpolation_topline)
    
            interpolation_bottomline = bottom_line[m] + (bottom_line[m+1] - bottom_line[m])*interpolation_number[i]
            interpolated_bottomline.append(interpolation_bottomline)
    else:
        intoplolated_mean.append(mean[m])
        interpolated_kurt.append(df_kurtosis[m])
        interpolated_topline.append(top_line[m])
        interpolated_bottomline.append(bottom_line[m])


colors = plt.get_cmap(colourmap)((intoplolated_mean - min(intoplolated_mean))/(max(intoplolated_mean) - min(mean)))

# colour mapping lines and shade on ax5
col_lines_ax5 = [0]*len(intoplolated_mean)
for i in range(len(intoplolated_mean)):
    i = i+2
    col_lines_ax5[i-2] = ax5.plot(interpolated_kurt[i-2:i], intoplolated_mean[i-2:i], color = colors[i-2], linewidth = 2)
    ax5.fill_between(interpolated_kurt[i-2:i], interpolated_topline[i-2:i], interpolated_bottomline[i-2:i], color=colors[i-2], alpha=.05)


# clear reference plots
p, = ax5.plot(df_kurtosis, mean, color = colors[0], linewidth = 0.7, alpha = 0)
f  = ax5.fill_between(df_kurtosis, top_line, bottom_line, color=colors[0], alpha=.0)#, label = '1 Standard Deviation')


#plotting elipses 
elipse_width_1 = 1

if abs(max(repeats[position_1]) - mean[position_1]) >  abs(min(repeats[position_1]) - mean[position_1]):
    height_1 = 2.2*abs(max(repeats[position_1]) - mean[position_1])
else:
    height_1 = 2.2*abs(min(repeats[position_1]) - mean[position_1])

ellipse_1 = Ellipse((kurt[position_1], mean[position_1]), elipse_width_1, height_1, angle=0, linewidth=1.5, fill=False, alpha = 1)
ax5.add_artist(ellipse_1)

elipse_width_2 = 1

if abs(max(repeats[position_2]) - mean[position_2]) >  abs(min(repeats[position_2]) - mean[position_2]):
    height_2 = 2.2*abs(max(repeats[position_2]) - mean[position_2])
else:
    height_2 = 2.2*abs(min(repeats[position_2]) - mean[position_2])
    
ellipse_2 = Ellipse((kurt[position_2], mean[position_2]), elipse_width_2, height_2, angle=0, linewidth=1.5, fill=False, alpha = 1)
ax5.add_artist(ellipse_2)

elipse_width_3 = 1

if abs(max(repeats[position_3]) - mean[position_3]) >  abs(min(repeats[position_3]) - mean[position_3]):
    height_3 = 2.2*abs(max(repeats[position_3]) - mean[position_3])
else:
    height_3 = 2.2*abs(min(repeats[position_3]) - mean[position_3])
    
ellipse_3 = Ellipse((kurt[position_3], mean[position_3]), elipse_width_3, height_3, angle=0, linewidth=1.5, fill=False, alpha = 1)
ax5.add_artist(ellipse_3)

elipse_width_4 = 1

if abs(max(repeats[position_4]) - mean[position_4]) >  abs(min(repeats[position_4]) - mean[position_4]):
    height_4 = 2.2*abs(max(repeats[position_4]) - mean[position_4])
else:
    height_4 = 2.2*abs(min(repeats[position_4]) - mean[position_4])
    
ellipse_4 = Ellipse((kurt[position_4], mean[position_4]), elipse_width_4, height_4, angle=0, linewidth=1.5, fill=False, alpha = 1)
ax5.add_artist(ellipse_4)

# plotting arrows
x_arrow_1 = [(min(particle_radii_input[position_4]) - 6*width_4), 0]

x_arrow_2_ax1 = [(kurt[position_1]+ elipse_width_1/2), mean[position_1]]
x_arrow_2_ax2 = [(kurt[position_2]+ elipse_width_2/2), mean[position_2]]
x_arrow_2_ax3 = [(kurt[position_3]+ elipse_width_3/2), mean[position_3]]
x_arrow_2_ax4 = [(kurt[position_4]+ elipse_width_4/2), mean[position_4]]

arrow_1 = patches.ConnectionPatch(
    x_arrow_2_ax1,
    x_arrow_1,
    coordsA=ax5.transData,
    coordsB=ax1a.transData,
    color="black",
    arrowstyle="-|>",  # "normal" arrow
    mutation_scale=10,  # controls arrow head size
    linewidth=1.5,
    alpha = 0.5
)

arrow_2 = patches.ConnectionPatch(
    x_arrow_2_ax2,
    x_arrow_1,
    coordsA=ax5.transData,
    coordsB=ax2a.transData,
    color="black",
    arrowstyle="-|>",  # "normal" arrow
    mutation_scale=10,  # controls arrow head size
    linewidth=1.5,
    alpha = 0.5
)


arrow_3 = patches.ConnectionPatch(
    x_arrow_2_ax3,
    x_arrow_1,
    coordsA=ax5.transData,
    coordsB=ax3a.transData,
    color="black",
    arrowstyle="-|>",  # "normal" arrow
    mutation_scale=10,  # controls arrow head size
    linewidth=1.5,
    alpha = 0.5
)


arrow_4 = patches.ConnectionPatch(
    x_arrow_2_ax4,
    x_arrow_1,
    coordsA=ax5.transData,
    coordsB=ax4a.transData,
    color="black",
    arrowstyle="-|>",  # "normal" arrow
    mutation_scale=10,  # controls arrow head size
    linewidth=1.5,
    alpha = 0.5
)

fig.patches.append(arrow_1)
fig.patches.append(arrow_2)
fig.patches.append(arrow_3)
fig.patches.append(arrow_4)

# text annotation
for l in range(len(particle_radii_input[position_1])):
        ax1.text((particle_radii_input[position_1][l] - width_1/4),(theoretical_dict_values_array[position_1][l] + 200), f"{theoretical_dict_values_array[position_1][l]}", rotation = 90, color = col_lines_ax5[position_1*intopolarion_runs][0].get_color())
for l in range(len(particle_radii_input[position_2])):
        ax2.text((particle_radii_input[position_2][l] - width_2/4),(theoretical_dict_values_array[position_2][l] + 200), f"{theoretical_dict_values_array[position_2][l]}", rotation = 90, color = col_lines_ax5[position_2*intopolarion_runs][0].get_color())
for l in range(len(particle_radii_input[position_3])):
        ax3.text((particle_radii_input[position_3][l] - width_3/4),(theoretical_dict_values_array[position_3][l] + 200), f"{theoretical_dict_values_array[position_3][l]}", rotation = 90, color = col_lines_ax5[position_3*intopolarion_runs][0].get_color())
for l in range(len(particle_radii_input[position_4])):
        ax4.text((particle_radii_input[position_4][l] - width_4/4),(theoretical_dict_values_array[position_4][l] + 200), f"{theoretical_dict_values_array[position_4][l]}", rotation = 90, color = col_lines_ax5[position_4*intopolarion_runs][0].get_color())

# ax5 figure formatting
ax5.grid(which='major', color='k', linestyle='-', alpha = 0.5)
ax5.grid(which='minor', color='black', linestyle='-', alpha = 0.2)
ax5.minorticks_on()
ax5.set_xlabel("Kurtosis of Particle Distibution (-)", fontsize = 11.5)
ax5.set_ylabel("Final Packing Density (-)", fontsize = 11.5)
ax5.tick_params(axis = 'both', labelsize = 11.5)
handles, labels = ax.get_legend_handles_labels()
legend_points = [(p, s)]
legend_labels = ["Fixed Distribution Parameters: \n$\mu_1$ = 0.01m   $\sqrt{\mu_2}$ = 1/330m    $\mu_3$ = 0"]
ax5.legend(legend_points, legend_labels,  loc="upper left", bbox_to_anchor =(0.1, -0.05), ncol = 2, numpoints=2, handler_map={tuple: HandlerTuple(ndivide=None)}, frameon = False, labelspacing=1.9, fontsize = 11.5)
ax5.set_yticklabels(ax5.get_yticks(), rotation=90, va='center')

# subplot bars
ax1.bar(x_data_1, y_data_1, width = width_1, color = col_lines_ax5[position_1*intopolarion_runs][0].get_color())
ax2.bar(x_data_2, y_data_2, width = width_2, color = col_lines_ax5[position_2*intopolarion_runs][0].get_color())
ax3.bar(x_data_3, y_data_3, width = width_3, color = col_lines_ax5[position_3*intopolarion_runs][0].get_color())
ax4.bar(x_data_4, y_data_4, width = width_4, color = col_lines_ax5[position_4*intopolarion_runs][0].get_color())

# subplot lines
ax1.plot(x_data_1, y_data_1, color = col_lines_ax5[position_1*intopolarion_runs][0].get_color())
ax2.plot(x_data_2, y_data_2, color = col_lines_ax5[position_2*intopolarion_runs][0].get_color())
ax3.plot(x_data_3, y_data_3, color = col_lines_ax5[position_3*intopolarion_runs][0].get_color())
ax4.plot(x_data_4, y_data_4, color = col_lines_ax5[position_4*intopolarion_runs][0].get_color())

# subplot y axis colour
ax1.tick_params(axis='y', colors= col_lines_ax5[position_1*intopolarion_runs][0].get_color())
ax2.tick_params(axis='y', colors= col_lines_ax5[position_2*intopolarion_runs][0].get_color())
ax3.tick_params(axis='y', colors= col_lines_ax5[position_3*intopolarion_runs][0].get_color())
ax4.tick_params(axis='y', colors= col_lines_ax5[position_4*intopolarion_runs][0].get_color())

# adding annotations 
ax1.annotate(f"b) Kurtosis = {kurt[position_1]}", xy = [(min(particle_radii_input[position_4]) - width_1), 4500], weight='bold', fontsize = 13)
ax2.annotate(f"c) Kurtosis = {kurt[position_2]}", xy = [(min(particle_radii_input[position_4]) - width_2), 4500], weight='bold', fontsize = 13)
ax3.annotate(f"d) Kurtosis = {kurt[position_3]}", xy = [(min(particle_radii_input[position_4]) - width_3), 4500], weight='bold', fontsize = 13)
ax4.annotate(f"e) Kurtosis = {kurt[position_4]}", xy = [(min(particle_radii_input[position_4]) - width_4), 4500], weight='bold', fontsize = 13)
ax5.annotate("a)", xy = [16.2, 0.63465], weight='bold', fontsize = 13)

# colourbar addition
divider = make_axes_locatable(ax5)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(s, ax=ax5, cax = cax, label = 'Final Packing Density (-)', ticks = [])

#saving figure 
plt.savefig('kurtosis_results_plot', bbox_inches="tight", dpi=600)