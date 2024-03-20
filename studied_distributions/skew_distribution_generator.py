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

#%% imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.special import beta
import scipy.stats as stats
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors


#%% Total particle volume check

def Distribution_Sphere_Volume(r, p, N):
    if len(r) != len(p):
        return print('len(x) != len(p)')
    else: 
        p_frac = p/sum(p)
        volume = [0]*len(r)
        for i in range(len(r)):
            volume[i] = p_frac[i]*4/3*np.pi*(r[i])**3*N
        volume_total = sum(volume)
        return volume_total
            
#%% Total particle volume calculator

def Distribution_Set_Volume_N(r, p, set_volume):
    if len(r) != len(p):
        return print('len(x) != len(p)')
    else: 
        p_frac = p/sum(p)
        volume = [0]*len(r)
        for i in range(len(r)):
            volume[i] = p_frac[i]*4/3*np.pi*(r[i])**3
            
        volume_total = sum(volume)
        N = round(set_volume/volume_total)       
            
        return N

#%%
def Distribution_Discritiser(N_particles, height_data):
    particle_split = [0]*len(height_data)
    for i in range(len(height_data)):
        particle_split[i] = round(height_data[i]*N_particles)
    return particle_split

            
    
#%%
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

        
def constant_V_distribution(r, p, mu, sigma, N_particles):
    
    p_norm, _ =  pearson_pdf(r, mu, sigma, 0, 3)
    norm_V = Distribution_Sphere_Volume(r, p_norm, N_particles)
    
    const_vol_N = [0]*len(p)
    N_particle_frac_raw = [0]*len(p)

    for i in range(len(p)):
        N_particle_frac_raw[i] = p[i]/sum(p[i])
        const_vol_N[i] = Distribution_Set_Volume_N(r, N_particle_frac_raw[i], norm_V)
            
    p_V = [0]*len(const_vol_N)

    for i in range(len(const_vol_N)):
        p_V[i] = Distribution_Discritiser(const_vol_N[i], N_particle_frac_raw[i])
        p_V[i] = np.array(p_V[i])

    return p_V, const_vol_N, norm_V
        


#%% 

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

cmap = colors.LinearSegmentedColormap.from_list("",['#260c51', '#360961', '#440a68', '#520e6d', '#61136e', '#6f196e', '#7d1e6d', '#8c2369', '#982766', '#a62d60', '#b43359', '#c13a50', '#ce4347', '#d94d3d', '#e25734', '#eb6429', '#f1731d','#f78212', '#fa9207','#fca309','#fcb216' ], N_curves)

#sigma = 1/330 #m
# skew^2 must be less than kurt - 1
sigma = 1/330 #m
skew = np.concatenate((np.linspace(-0.85, -0.15, int((N_curves-1)/2)),[0], np.linspace(0.15, 0.85, int((N_curves-1)/2))))
kurt = 3

x_range = 3*1/330 # This ensures that changes in sigma do not result in too small particle sizes

x_bar = np.linspace(mu-x_range, mu+x_range, n_bars)
x_cont = np.linspace(mu-x_range, mu+x_range, 200)

#------------------------------------------------------------------------------------------
# Use radii data for input  
#------------------------------------------------------------------------------------------

x_bar_radii = x_bar/2
x_cont_radii = x_cont/2
mu_radii = mu/2
sigma_radii = sigma/2

#------------------------------------------------------------------------------------------
# Plotting Distribution for one of skew and kurt as array  
#------------------------------------------------------------------------------------------

p_bar, dist_type_bar =  pearson_array_intorfloat(x_bar_radii, mu_radii, sigma_radii, skew, kurt)
p_cont, dist_type_cont =  pearson_array_intorfloat(x_cont_radii, mu_radii, sigma_radii, skew, kurt)

#------------------------------------------------------------------------------------------
# Generating Normal Distribution for standardising particle volume
#------------------------------------------------------------------------------------------

norm_cont, norm_type_cont =  pearson_pdf(x_cont_radii, mu_radii, 1/330, 0, 3)
norm_bar, norm_type_bar =  pearson_pdf(x_bar_radii, mu_radii, 1/330, 0, 3)

#------------------------------------------------------------------------------------------
#  Distribution of particle fractions for constant number of particles
#------------------------------------------------------------------------------------------

N_particle_split_bar = constant_N_distribution(p_bar, N_particles)
N_norm_particle_split = constant_N_distribution(norm_bar, N_particles)

#------------------------------------------------------------------------------------------
#  Distribution of particle fractions for constant volume of equivelent normal distibution
#------------------------------------------------------------------------------------------

V_particle_split_bar, const_vol_N, volume = constant_V_distribution(x_bar_radii, p_bar, mu_radii, sigma_radii, N_particles)

#------------------------------------------------------------------------------------------
# Plotting distributions
#------------------------------------------------------------------------------------------
fig, (ax1, ax3) = plt.subplots(2, 1, figsize = (10, 6), dpi = 600)


for i in range(len(p_bar)):
    ax1.scatter(x_bar_radii, p_bar[i], color = cmap(i), marker = 'x')

for i in range(len(p_cont)):
    ax1.plot(x_cont_radii, p_cont[i], color = cmap(i))

ax1.grid()
ax1.grid(which='minor', alpha=0.2)
ax1.minorticks_on()
ax1.annotate("a) Pearson Probability Density Function", xy = [(min(x_bar_radii)-(x_bar_radii[-1] - x_bar_radii[-2])/2), 616], weight='bold', fontsize = 13)


for i in range(len(V_particle_split_bar)):
    ax3.plot(x_bar_radii, V_particle_split_bar[i], color = cmap(i))
    
ax3.annotate("b) Constant Volume Distributon", xy = [(min(x_bar_radii)-(x_bar_radii[-1] - x_bar_radii[-2])/2), 4840], weight='bold', fontsize = 13)
ax3.grid()
ax3.grid(which='minor', alpha=0.2)
ax3.minorticks_on()

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')

norm = colors.Normalize(vmin=(min(skew) - (skew[-1] - skew[-2])/2), vmax=(max(skew) + (skew[-1] - skew[-2])/2)) 
  
cbar_ax = fig.add_axes([0.92, 0.13, 0.03, 0.75])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
sm.set_array([]) 
cb = fig.colorbar(sm, cax =cbar_ax, orientation = 'vertical', ticks=np.linspace(min(skew),max(skew), N_curves))
cb.set_ticklabels(f"{skew_in:.2f}" for skew_in in skew)
cb.set_label('Skew of Distribution', fontsize = 13)
ax1.set_ylim(0, 700)
ax1.set_ylabel("Probability Density (-)")

ax3.set_ylim(0, 5500)
ax3.set_ylabel("Number of Particles (-)")

ax1.yaxis.set_major_locator(MultipleLocator(50))
ax3.yaxis.set_major_locator(MultipleLocator(500))

ax1.set_xticks(x_bar_radii, labels =[])
ax3.set_xticks(x_bar_radii, labels = [f"{x_bar_radii_in:.6f}" for x_bar_radii_in in x_bar_radii] , rotation = 70)
ax3.set_xlabel("Particle Radii (m)", fontsize = 11.5)

plt.savefig('generated_skew_distribution', bbox_inches="tight", dpi=600)