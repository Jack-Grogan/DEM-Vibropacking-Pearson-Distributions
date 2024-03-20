# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:23:51 2024

@author: JACK GROGAN

      ___  ________  ________  ___  __            ________     
     |\  \|\   __  \|\   ____\|\  \|\  \         |\   ____\    
     \ \  \ \  \|\  \ \  \___|\ \  \/  /|_       \ \  \___|    
   __ \ \  \ \   __  \ \  \    \ \   ___  \       \ \  \  ___  
  |\  \\_\  \ \  \ \  \ \  \____\ \  \\ \  \       \ \  \|\  \ 
  \ \________\ \__\ \__\ \_______\ \__\\ \__\       \ \_______\
   \|________|\|__|\|__|\|_______|\|__| \|__|        \|_______|
   
"""

import numpy as np
from scipy.special import gamma
from scipy.special import beta
import scipy.stats as stats
import matplotlib.pyplot as plt


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
        
        coeffs = [C0, C1, C2]

    else:
        dist_type = 1
        
        # division by zero
        coeffs = [np.inf, np.inf, np.inf]


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
    return p, dist_type, coeffs


x = np.linspace(-3/330, 3/330, 100)
mu = 0
sigma = 1/330
skew = -0.5
kurt = 3

[f ,dist_type ,coefs]  = pearson_pdf(x,mu,sigma,skew,kurt)
plt.plot(x, f, label = f"Type {dist_type} Pearson Distribution")

plt.grid()
plt.grid(which='minor', alpha=0.2)
plt.minorticks_on()
plt.legend()
plt.show()

