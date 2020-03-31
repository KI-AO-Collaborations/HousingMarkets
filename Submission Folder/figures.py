import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import scipy.optimize as opt
import scipy.integrate as sci_integrate

beta = np.array([2, 1.75, 3])
sigma_L = 1
alpha = 1
c = 0
gamma = 1
epsilon = 1
eta = 10

def V(p_L, X):
    return p_L - X @ beta


def lambda_fn(p_L, X, alpha, gamma, epsilon, c):
    return ( alpha / np.exp( V(p_L, X) ) - c ) ** ( 1 / (1 + gamma) ) 


def integrand_price(d, p_L, X, alpha, gamma, epsilon, c):
    lambda_ = lambda_fn(p_L, X, alpha, gamma, epsilon, c)
    return d ** (1 / (1 + epsilon) ) * lambda_ * np.exp( - lambda_ * d )


def price_int(p_L, X, alpha, gamma, eta, epsilon, c):
    val = sci_integrate.quad(integrand_price, 0, np.inf, args=(p_L, X, alpha, gamma, epsilon, c))[0]
    price = p_L - eta * val
    return price


# Comparative Statistics
X_matrix_s = []
X_matrix_m = []
X_matrix_l = []

for s in range(1, 4):
    if s == 1:
        for ba in range(1,3):
            for be in range(1,3):
                X_matrix_s.append([ba,be,s])
                
    if s == 2:
        for ba in range(2,4):
            for be in range(2,4):
                X_matrix_m.append([ba,be,s])
                
    if s == 3:
        for ba in range(3,5):
            for be in range(3,5):
                X_matrix_l.append([ba,be,s])


fig = plt.figure(figsize=(8,6))
colors = ['blue', 'orange', 'green', 'red']
for i in range(len(X_matrix_s)):
    X = X_matrix_s[i]
    p_L = np.linspace(1, 10, 100)
    prices = []
    for p in p_L:
        prices.append(price_int(p, X, alpha, gamma, eta, epsilon, c))
    plt.plot(p_L, 10 + np.array(prices), color=colors[i], label=r'X={}'.format(X))
    
    max_price = p_L[np.argmax(prices)]
    plt.axvline(x=max_price, color=colors[i], linestyle='--')
    
    plt.xlabel('Listing Price ($p^L$)')
    plt.ylabel('Selling Price ($p$)')
    plt.yticks([])
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('Output/X_s.png')
    plt.close()


fig = plt.figure(figsize=(8,6))
colors = ['blue', 'orange', 'green', 'red']
for i in range(len(X_matrix_m)):
    X = X_matrix_m[i]
    p_L = np.linspace(1, 16, 100)
    prices = []
    for p in p_L:
        prices.append(price_int(p, X, alpha, gamma, eta, epsilon, c))
    plt.plot(p_L, 10 * np.array(prices), color=colors[i], label=r'X={}'.format(X))
    
    max_price = p_L[np.argmax(prices)]
    plt.axvline(x=max_price, color=colors[i], linestyle='--')
    
    plt.xlabel('Listing Price ($p^L$)')
    plt.ylabel('Selling Price ($p$)')
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('Output/X_m.png')
    plt.close()


fig = plt.figure(figsize=(8,6))
colors = ['blue', 'orange', 'green', 'red']
for i in range(len(X_matrix_l)):
    X = X_matrix_l[i]
    p_L = np.linspace(1, 24, 100)
    prices = []
    for p in p_L:
        prices.append(price_int(p, X, alpha, gamma, eta, epsilon, c))
    plt.plot(p_L, 10 * np.array(prices), color=colors[i], label=r'X={}'.format(X))
    
    max_price = p_L[np.argmax(prices)]
    plt.axvline(x=max_price, color=colors[i], linestyle='--')
    
    plt.xlabel('Listing Price ($p^L$)')
    plt.ylabel('Selling Price ($p$)')
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('Output/X_l.png')
    plt.close()


fig = plt.figure(figsize=(8,6))
X = [4, 3, 3]
alphas = np.linspace(0.5,1.5,5)
colors = ['blue', 'orange', 'green', 'red', 'purple']
for i in range(len(alphas)):
    alpha = alphas[i]
    p_L = np.linspace(1, 24, 100)
    prices = []
    for p in p_L:
        prices.append(price_int(p, X, alpha, gamma, eta, epsilon, c))
    plt.plot(p_L, 10 * np.array(prices), label=r'$\alpha$={}'.format(alpha))
    
    max_price = p_L[np.argmax(prices)]
    plt.axvline(x=max_price, color=colors[i], linestyle='--')
    
    plt.xlabel('Listing Price ($p^L$)')
    plt.ylabel('Selling Price ($p$)')
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('Output/alpha.png')
    plt.close()


fig = plt.figure(figsize=(8,6))
X = [4, 3, 3]
gammas = np.linspace(0.5,1.5,5)
colors = ['blue', 'orange', 'green', 'red', 'purple']
for i in range(len(gammas)):
    gamma = gammas[i]
    p_L = np.linspace(1, 24, 100)
    prices = []
    for p in p_L:
        prices.append(price_int(p, X, alpha, gamma, eta, epsilon, c))
    plt.plot(p_L, 10 * np.array(prices), color=colors[i], label=r'$\gamma$={}'.format(gamma))
    
    max_price = p_L[np.argmax(prices)]
    plt.axvline(x=max_price, color=colors[i], linestyle='--')    
    
    plt.xlabel('Listing Price ($p^L$)')
    plt.ylabel('Selling Price ($p$)')
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('Output/gamma.png')
    plt.close()


fig = plt.figure(figsize=(8,6))
X = [4, 3, 3]
etas = np.linspace(7, 12, 6)
colors = ['blue', 'orange', 'green', 'red', 'purple', 'pink']
for i in range(len(etas)):
    eta = etas[i]
    p_L = np.linspace(1, 24, 100)
    prices = []
    for p in p_L:
        prices.append(price_int(p, X, alpha, gamma, eta, epsilon, c))
    plt.plot(p_L, 10 * np.array(prices), color=colors[i], label=r'$\eta$={}'.format(eta))
    
    max_price = p_L[np.argmax(prices)]
    plt.axvline(x=max_price, color=colors[i], linestyle='--')  
    
    plt.xlabel('Listing Price ($p^L$)')
    plt.ylabel('Selling Price ($p$)')
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('Output/eta.png')
    plt.close()


fig = plt.figure(figsize=(8,6))
X = [4, 3, 3]
epsilons = np.linspace(0.5,1.5,5)
colors = ['blue', 'orange', 'green', 'red', 'purple']
for i in range(len(epsilons)):
    epsilon = epsilons[i]
    p_L = np.linspace(1, 24, 100)
    prices = []
    for p in p_L:
        prices.append(price_int(p, X, alpha, gamma, eta, epsilon, c))
    plt.plot(p_L, 10 * np.array(prices), color=colors[i], label=r'$\epsilon$={}'.format(epsilon))
    
    max_price = p_L[np.argmax(prices)]
    plt.axvline(x=max_price, color=colors[i], linestyle='--')     
    
    plt.xlabel('Listing Price ($p^L$)')
    plt.ylabel('Selling Price ($p$)')
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('Output/epsilon.png')
    plt.close()


fig = plt.figure(figsize=(8,6))
X = [4, 3, 3]
for c in np.linspace(1, 5, 5):
    p_L = np.linspace(1, 24, 100)
    prices = []
    for p in p_L:
        prices.append(price_int(p, X, alpha, gamma, eta, epsilon, c))
    plt.plot(p_L, 10 * np.array(prices), label=r'$c$={}'.format(c))
    plt.xlabel('Listing Price ($p^L$)')
    plt.ylabel('Selling Price ($p$)')
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('Output/c.png')
    plt.close()



