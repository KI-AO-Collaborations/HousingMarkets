import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd
from statsmodels.iolib.summary2 import summary_col
from scipy.stats import norm
import scipy.optimize as opt
import scipy.integrate as integrate
import math
from math import e, pi, log
from stargazer.stargazer import Stargazer

#Set current directory
path = '/Volumes/GoogleDrive/My Drive/4th Year/Quarter 1/Topics in Microeconometrics/ECON21130Project'
os.chdir(path)

def pdf(x, mu, sigma_2):
    pdf_val = 1 / math.sqrt(2 * pi * sigma_2) * e ** ( - (x - mu) ** 2 / (2 * sigma_2) )
    return pdf_val

def integ(x, reservation_price, mu, sigma_2):
    pdf_val = pdf(x, mu, sigma_2)
    return (x - reservation_price) * pdf_val

def crit_reservation_price(params, *args):
    reservation_price, = params
    #print(reservation_price)
    mu, sigma_2, beta, integ = args
    lhs = (1 - beta) * reservation_price
    rhs = beta * integrate.quad(integ, reservation_price, np.inf, args=(reservation_price, mu, sigma_2))[0]
    # print('lhs', lhs)
    # print('rhs', rhs)
    # if np.isnan(lhs):
    #     print('lhs nan')
    # if np.isnan(rhs):
    #     print('rhs nan')
    crit_val = np.sum( (lhs - rhs) ** 2)
    return crit_val

def get_res_price(data_buy, N):
    # data_buy is the buying offer price
    sigma_2 = 400
    beta = 0.98

    res_prices = []
    for i in range(N):
        res_price_0 = data_buy[i]
        params_0 = res_price_0
        args = (data_buy[i], sigma_2, beta, integ)

        results_b = opt.minimize(crit_reservation_price, params_0, args=args, tol=1e-200,
                        method='L-BFGS-B')
        reservation_price, = results_b.x
        res_prices.append(reservation_price)
        # print('Reservation price:', reservation_price)
        # diff = (1 - beta) * reservation_price -\
        #     beta * integrate.quad(integ, reservation_price, np.inf, args=(reservation_price, data_buy[i], sigma_2))[0]
        # print('Difference between lhs and rhs:', diff)

    return np.array(res_prices)

def probit(Y, X):
    # run probit regression
    model = smd.Probit(Y, X).fit()
    print(model.params)
    baths_coeff, beds_coeff, other_rms_coeff, house_age_coeff, res_price_coeff = model.params
    print(model.summary()) # summary statistics
    return model, baths_coeff, beds_coeff, other_rms_coeff, house_age_coeff, res_price_coeff

N = 100000
data_x = pd.DataFrame()
data_x['baths'] = np.random.randint(1, high=5, size=N)
data_x['beds'] = np.random.randint(1, high=5, size=N)
data_x['other_rms'] = np.random.randint(1, high=5, size=N)
data_x['house_age'] = 100 * np.random.random(size=N)
#data_x['total_rms'] = data_x['baths'] + data_x['beds'] + data_x['other_rms']

beta = np.array([5, 5, 5, 0.25]) #np.random.normal(size=5) ** 2
#eps_sell = np.random.normal(size=N, scale=20)
#data_sell = data_x @ beta + eps_sell
eps_buy = np.random.normal(size=N, scale=20)
data_buy = data_x @ beta + eps_buy

res_prices = get_res_price(data_buy - eps_buy, N)

eps_res = np.random.normal(size=N, scale=10)
res_prices_true = res_prices + eps_res

visible_data = data_buy > res_prices_true
d = data_buy.copy()
d[visible_data] = 1
d[~visible_data] = 0

data_x['res_price'] = res_prices

#X = sm.add_constant(data_x)
X = data_x

probit, baths_coeff, beds_coeff, other_rms_coeff, house_age_coeff, res_price_coeff = probit(d, X)
print(baths_coeff, beds_coeff, other_rms_coeff, house_age_coeff, res_price_coeff)

#probit, const_coeff, baths_coeff, beds_coeff, other_rms_coeff, house_age_coeff, res_price_coeff = probit(d, X)
#print(const_coeff, baths_coeff, beds_coeff, other_rms_coeff, house_age_coeff, res_price_coeff)
