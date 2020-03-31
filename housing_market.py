import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.optimize as opt
import scipy.integrate as sci_integrate
from scipy.stats import norm, expon

#############################
##### Generate the Data #####
#############################

##################
##### Houses #####
##################

N = 20
num_size = 3

data = pd.DataFrame()
data['size'] = np.random.randint(low=1, high=1 + num_size, size=N)

def gen_room(data, num_size):
    # Generate room data
    data_new = np.zeros(data.shape[0])

    for i in range(1, num_size + 1):
        relevant_data = data['size'] == i
        len_ = len(data[relevant_data])
        data_new[relevant_data] \
            = np.random.randint(low=i, high=i + 2, size=len_)
    return data_new

data['beds'] = gen_room(data, num_size)
data['baths'] = gen_room(data, num_size)

##############################
##### Price and Duration #####
##############################
beta = np.array([2, 1.75, 3])
sigma_L = 1
alpha = 1
c = 0
gamma = 1
epsilon = 1
eta = 10

data['x_i_beta'] = data @ beta
data['p_L'] = np.random.normal(loc=data['x_i_beta'], scale=sigma_L)
data['value_gap'] = data['p_L'] - data['x_i_beta']
data['lambda'] = (alpha / np.exp(data['value_gap']) - c) ** (1 / (1 + gamma) )
data['duration'] = np.random.exponential(scale = 1 / data['lambda'])
data['p'] = data['p_L'] - eta * data['duration'] ** (1 / (1 + epsilon) )

###############
##### MLE #####
###############
def integrand_MLE(p_L, house_rooms, duration, params):
    beta_0, beta_1, beta_2, sigma_L, alpha, \
                c, gamma, epsilon, eta = params
    beta = np.array([beta_0, beta_1, beta_2])
    value_gap = p_L - house_rooms @ beta
    lambda_ = (alpha / np.exp(value_gap) - c) ** (1 / (1 + gamma) )
    #if abs(np.exp(value_gap)) == np.inf:
    #    print('Issue is with value_gap')
    #    return 0
    if abs(np.exp( - 1 / 2 * (value_gap / sigma_L) ** 2) ) == np.inf:
        print('Issue is with value_gap squared')
        return 0
    if abs(np.exp( - lambda_ * duration) ) == np.inf:
        print('Issue is with lambda')
        return 0
    val = lambda_ * np.exp( - lambda_ * duration) \
            * np.exp( - 1 / 2 * (value_gap / sigma_L) ** 2)

    if abs(val) == np.inf:
        print('Issue is with val')
        return 0

    return val

def integrate_MLE(data, params):
    duration = data['duration']
    house_rooms = np.array([data['size'], data['beds'], \
                            data['baths']])
    val = sci_integrate.quad(integrand_MLE, - np.inf, np.inf,\
                            args=(house_rooms, duration, params))[0]
    return val

def crit(params, *args):
    beta_0, beta_1, beta_2, sigma_L, alpha, \
                c, gamma, epsilon, eta = params
    beta = np.array([beta_0, beta_1, beta_2])
    print('--------------------------------------')
    print('beta guess:', beta)
    house_rooms, p, duration = args[0]

    first_term = - 2 * np.log(sigma_L * np.sqrt(2 * np.pi) ) \
                    - 1 / 2 * ( (p - house_rooms @ beta \
                    + eta * duration ** (1 / (1 + epsilon) ) ) \
                    / sigma_L) ** 2

    second_term_data = pd.concat([house_rooms, duration], axis=1)
    second_term = second_term_data.apply(integrate_MLE, axis=1, params=params)
    second_term = np.log(second_term)

    log_lik = np.sum(first_term) + np.sum(second_term)
    print('log lik:', log_lik)

    return - log_lik


def run_MLE():
    beta_0_0 = 2
    beta_0_1 = 1.75
    beta_0_2 = 3
    sigma_L_0 = 1
    alpha_0 = 1
    c_0 = 0
    gamma_0 = 1
    epsilon_0 = 1
    eta_0 = 10

    params_init = np.array([beta_0_0, beta_0_1, beta_0_2, sigma_L_0, alpha_0, \
                    c_0, gamma_0, epsilon_0, eta_0])

    args = np.array([data[['size', 'beds', 'baths']], data['p'], data['duration']])

    results = opt.minimize(crit, params_init, args=args, method='L-BFGS-B')
    beta_0_MLE, beta_1_MLE, beta_2_MLE, sigma_L_MLE, alpha_MLE, c_MLE, \
        gamma_MLE, epsilon_MLE, eta_MLE = results.x

    beta_MLE = np.array([beta_0_MLE, beta_1_MLE, beta_2_MLE])

    print('beta_MLE:', beta_MLE)
    print('sigma_L_MLE:', sigma_L_MLE)
    print('alpha_MLE:', alpha_MLE)
    print('c_MLE:', c_MLE)
    print('gamma_MLE:', gamma_MLE)
    print('epsilon_MLE:', epsilon_MLE)
    print('eta_MLE:', eta_MLE)

    # calculate standard errors using inverse Hessian

    # Invese Hessian Matrix from optimizer function
    vcv_mle = (results.hess_inv).matmat(np.eye(9))
    print('VCV_mle =', vcv_mle)
    stderr_beta_0_MLE = np.sqrt(vcv_mle[0,0])
    stderr_beta_1_MLE = np.sqrt(vcv_mle[1,1])
    stderr_beta_2_MLE = np.sqrt(vcv_mle[2,2])
    stderr_beta_MLE = np.array((stderr_beta_0_MLE, \
                        stderr_beta_1_MLE, stderr_beta_2_MLE))
    stderr_sigma_L_MLE = np.sqrt(vcv_mle[3,3])
    stderr_alpha_MLE = np.sqrt(vcv_mle[4,4])
    stderr_c_MLE = np.sqrt(vcv_mle[5,5])
    stderr_gamma_MLE = np.sqrt(vcv_mle[6,6])
    stderr_epsilon_MLE = np.sqrt(vcv_mle[7,7])
    stderr_eta_MLE = np.sqrt(vcv_mle[8,8])
    print('Standard error for beta estimate =', stderr_beta_MLE)
    print('Standard error for sigma_L estimate =', stderr_sigma_L_MLE)
    print('Standard error for alpha estimate =', stderr_alpha_MLE)
    print('Standard error for c estimate =', stderr_c_MLE)
    print('Standard error for gamma estimate =', stderr_gamma_MLE)
    print('Standard error for epsilon estimate =', stderr_epsilon_MLE)
    print('Standard error for eta estimate =', stderr_eta_MLE)

##############
##### EM #####
##############
def integrand_p_L(p_L, house_rooms, duration, params):
    beta_0, beta_1, beta_2, sigma_L, alpha, \
                c, gamma, epsilon, eta = params
    beta = np.array([beta_0, beta_1, beta_2])
    value_gap = p_L - house_rooms @ beta
    lambda_ = (alpha / np.exp(value_gap) - c) ** (1 / (1 + gamma) )

    val_1 = expon.pdf(duration, scale=1 / lambda_)
    val_2 = norm.pdf(p_L, loc=house_rooms @ beta, scale=sigma_L)

    return val_1 * val_2

def integrate_EM(data, params):
    duration = data['duration']
    house_rooms = np.array([data['size'], data['beds'], \
                            data['baths']])
    val = sci_integrate.quad(integrand_p_L, - 30, 30,\
                            args=(house_rooms, duration, params))[0]
    return val

def posterior_p_L(p_L, house_rooms, duration, p, params):
    beta_0, beta_1, beta_2, sigma_L, alpha, \
                c, gamma, epsilon, eta = params
    beta = np.array([beta_0, beta_1, beta_2])
    value_gap = p_L - house_rooms @ beta
    lambda_ = (alpha / np.exp(value_gap) - c) ** (1 / (1 + gamma) )

    numerator_1 = p == p_L - eta * duration
    # Below is hypothetical solution to issue of above line being
    # measure zero:
    #numerator_1 = norm.pdf(p - (p_L - eta * duration \
    #                ** (1 / (1 + epsilon) ) ) ) / norm.pdf(0)
    numerator_2 = expon.pdf(duration, scale=1 / lambda_)
    numerator_3 = norm.pdf(p_L, loc=house_rooms @ beta, scale=sigma_L)

    denominator_1 = norm.pdf(p, loc=house_rooms @ beta - eta \
                    * duration ** (1 / (1 + epsilon) ), scale=sigma_L)
    denominator_2_data = pd.concat([house_rooms, duration], axis=1)
    denominator_2 = denominator_2_data.apply(integrate_EM, axis=1, params=params)
    
    numerator = numerator_1 * numerator_2 * numerator_3
    denominator = denominator_1 * denominator_2
    #print('Numerator', numerator)
    #print('Denominator', denominator)
    #stop

    return numerator / denominator

def integrand_EM(p_L, house_rooms, duration, p, params, params_old):
    beta_0, beta_1, beta_2, sigma_L, alpha, \
                c, gamma, epsilon, eta = params
    beta = np.array([beta_0, beta_1, beta_2])
    value_gap = p_L - house_rooms @ beta
    lambda_ = (alpha / np.exp(value_gap) - c) ** (1 / (1 + gamma) )
    #if abs(np.exp(value_gap)) == np.inf:
    #    print('Issue is with value_gap')
    #    return 0
    #if abs(np.exp( - 1 / 2 * (value_gap / sigma_L) ** 2) ) == np.inf:
    #    print('Issue is with value_gap squared')
    #    return 0
    #if abs(np.exp( - lambda_ * duration) ) == np.inf:
    #    print('Issue is with lambda')
    #    return 0
    val_1 = posterior_p_L(p_L, house_rooms, duration, p, params_old)

    log_val_1 = p == p_L - eta * duration
    ## Set any values where p != p_L - eta * duration to have
    ## 1e-10 probability, so don't encounter log(0) errors
    #log_val_1[log_val_1 == 0] = 1e-10
    # Below is hypothetical solution to issue of above line being
    # measure zero:
    #log_val_1 = norm.pdf(p - (p_L - eta * duration \
    #                        ** (1 / (1 + epsilon) )) ) / norm.pdf(0)
    log_val_2 = expon.pdf(duration, scale=1 / lambda_)
    log_val_3 = norm.pdf(p_L, loc=house_rooms @ beta, scale=sigma_L)
    val_2 = np.log(log_val_1) \
                + np.log(log_val_2) + np.log(log_val_3)

    val = np.sum(val_1 * val_2)

    if abs(val) == np.inf:
        print('Issue is with val')
        return 0

    return val

def calc_Q_EM(params, *args):
    house_rooms, duration, p, params_old = args
    val = sci_integrate.quad(integrand_EM, - 30, 30,\
                            args=(house_rooms, duration, p, params, params_old))[0]
    print('Params_old:', params_old)
    print('Params_new:', params)
    print('Q:', val)
    return - val

def run_EM():
    beta_0_0 = 2
    beta_0_1 = 1.75
    beta_0_2 = 3
    sigma_L_0 = 1
    alpha_0 = 1
    c_0 = 0
    gamma_0 = 1
    epsilon_0 = 1
    eta_0 = 10

    params_old = np.array([beta_0_0, beta_0_1, beta_0_2, sigma_L_0, alpha_0, \
                    c_0, gamma_0, epsilon_0, eta_0])
    params_new = np.zeros(params_old.shape)

    thresh = 1e-5

    i = 0
    while True:
        print('Loop', i)
        i += 1
        args = (data[['size', 'beds', 'baths']], data['duration'], data['p'], params_old)

        results = opt.minimize(calc_Q_EM, params_old.copy(), \
                    args=args, method='L-BFGS-B')
        params_new = results.x

        diff_params = np.sum( (params_old - params_new) ** 2)
        
        params_old = params_new.copy()

        if diff_params < thresh:
            print('params_new:', params_new)
            break

run_EM()
