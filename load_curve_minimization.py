import numpy as np
import pandas as pd
import time
import math
from model import getLoadCurve, getCUOREData
from scipy.stats import norm
from scipy.optimize import minimize
import click

def get_power(test_data):
    return np.divide(np.power(test_data.VBol,2),test_data.RBol)

def neg_loglike(circuit_params, VBias, alpha, beta, k, Rl, prior_sigma = 1, verbose=True):

    try:

        start_time = time.time()
        
        (R0, T0) = tuple(circuit_params)

        model = getLoadCurve(VBias,alpha,beta,k,R0,Rl,T0)

        log_like_power = -1*norm(get_power(model),prior_sigma).logpdf(get_power(data)).sum()

        if math.isnan(log_like_power):
            print('Model Failed')
            return 1e10

        time_taken = round((time.time()-start_time)/60.0,2)
            
        global iteration_counter
        iteration_counter += 1
        
        if verbose:
            print("Iteration {0} finished in {1} minutes: {2} with Log Likelihood {3}".format(iteration_counter,time_taken,circuit_params,round(log_like_power,2)))
        
        return log_like_power

    except: 
        print('Model Failed')
        return 1e10

@click.command()
@click.option('--min_method', prompt='Method', help='Minimization Method', type=str)

def main(min_method):
    res = minimize(neg_loglike, [R0,T0], args=(data.VBias,alpha,beta,k,Rl), method = min_method, options={'disp': True})
    print(res.x)

data = getCUOREData(486)

alpha_params = {
    'a1': 2.7e-8,      # phonon 
    'a2': 9.9e-9,      # electron 
    'a3': 1e-11,       # heater 
    'a4': 2.29e-3,     # crystal 
    'a5': 2.11e-6,     # teflon first power
    'a6': 1.93e-4      # teflon third power
}

beta_params = {
    'b1': 3,           # NTD Glue 
    'b2': 4.37,        # EP Coupling
    'b3': 2.4,         # NTD Gold Wire
    'b4': 3,           # Heater Glue
    'b5': 2.4,         # Heater Gold Wire 
    'b6': 2,           # Crystal <-> Teflon Boundary
    'b7': 1            # Teflon <-> Heat Sink Boundary 
}

k_params = {
    'k1': 2.34e-3,     # NTD Glue
    'k2': 0.7,         # EP Coupling
    'k3': 4.8e-5,      # NTD Gold Wire
    'k4': 1.3e-3,      # Heater Glue
    'k5': 3.2e-5,      # Heater Gold Wire
    'k6': 4e-5,        # Crystal <-> Teflon Boundary
    'k7': 1.25e-3      # Teflon <-> Heat Sink Boundary
}

R0 = 1.75
s = .015
Rl = 6e10
T0 = 5.5
gamma = 0.5

alpha = [alpha_params[param] for param in sorted(alpha_params)]

beta = [beta_params[param] for param in sorted(beta_params)]

k = [k_params[param] for param in sorted(k_params)]

#parameters_df = pd.DataFrame(columns=['log_like_power','R0','T0'])

iteration_counter = 0

if __name__ == "__main__":
    print("\nCUORE Thermal Modeling Minimization\n")
    main()