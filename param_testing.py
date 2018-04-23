import os
import numpy as np
import pandas as pd
import time
import multiprocessing as mp
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from decimal import Decimal

from model import getLoadCurve

base_dir = 'load_curve_data'
channel = 486
run = 301301

path = os.path.join(*[base_dir,'run{0}'.format(run),'LoadCurveData_run{0}_chan{1}.txt'.format(run,channel)])

def skip_to(fle, line,**kwargs):
    if os.stat(fle).st_size == 0:
        raise ValueError("File is empty")
    with open(fle) as f:
        pos = 0
        cur_line = f.readline()
        while not cur_line.startswith(line):
            pos = f.tell()
            cur_line = f.readline()
        f.seek(pos)
        return pd.read_csv(f, **kwargs)

data = skip_to(path, 'Vbias', delim_whitespace=True)
data = data.iloc[:,:4]
data.columns=['VBias','VBol','IBol','RBol']
data = data[data.VBias>0] #only keep positive bias voltages

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
Cp = 5e-10

alpha = [alpha_params[param] for param in sorted(alpha_params)]

beta = [beta_params[param] for param in sorted(beta_params)]

k = [k_params[param] for param in sorted(k_params)]

ratio = 0.05


for index, parameter in enumerate(alpha):

    param_name = list(sorted(alpha_params))[index]
    print("Test Param {0}".format(param_name))

    start_time = time.time()

    fig_lc, ax_lc = plt.subplots()
    fig_pulse, ax_pulse = plt.subplots()

    for new_parameter in [parameter*(1-3*ratio),parameter*(1-2*ratio), parameter*(1-ratio),parameter,parameter*(1+ratio),parameter*(1+2*ratio),parameter*(1+3*ratio)]:
        alpha_test = alpha[:]
        alpha_test[index] = new_parameter
        print("New Value: {0}".format(alpha_test[index]))
        model, pulse = getLoadCurve(data.VBias,alpha_test,beta,k,R0,Rl,T0,Cp,s=0.015,gamma=0.5)
        print(model)
        print('\n')

        IBol_smooth = np.linspace(model.IBol.min(),model.IBol.max(),300)
        VBol_smooth = spline(model.IBol,model.VBol,IBol_smooth)
        ax_lc.plot(IBol_smooth,VBol_smooth,label='%.2E' % Decimal(str(new_parameter)))
        ax_lc.scatter(model.IBol, model.VBol,c='black',label='_nolegend_')

        t = np.linspace(0, 5,len(pulse))
        ax_pulse.plot(t,pulse,label='%.2E' % Decimal(str(new_parameter)))

    ax_lc.legend(loc='lower right', title=param_name)
    ax_lc.savefig("plot_output/loadcurve_{0}.png".format(param_name), bbox_inches='tight')

    ax_pulse.legend(loc='upper right', title=param_name)
    ax_pulse.savefig("plot_output/pulse_{0}.png".format(param_name), bbox_inches='tight')

    time_taken = round((time.time()-start_time)/60.0,2)
    print('Done with parameter {0} in {1} mins'.format(param_name,time_taken))


for index, parameter in enumerate(beta):

    param_name = list(sorted(beta_params))[index]
    print("Test Param {0}".format(param_name))

    start_time = time.time()

    fig_lc, ax_lc = plt.subplots()
    fig_pulse, ax_pulse = plt.subplots()

    for new_parameter in [parameter*(1-3*ratio),parameter*(1-2*ratio), parameter*(1-ratio),parameter,parameter*(1+ratio),parameter*(1+2*ratio),parameter*(1+3*ratio)]:
        beta_test = beta[:]
        beta_test[index] = new_parameter
        print("New Value: {0}".format(beta_test[index]))
        model, pulse = getLoadCurve(data.VBias,alpha,beta_test,k,R0,Rl,T0,Cp,s=0.015,gamma=0.5)
        print(model)
        print('\n')

        IBol_smooth = np.linspace(model.IBol.min(),model.IBol.max(),300)
        VBol_smooth = spline(model.IBol,model.VBol,IBol_smooth)
        ax_lc.plot(IBol_smooth,VBol_smooth,label='%.2E' % Decimal(str(new_parameter)))
        ax_lc.scatter(model.IBol, model.VBol,c='black',label='_nolegend_')

        t = np.linspace(0, 5,len(pulse))
        ax_pulse.plot(t,pulse,label='%.2E' % Decimal(str(new_parameter)))

    ax_lc.legend(loc='lower right', title=param_name)
    ax_lc.savefig("plot_output/loadcurve_{0}.png".format(param_name), bbox_inches='tight')

    ax_pulse.legend(loc='upper right', title=param_name)
    ax_pulse.savefig("plot_output/pulse_{0}.png".format(param_name), bbox_inches='tight')

for index, parameter in enumerate(k):

    param_name = list(sorted(k_params))[index]
    print("Test Param {0}".format(param_name))

    start_time = time.time()

    fig_lc, ax_lc = plt.subplots()
    fig_pulse, ax_pulse = plt.subplots()

    for new_parameter in [parameter*(1-3*ratio),parameter*(1-2*ratio), parameter*(1-ratio),parameter,parameter*(1+ratio),parameter*(1+2*ratio),parameter*(1+3*ratio)]:
        k_test = k[:]
        k_test[index] = new_parameter
        print("New Value: {0}".format(k_test[index]))
        model, pulse = getLoadCurve(data.VBias,alpha,beta,k_test,R0,Rl,T0,Cp,s=0.015,gamma=0.5)
        print(model)
        print('\n')

        IBol_smooth = np.linspace(model.IBol.min(),model.IBol.max(),300)
        VBol_smooth = spline(model.IBol,model.VBol,IBol_smooth)
        ax_lc.plot(IBol_smooth,VBol_smooth,label='%.2E' % Decimal(str(new_parameter)))
        ax_lc.scatter(model.IBol, model.VBol,c='black',label='_nolegend_')

        t = np.linspace(0, 5,len(pulse))
        ax_pulse.plot(t,pulse,label='%.2E' % Decimal(str(new_parameter)))

    ax_lc.legend(loc='lower right', title=param_name)
    ax_lc.savefig("plot_output/loadcurve_{0}.png".format(param_name), bbox_inches='tight')

    ax_pulse.legend(loc='upper right', title=param_name)
    ax_pulse.savefig("plot_output/pulse_{0}.png".format(param_name), bbox_inches='tight')

for index, parameter in enumerate([R0,T0,Rl,Cp]):

    param_name = list(['R0','T0','Rl','Cp'])[index]
    print("Test Param {0}".format(param_name))

    start_time = time.time()

    fig_lc, ax_lc = plt.subplots()
    fig_pulse, ax_pulse = plt.subplots()

    for new_parameter in [parameter*(1-3*ratio),parameter*(1-2*ratio), parameter*(1-ratio),parameter,parameter*(1+ratio),parameter*(1+2*ratio),parameter*(1+3*ratio)]:
        circuit_test = [R0,T0,Rl,Cp][:]
        circuit_test[index] = new_parameter
        print("New Value: {0}".format(circuit_test[index]))
        R0_n,T0_n,Rl_n,Cp_n = tuple(circuit_test)
        model, pulse = getLoadCurve(data.VBias,alpha,beta,k,R0_n,Rl_n,T0_n,Cp_n,s=0.015,gamma=0.5)
        print(model)
        print('\n')

        IBol_smooth = np.linspace(model.IBol.min(),model.IBol.max(),300)
        VBol_smooth = spline(model.IBol,model.VBol,IBol_smooth)
        ax_lc.plot(IBol_smooth,VBol_smooth,label='%.2E' % Decimal(str(new_parameter)))
        ax_lc.scatter(model.IBol, model.VBol,c='black',label='_nolegend_')

        t = np.linspace(0, 5,len(pulse))
        ax_pulse.plot(t,pulse,label='%.2E' % Decimal(str(new_parameter)))

    ax_lc.legend(loc='lower right', title=param_name)
    ax_lc.savefig("plot_output/loadcurve_{0}.png".format(param_name), bbox_inches='tight')

    ax_pulse.legend(loc='upper right', title=param_name)
    ax_pulse.savefig("plot_output/pulse_{0}.png".format(param_name), bbox_inches='tight')