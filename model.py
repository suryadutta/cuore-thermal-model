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

def getConvergentTemps(alpha,beta,k,R0,Rl,T0,Cp,s,gamma,Vb,
                       burnin_stepSize=1e-4,burnin_t=300,
                       temp_ratio=1e-7, feedback_ratio=1e-4,
                       stop_at_convergence=True, verbose=True):         
    
    with np.errstate(invalid='raise'):
        try:
            #thermal model equations
            def cond(index,start,end):
                return ((k[index-1]/(beta[index-1]+1))*((start)**((beta[index-1])+1) - (end)**((beta[index-1])+1)))
            def Rntd(temp):
                return R0*math.exp((T0/temp)**gamma)
            def phonon(a,b,c,d,e,f):
                return ((Entd/dur) + cond(1,d,a) + cond(2,b,a) - cond(3,a,s))/(alpha[0]*(a**3))
            def electron(a,b,c,d,e,f):
                return ((Eelectron/dur) + (f**2/Rntd(b)) - cond(2,b,a))/(alpha[1]*b)
            def heater(a,b,c,d,e,f):
                return ((Eheater/dur) + cond(4,d,c) - cond(5,c,s))/(alpha[2]*c)
            def crystal(a,b,c,d,e,f):
                return ((Ecrystal/dur) - cond(1,d,a) - cond(4,d,c) - cond(6,d,e))/(alpha[3]*(d**3))
            def teflon(a,b,c,d,e,f):
                return (cond(6,d,e) - cond(7,e,s)) / ((alpha[4]*e) + (alpha[5]*(e**3)))
            def feedback (a,b,c,d,e,f):
                return (Vb-(f*((Rl+Rntd(b))/(Rntd(b)))))/(Rl*Cp)

            # fourth order Runge-Kutta method in 6 dimensions
            def rK6(a, b, c, d, e, f, fa, fb, fc, fd, fe, ff, hs):
                a1 = fa(a, b, c, d, e, f)*hs
                b1 = fb(a, b, c, d, e, f)*hs
                c1 = fc(a, b, c, d, e, f)*hs
                d1 = fd(a, b, c, d, e, f)*hs
                e1 = fe(a, b, c, d, e, f)*hs
                f1 = ff(a, b, c, d, e, f)*hs
                ak = a + a1*0.5
                bk = b + b1*0.5
                ck = c + c1*0.5
                dk = d + d1*0.5
                ek = e + e1*0.5
                fk = f + f1*0.5
                a2 = fa(ak, bk, ck, dk, ek, fk)*hs
                b2 = fb(ak, bk, ck, dk, ek, fk)*hs
                c2 = fc(ak, bk, ck, dk, ek, fk)*hs
                d2 = fd(ak, bk, ck, dk, ek, fk)*hs
                e2 = fe(ak, bk, ck, dk, ek, fk)*hs
                f2 = ff(ak, bk, ck, dk, ek, fk)*hs
                ak = a + a2*0.5
                bk = b + b2*0.5
                ck = c + c2*0.5
                dk = d + d2*0.5
                ek = e + e2*0.5
                fk = f + f2*0.5
                a3 = fa(ak, bk, ck, dk, ek, fk)*hs
                b3 = fb(ak, bk, ck, dk, ek, fk)*hs
                c3 = fc(ak, bk, ck, dk, ek, fk)*hs
                d3 = fd(ak, bk, ck, dk, ek, fk)*hs
                e3 = fe(ak, bk, ck, dk, ek, fk)*hs
                f3 = ff(ak, bk, ck, dk, ek, fk)*hs
                ak = a + a3
                bk = b + b3
                ck = c + c3
                dk = d + d3
                ek = e + e3
                fk = f + f3
                a4 = fa(ak, bk, ck, dk, ek, fk)*hs
                b4 = fb(ak, bk, ck, dk, ek, fk)*hs
                c4 = fc(ak, bk, ck, dk, ek, fk)*hs
                d4 = fd(ak, bk, ck, dk, ek, fk)*hs
                e4 = fe(ak, bk, ck, dk, ek, fk)*hs
                f4 = ff(ak, bk, ck, dk, ek, fk)*hs
                a = a + (a1 + 2*(a2 + a3) + a4)/6
                b = b + (b1 + 2*(b2 + b3) + b4)/6
                c = c + (c1 + 2*(c2 + c3) + c4)/6
                d = d + (d1 + 2*(d2 + d3) + d4)/6
                e = e + (e1 + 2*(e2 + e3) + e4)/6
                f = f + (f1 + 2*(f2 + f3) + f4)/6
                return a, b, c, d, e, f

            #power input parameters
            Ecrystal, Entd, Eheater, Eelectron = 0,0,0,0
            dur = 1e-3
            sampling_rate = 125

            # initial conditions
            a,b,c,d,e,f,hs = s,s,s,s,s,0,burnin_stepSize

            b_data, feedback_data, r_data = [],[],[]
            b_previous, f_previous = 0,0

            if verbose:
                burnin_step_range = tqdm(range(int((burnin_t)/burnin_stepSize)))
            else:
                burnin_step_range = range(int((burnin_t)/burnin_stepSize))

            #burnin steps - get to convergence
            for i in burnin_step_range:
                
                if stop_at_convergence and len(feedback_data)>100:
                    if (np.absolute(b - b_previous)/b < burnin_stepSize*temp_ratio and np.absolute(f - f_previous)/f < burnin_stepSize*feedback_ratio):
                        if verbose:
                            print('Convergence Reached')
                        return ([a,b,c,d,e,f], Vb, b_data, feedback_data, r_data)
                b_previous = b
                f_previous = f

                if i%int(1/(sampling_rate*burnin_stepSize))==0:
                    b_data.append(b)
                    feedback_data.append(1000*f)
                    r_data.append(Rntd(b)*1e-9)

                a, b, c, d, e, f = rK6(a, b, c, d, e, f, phonon, electron, heater, crystal, teflon, feedback, hs) 
            
            return ([a,b,c,d,e,f], Vb, b_data, feedback_data, r_data)
                            
        except:
            if verbose:
                print("Increasing Step Size")
            return getConvergentTemps(alpha,beta,k,R0,Rl,T0,Cp,s,gamma,Vb,burnin_stepSize=burnin_stepSize/2)

def simulatePulse(alpha,beta,k,R0,Rl,T0,Cp,s,gamma,Vb,
                  a=0.015,b=0.015,c=0.015,d=0.015,e=0.015,f=0,
                  event_energy = 4.0487e-13,
                  active_stepSize=1e-4,active_t=5,
                  stop_at_convergence=True, verbose=True):
    
    a_s, b_s, c_s, d_s, e_s, f_s = a, b, c, d, e, f
        
    with np.errstate(invalid='raise'):
        try:

            #thermal model equations
            def cond(index,start,end):
                return ((k[index-1]/(beta[index-1]+1))*((start)**((beta[index-1])+1) - (end)**((beta[index-1])+1)))
            def Rntd(temp):
                return R0*math.exp((T0/temp)**gamma)
            def phonon(a,b,c,d,e,f):
                return ((Entd/dur) + cond(1,d,a) + cond(2,b,a) - cond(3,a,s))/(alpha[0]*(a**3))
            def electron(a,b,c,d,e,f):
                return ((Eelectron/dur) + (f**2/Rntd(b)) - cond(2,b,a))/(alpha[1]*b)
            def heater(a,b,c,d,e,f):
                return ((Eheater/dur) + cond(4,d,c) - cond(5,c,s))/(alpha[2]*c)
            def crystal(a,b,c,d,e,f):
                return ((Ecrystal/dur) - cond(1,d,a) - cond(4,d,c) - cond(6,d,e))/(alpha[3]*(d**3))
            def teflon(a,b,c,d,e,f):
                return (cond(6,d,e) - cond(7,e,s)) / ((alpha[4]*e) + (alpha[5]*(e**3)))
            def feedback (a,b,c,d,e,f):
                return (Vb-(f*((Rl+Rntd(b))/(Rntd(b)))))/(Rl*Cp)

            # fourth order Runge-Kutta method in 6 dimensions
            def rK6(a, b, c, d, e, f, fa, fb, fc, fd, fe, ff, hs):
                a1 = fa(a, b, c, d, e, f)*hs
                b1 = fb(a, b, c, d, e, f)*hs
                c1 = fc(a, b, c, d, e, f)*hs
                d1 = fd(a, b, c, d, e, f)*hs
                e1 = fe(a, b, c, d, e, f)*hs
                f1 = ff(a, b, c, d, e, f)*hs
                ak = a + a1*0.5
                bk = b + b1*0.5
                ck = c + c1*0.5
                dk = d + d1*0.5
                ek = e + e1*0.5
                fk = f + f1*0.5
                a2 = fa(ak, bk, ck, dk, ek, fk)*hs
                b2 = fb(ak, bk, ck, dk, ek, fk)*hs
                c2 = fc(ak, bk, ck, dk, ek, fk)*hs
                d2 = fd(ak, bk, ck, dk, ek, fk)*hs
                e2 = fe(ak, bk, ck, dk, ek, fk)*hs
                f2 = ff(ak, bk, ck, dk, ek, fk)*hs
                ak = a + a2*0.5
                bk = b + b2*0.5
                ck = c + c2*0.5
                dk = d + d2*0.5
                ek = e + e2*0.5
                fk = f + f2*0.5
                a3 = fa(ak, bk, ck, dk, ek, fk)*hs
                b3 = fb(ak, bk, ck, dk, ek, fk)*hs
                c3 = fc(ak, bk, ck, dk, ek, fk)*hs
                d3 = fd(ak, bk, ck, dk, ek, fk)*hs
                e3 = fe(ak, bk, ck, dk, ek, fk)*hs
                f3 = ff(ak, bk, ck, dk, ek, fk)*hs
                ak = a + a3
                bk = b + b3
                ck = c + c3
                dk = d + d3
                ek = e + e3
                fk = f + f3
                a4 = fa(ak, bk, ck, dk, ek, fk)*hs
                b4 = fb(ak, bk, ck, dk, ek, fk)*hs
                c4 = fc(ak, bk, ck, dk, ek, fk)*hs
                d4 = fd(ak, bk, ck, dk, ek, fk)*hs
                e4 = fe(ak, bk, ck, dk, ek, fk)*hs
                f4 = ff(ak, bk, ck, dk, ek, fk)*hs
                a = a + (a1 + 2*(a2 + a3) + a4)/6
                b = b + (b1 + 2*(b2 + b3) + b4)/6
                c = c + (c1 + 2*(c2 + c3) + c4)/6
                d = d + (d1 + 2*(d2 + d3) + d4)/6
                e = e + (e1 + 2*(e2 + e3) + e4)/6
                f = f + (f1 + 2*(f2 + f3) + f4)/6
                return a, b, c, d, e, f

            #power input parameters
            Ecrystal, Entd, Eheater, Eelectron = 0,0,0,0
            dur = 1e-3
            sampling_rate = 125

            # initial conditions
            hs = active_stepSize
            a_data, b_data, c_data, d_data, e_data, feedback_data, r_data = [],[],[],[],[],[],[]

            if verbose:
                active_step_range = tqdm(range(int((active_t)/active_stepSize)))
            else:
                active_step_range = range(int((active_t)/active_stepSize))

            #active steps - model pulse
            for i in active_step_range:
                
                #turn off power deposition
                if (i>int((active_t/5)/active_stepSize) and i<int((dur + active_t/5)/active_stepSize)):
                    Ecrystal = event_energy
                else:
                    Ecrystal = 0

                if i%int(1/(sampling_rate*active_stepSize))==0:
                    a_data.append(a)
                    b_data.append(b)
                    c_data.append(c)
                    d_data.append(d)
                    e_data.append(e)
                    feedback_data.append(1000*f)
                    r_data.append(Rntd(b)*1e-9)

                a, b, c, d, e, f = rK6(a, b, c, d, e, f, phonon, electron, heater, crystal, teflon, feedback, hs) 

            return (a_data, b_data, c_data, d_data, e_data, feedback_data, r_data)


        except:
            if verbose:
                print("Increasing Step Size")
            return simulatePulse(alpha,beta,k,R0,Rl,T0,Cp,s,gamma,Vb,
                                 a=a_s,b=b_s,c=c_s,d=d_s,e=e_s,f=f_s,
                                 active_stepSize=active_stepSize/2)

def runFullModel(alpha,beta,k,R0,Rl,T0,Cp,s,gamma,Vb):

    converged_temps, Vb, b_data_burnin, feedback_data_burnin, r_data_burnin = getConvergentTemps(alpha,beta,k,R0,Rl,T0,Cp,s,gamma,Vb)

    a_data_pulse, b_data_pulse, c_data_pulse, d_data_pulse, e_data_pulse, feedback_data_pulse, r_data_pulse = simulatePulse(alpha,beta,k,R0,Rl,T0,Cp,s,gamma,Vb,*tuple(converged_temps))

    return [Vb, b_data_burnin, feedback_data_burnin, r_data_burnin, a_data_pulse, b_data_pulse, c_data_pulse, d_data_pulse, e_data_pulse, feedback_data_pulse, r_data_pulse]

def getLoadCurve(VBias,alpha,beta,k,R0,Rl,T0,Cp,s=0.015,gamma=0.5):

    modeled_data = pd.DataFrame(columns=['VBias','VBol','RBol','IBol'])
        
    pool = mp.Pool(processes=len(VBias))
    results = [pool.apply_async(runFullModel, 
                                args=(alpha,beta,k,R0,Rl,T0,Cp,s,gamma,Vb)) 
               for Vb in VBias]
    
    results = [p.get() for p in results]
    results.sort()

    VBol_max = 0
    pulse = []

    for result in results:

        VBol = result[2][-1]
        RBol = result[3][-1]
        IBol = VBol/RBol
        df = pd.DataFrame([(result[0],VBol,RBol,IBol)],columns=['VBias','VBol','RBol','IBol'])
        modeled_data = modeled_data.append(df)

        if VBol > VBol_max:
            VBol_max = VBol
            pulse = result[5]
    
    #clean up
    del results
    pool.close()
    pool.join()
    
    return modeled_data, pulse
