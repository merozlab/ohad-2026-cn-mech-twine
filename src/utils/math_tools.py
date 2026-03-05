import numpy as np
# import math as m

def linfunc(x, a, b):
    x = np.asarray(x)
    return a * x + b

def shifted_linfunc(x, a, b, c):
    x = np.asarray(x)
    return a * (x - c) + b

def parafunc(x,a,b): return a*x**2+b # parabolic

def polyfunc(x,an): # n'th degree polynomial
    '''n is the degree of the polynomial'''
    for i in range(len(an)):
        y = an[i]*x**i
    return sum(y)

def one_over_x(x,a,b): return a / x + b  # 1/x decay


def sine_2param(t, A, w): return A * np.sin(w * t)

def logfunc(x, A, c):  return A * (np.log(x)) + c # logarithmic

def expfunc1(x, a, b):  return a * np.exp(b * x) # exponential rise

def expfunc2(x, a, b): return a * np.exp(- b * x) # decaying exponential

def expfunc3(x,a,b,c): return a *(1- np.exp(- b * x) ) + c # decaying growth exponential

def gaussfunc(x, a, b, c): return a * np.exp(-b * (x - c)**2) # gaussian

def powerfunc(x, a, b, c): return a * x**b + c # power function