#!/usr/bin/env python
# coding: utf-8

# Import stuff
import numpy as np
from numpy import ma
from matplotlib import pyplot as plt
import george
from george import kernels
import emcee
import corner
import scipy.optimize as op

# ## Define all functions
def get_gp(y,mynoise):
    # Define the objective function (negative log-likelihood in this case).
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25
    
    # And the gradient of the objective function.
    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)
    
    kernel = np.var(y) * kernels.ExpSquaredKernel(metric=10)
    t = np.arange(len(y))
    gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
                   white_noise=np.log(mynoise**2),
                   fit_white_noise=True)
    # You need to compute the GP once before starting the optimization.
    gp.compute(t)
    

    # Run the optimization routine.
    p0 = gp.get_parameter_vector()
    results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
    
    # Update the kernel and print the final log-likelihood.
    gp.set_parameter_vector(results.x)
    print(gp.get_parameter_names())
    return gp

def get_boundaries(mu, mynoise, snr):
    bins = len(mu)
    left = bins
    right = 0
    index = 0
    threshold = snr*mynoise
    
    baseline = 0.0
    while left == bins and index < bins:
        if mu[index] - baseline > threshold and \
           mu[index+1] - baseline > threshold and \
           mu[index+2] - baseline > threshold and \
           mu[index+3] - baseline > threshold:
            left = index
        index += 1
    index = 1
    while right == 0 and index < bins:
        if mu[-index] - baseline > threshold and\
           mu[-(index+1)] - baseline > threshold and\
           mu[-(index+2)] - baseline > threshold and\
           mu[-(index+3)] - baseline > threshold:
            right = bins - index
        index += 1
    return left, right

def get_wX(mu, rms, X):
    bins = len(mu)
    #baseline = np.min(mu)
    baseline = 0.0
    mu = mu - baseline
    peak = np.max(mu)
    wX_level = X*peak /100.
    wX_level1sp = wX_level + rms
    wX_level1sn = wX_level - rms
    #   print ('Wx report: ',wX_level, wX_level1sp, wX_level1sn, peak) 
    left = bins
    left1p = bins
    left1n = bins
    right = 0
    right1p = 0
    right1n = 0
    index = 0
    while left == bins and index < bins:
        if mu[index] > wX_level:
            left = index
        index += 1
    index = 0
    while left1p == bins and index < bins:
        if mu[index] > wX_level1sp:
            left1p = index
        index += 1
    index = 0
    while left1n == bins and index < bins:
        if mu[index] > wX_level1sn:
            left1n = index
        index += 1
    index = 1
    while right == 0 and index < bins:
        if mu[-index] > wX_level:
            right = bins - index
        index += 1
    index = 0
    while right1p == 0 and index < bins:
        if mu[-index] > wX_level1sp:
            right1p = bins -index
        index += 1
    index = 0
    while right1n == 0 and index < bins:
        if mu[-index] > wX_level1sn:
            right1n = bins - index
        index += 1


    wX = right - left + 1
    wXp = right1n - left1n - wX + 1
    wXn = right1p - left1p - wX + 1
    return wX, wXp, wXn
def minirms(myprofile,segments):
    rms16 = np.zeros((segments))
    seglength = int(len(myprofile)/segments)
    for i in np.arange(segments):
        rms16[i] = np.std(myprofile[i*seglength:(i+1)*seglength])
    min16 = np.min(rms16)
    return min16

def is_outlier(points, thresh=6):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
# ## Define the main callable function pulseshape(1D_data_array)
def pulseshape(data,mysnr,mcmc):
    originalbins = len(data)
    # roll the profile to bring the peak to the centre
    binmax = np.argmax(data)
    rollbins = -np.argmin(data)
    #profile = np.roll(data, int(originalbins/4 - binmax))
    profile = np.roll(data, rollbins)
    #noise_start = np.std(profile[0:50])
    noise_start = minirms(profile,16) 
    print("noise start: ", noise_start)
    profile = profile/noise_start
    margin = int(originalbins/10)
    # in units of SNR from here
    
    # ## Start the GP 

    gp1 = get_gp(profile,1.0)

    # ## Find a window of data around the pulse and produce noiseless data
    mu_b, var_b = gp1.predict(profile, np.arange(len(profile)), return_var=True)
    #profileout = np.roll(mu_b, -int(originalbins/4 - binmax))
    profileout = np.roll(mu_b, -rollbins)
    residual = data - profileout*noise_start
    plt.show()
    outlier_mask = is_outlier(residual,6.0)
    mdata = ma.masked_array(residual, mask=outlier_mask)
    # residual noise after outlier removal
    residual_rms = mdata.std()
    # GP noise
    noise = np.sqrt(np.exp(gp1.get_parameter_vector()[1]))
    min_noise = min((noise,residual_rms, noise_start))
    print('Minimum estimate of noise is: ', min_noise)
    left, right = get_boundaries(mu_b*noise_start, min_noise, mysnr)
    #    print('SNR boundaries found: ', left, right, noise)
    if right == 0 and left == originalbins:
        return 0,margin,originalbins-margin,np.zeros((originalbins))
    bins = right + margin - (left - margin) 
    #    print('boundaries: ', left, right)
    if left > margin and right < originalbins-margin:
        base1 = int(left/2)
        base2 = int(right + 0.5*(originalbins-right)) 
        baseline = 0.5 * (np.mean(profile[0:base1]) + np.mean(profile[base2:]))
        std_baseline = 0.5 * (np.std(profile[0:base1]) + np.std(profile[base2:]))
    else:
        baseline = 0
        
    # ## Return the standard deviation of the noise, the left and
    # ## right pulse boundaries referring to the input array, and the
    # ## noiseless profile 
    #leftout = left - (originalbins/4 - binmax)
    #rightout = right - (originalbins/4 - binmax)
    leftout = left - rollbins
    rightout = right - rollbins
    if not mcmc:
        return noise_start, noise*noise_start, residual_rms, leftout, rightout, profileout*noise_start

    # Get a more accurate noiseless profile with mcmc for hyperparameter exploration
    #    print('Off pulse baseline to subtract is: ', baseline, std_baseline)
    yn = profile[left-min(left,margin):min(right+margin,len(profile))] #- baseline
    gp_last = get_gp(yn)
    t = np.arange(len(yn))
    mu, var = gp_last.predict(yn, t, return_var=True)
    #    print('Second GP')
    noise = np.sqrt(np.exp(gp_last.get_parameter_vector()[1]))
    #    print('GP Noise standard deviation: ', noise)

    def lnprob(p):
        # Trivial uniform prior. gp_last is a global
        if np.any((-100 > p[1:]) + (p[1:] > 100)):
                return -np.inf
        
        # Update the kernel and compute the lnlikelihood.
        gp_last.set_parameter_vector(p)
        return gp_last.lnlikelihood(yn, quiet=True)


    # Set up the sampler.
    nwalkers, ndim = 36, len(gp_last)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    
    # Initialize the walkers.
    p0 = gp_last.get_parameter_vector() + 1e-4 * np.random.randn(nwalkers, ndim)
    
    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, 200)
    
    print("Running production chain")
    sampler.run_mcmc(p0, 200)
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    fig = corner.corner(flat_samples);
    nsamples = 50
    allsamples  = np.zeros((nsamples, len(yn)))
    for i in range(nsamples):
        # Choose a random walker and step.
        w = np.random.randint(sampler.chain.shape[0])
        n = np.random.randint(sampler.chain.shape[1])
        gp_last.set_parameter_vector(sampler.chain[w, n])
        
        # Plot a single sample.
        # plt.plot(t, gp_last.sample_conditional(yn, t), "g", alpha=0.1)
        allsamples[i,:] = gp_last.sample_conditional(yn, t)
    noiseless=np.mean(allsamples,0)
    mu_b[left-min(left,margin):min(right+margin,len(profile))] = noiseless
    #profileout = np.roll(mu_b, -int(originalbins/4 - binmax))
    profileout = np.roll(mu_b, -rollbins)
    residual = data - profileout*noise_start
    residual_rms = np.std(residual)
    return noise_start, noise*noise_start, residual_rms, leftout, rightout, profileout*noise_start

#    plt.plot(t, yn, ".k")
#    plt.plot(t, noiseless, "-b")
#    plt.show()
def getflux(profile,left,right, std):
    maskedprofile = np.copy(profile)
#    maskedprofile[int(left):int(right)] = 0.0
    sumprofile = np.sum(maskedprofile[int(left):int(right)])
    sumall = np.sum(maskedprofile)

#    baseline = np.mean(maskedprofile)
    denom = len(profile)-(right-left+1)
    if denom == 0:
        baseline = 0.0
    else:
        baseline = (sumall - sumprofile)/denom
    errorflux =  np.sqrt(right-left+1)*std/float(len(profile))
    flux = (np.sum(profile[left:right]) - baseline*(right-left))/float(len(profile))
    return flux, errorflux

