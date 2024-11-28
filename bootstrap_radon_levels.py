# Date: November 28th, 2024
# Author: Robert Stainforth (robert.stainforth@hc-sc.gc.ca)
# Description: This code demonstrates a bootstrap method for
# 			   computing the 90% confidence interval for a collections
#			   of samples (each of size n) randomly sampled from a parent
#			   lognormal distribution.
#
#			   Note 1: for a lognormal distribution, the median is equivalent
#					 to the geometric mean.
#			   Note 2: replace the TEST MODEL created in this code with the
#					   actual array of radon values from a specific radon study.
#					   Replace the 'rnp' numpy array with actual data.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Define a Lognormal distribution with a median (geometric mean)
# close to 84.7 Bq/m3 and for which has ~7.8% of the distribution > 200 Bq/m3
# This corresponds to a lognormal representation of the data
# collected as part of the CCRS 2024 study:
# reference: https://crosscanadaradon.ca/survey/
# Quote: 'When all Canadian data (for all regions, communities, and building types) 
#		  are combined in a manner that is balanced by distribution of these factors as 
#		  established by the 2021 Canada Census, the geometric average household radon 
#		  level was 84.7 Bq/m³, with just under 1 in 5 (17.8%) of single-detached, 
#		  semi-detached, and row-type residential buildings containing radon levels 
#		  that are at or over 200 Bq/m³.'
########################
### TEST MODEL BEGIN ###
########################
mu = 129.6
sigma = 150.35

a = 1 + (sigma / mu) ** 2
s = np.sqrt(np.log(a))
scale = mu / np.sqrt(a)

distr = stats.lognorm(s, 0, scale)


# generate some random values according to the model (1,000,000)
randomvals = distr.rvs(1_000_000)
# convert into a numpy array for ease-of-use
rnp = np.array(randomvals)

print("Target: 84.7 Bq/m3 with 17.8% above 200")
tot_num = rnp.size
percent_above_200 =  100.0*((rnp > 200.0).sum()/tot_num)
# median (=Geometric mean for Lognormal distributions)
gm_value = np.percentile(randomvals,50)

print("Gemetric mean of distribution: "+str(gm_value)+", % above 200 Bq/m3: "+str(percent_above_200))
######################
### TEST MODEL END ###
######################

# Number of random sampling iterations

# For each sample size, we will randomly sample 5000 (=num_iterations) times
num_iterations = 1000
sample_sizes = [5, 10, 25, 50, 100, 250, 500, 750, 1000, 2000, 10000, 100000]
# Ignore below, for testing
#sample_sizes = [5, 100000]

# For each set of sampling, we'll store the 90% lower and upper confidence bounds
sample_gm_lower_bounds = []
sample_gm_upper_bounds = []

for s in (sample_sizes):
 print("----------------------------------------")
 
 # For this sample size, store the observed geometric means
 cur_sample_gm_means = []
 print("Sample size: "+str(s)+", iterations: "+str(num_iterations))
	
 for i in range(0, num_iterations):
 
  # Perform a sample
  sampled_values = np.random.choice(rnp, size=s, replace=False)
  
  # Compute the geometric mean of the sample
  sample_gm_mean = np.percentile(np.array(sampled_values),50)
  
  # Store it in the current array for this sample size
  cur_sample_gm_means.append(sample_gm_mean) 	
  
  # Output some progress
  if ( (i % 100) == 0):
   print("--"+str(100.0*i/num_iterations)+ "% complete")
 
 # Compute the 5th percentile and 95th percentile (spread = 90% CI)
 sample_gm_lower_bounds.append(np.percentile(np.array(cur_sample_gm_means),5))
 sample_gm_upper_bounds.append(np.percentile(np.array(cur_sample_gm_means),95))
 
 #Output some progress on the confidence intervals being computed
 print("Current lower bounds: "+str(sample_gm_lower_bounds))
 print("Current upper bounds: "+str(sample_gm_upper_bounds))
 
### Figure 1, the parent distribution (sanity check)

plt.figure(1)
plt.hist(rnp, bins='auto')  # arguments are passed to np.histogram
plt.xlabel("Radon level (Bq/m3)")
plt.ylabel("Counts")
plt.title('')
plt.grid()
#plt.yscale("log")
plt.xlim(0, 1000)
plt.ylim(0, 20000)

### Figure 2, confidence intervals

plt.figure(2)
plt.plot(sample_sizes, sample_gm_lower_bounds, '--ro', linewidth=2.0)
plt.plot(sample_sizes, sample_gm_upper_bounds, '--bo', linewidth=2.0)

plt.xlabel("Sampling number (# radon tests per region)")
plt.ylabel("90% Confidence interval")
plt.title('')
plt.grid()
plt.xscale("log")
plt.xlim(1, 200000)
plt.ylim(0, 220)
plt.show()