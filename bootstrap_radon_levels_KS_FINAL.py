# Date: December 11th, 2024
# Author: Robert Stainforth (robert.stainforth@hc-sc.gc.ca)
# Description: This code demonstrates a bootstrap method for
# 			   computing the 90% confidence interval of the KS statistic 
#			   for a collection of samples (each of size n) randomly sampled,
#			   tested against a a parent lognormal distribution.
#			   The KS statistics plotted is the 'D' statistic, i.e. the max
#			   distance between the CDFs.
#
#			   Note 1: for a lognormal distribution, the median is equivalent
#					 to the geometric mean.
#			   Note 2: replace the TEST MODEL created in this code with the
#					   actual array of radon values from a specific radon study.
#					   Replace the 'rnp' numpy array with actual data.

# More useful information on KS tests
# Link 1: https://stackoverflow.com/questions/10884668/two-sample-kolmogorov-smirnov-test-in-python-scipy
# Link 2: https://sparky.rice.edu/astr360/kstest.pdf
# Link 3: https://blogs.sas.com/content/iml/2019/05/20/critical-values-kolmogorov-test.html

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Define a Lognormal distribution with a median (geometric mean)
# close to 84.7 Bq/m3 and for which ~17.8% of the distribution > 200 Bq/m3
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
# For each sample size, we will randomly sample =num_iterations times
# 10,000 takes quite a while, recommend 100 for testing
num_iterations = 10000

sample_sizes = [5, 10, 25, 50, 100, 250, 500, 750, 1000, 2000, 10000, 100000]

# KS values for 0.05 confidence taken from here:
# https://real-statistics.com/statistics-tables/kolmogorov-smirnov-table/
# Each of the elements in this array corresponds to the 0.05 critical value
# for the equivalent element in the array above ('sample_sizes')
ks_crit_d_05 = [0.56327, 0.40925, 0.26404, 0.18845, 0.13581, 0.085893786, 0.060736078, 0.0495908, 0.042946893, 0.030368039, 0.013581, 0.004294689]
# Ignore below, for testing
#sample_sizes = [1000000]

# For each set of sampling, we'll store the 90% lower and upper confidence bounds
# of the KS-test D-statistic
sample_ksd_lower_bounds = []
sample_ksd_upper_bounds = []
sample_ksd_medians = []

# ... and the p-Value
sample_ksp_lower_bounds = []
sample_ksp_upper_bounds = []
sample_ksp_medians = []

sample_arrs_ksd = []
sample_arrs_ksp = []

# ... and the fraction of samples > 200 Bq/m3
fr_above_200 = []
fr_above_200_lower_bounds = []
fr_above_200_upper_bounds = []


for s in (sample_sizes):
 print("----------------------------------------")
 
 # For this sample size, store the observed geometric means
 cur_sample_ksd = []
 cur_sample_ksp = []
 cur_fr_above_200 = []
 
 print("Sample size: "+str(s)+", iterations: "+str(num_iterations))
 
 # For more information on the critical value,
 # see: https://blogs.sas.com/content/iml/2019/05/20/critical-values-kolmogorov-test.html
 crit_value = 1.36*pow((s+1e6)/(s*1e6),0.5)
 print("Critical value: "+str(crit_value))

 count_above_crit = 0	
 for i in range(0, num_iterations):
 
  # Perform a sample
  sampled_values = np.random.choice(rnp, size=s, replace=False)
  
  count_fr = 0
  count_tot = 0
  for z in sampled_values:
  	count_tot = count_tot + 1
  	if (z > 200):
  		count_fr = count_fr + 1
  
  cur_fr_above_200.append(100.0*count_fr/count_tot)		
  	
  
  # Compute the KS test
  cur_ks_test = stats.kstest(sampled_values, rnp)
  
  # Obtain the D-statistic and pValue
  cur_ksd = cur_ks_test.statistic
  cur_ksp = cur_ks_test.pvalue
  
  # Store it in the current array for this sample size
  cur_sample_ksd.append(cur_ksd) 
  cur_sample_ksp.append(cur_ksp) 	
  
  # Output some progress
  if ( (i % 10) == 0):
   print("--"+str(100.0*i/num_iterations)+ "% complete")
 
 # Compute the 5th percentile and 95th percentile (spread = 90% CI)
 # KS-test D-Statistic
 sample_ksd_lower_bounds.append(np.percentile(np.array(cur_sample_ksd),5))
 sample_ksd_upper_bounds.append(np.percentile(np.array(cur_sample_ksd),95))
 sample_ksd_medians.append(np.percentile(np.array(cur_sample_ksd),50))
 
 #KS-test p-Value
 sample_ksp_lower_bounds.append(np.percentile(np.array(cur_sample_ksp),5))
 sample_ksp_upper_bounds.append(np.percentile(np.array(cur_sample_ksp),95))
 sample_ksp_medians.append(np.percentile(np.array(cur_sample_ksp),50))
 
 sample_arrs_ksd.append(cur_sample_ksd)
 sample_arrs_ksp.append(cur_sample_ksp)
 
 fr_above_200.append(np.percentile(np.array(cur_fr_above_200),50))
 fr_above_200_lower_bounds.append(np.percentile(np.array(cur_fr_above_200),5))
 fr_above_200_upper_bounds.append(np.percentile(np.array(cur_fr_above_200),95))
   
 #Output some progress on the confidence intervals being computed
 print("Current lower bounds KS D-Statistic: "+str(sample_ksd_lower_bounds))
 print("Current upper bounds KS D-Statistic: "+str(sample_ksd_upper_bounds))
 
 print("Current lower bounds KS p-Value: "+str(sample_ksp_lower_bounds))
 print("Current upper bounds KS p-Value: "+str(sample_ksp_upper_bounds))
 
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

###################################
### Figure 2, confidence intervals - KS test D-Statistic
plt.figure(2)
plt.plot(sample_sizes, sample_ksd_medians, '-r', linewidth=2.0, label='Average (median)')
plt.plot(sample_sizes, sample_ksd_lower_bounds, '--b', linewidth=2.0, label='90% C.I.')
plt.plot(sample_sizes, sample_ksd_upper_bounds, '--b', linewidth=2.0)
plt.plot(sample_sizes, ks_crit_d_05, color='black', linestyle='solid', linewidth=2.0, label='KS test critical value (0.05)')

legend = plt.legend(loc='upper right')

plt.xlabel("Sampling number (# radon tests per region)")
plt.ylabel("90% Confidence interval of KS-test D-statistic distribution")
plt.title('')
plt.grid()
plt.xscale("log")
#plt.xlim(1, 200000)
#plt.ylim(0, 220)

### Figure 3, confidence intervals - KS test p-Value
###################################
plt.figure(3)
plt.plot(sample_sizes, sample_ksp_lower_bounds, '--bo', linewidth=2.0)
plt.plot(sample_sizes, sample_ksp_upper_bounds, '--bo', linewidth=2.0)
plt.plot(sample_sizes, sample_ksp_medians, '--ro', linewidth=2.0)

plt.xlabel("Sampling number (# radon tests per region)")
plt.ylabel("90% Confidence interval of KS-test p-Value distribution")
plt.title('')
plt.grid()
plt.xscale("log")
#plt.xlim(1, 200000)
#plt.ylim(0, 220)

### Figure 4, confidence intervals - KS test p-Value
###################################
plt.figure(4)
plt.plot(sample_sizes, fr_above_200_lower_bounds, '--bo', linewidth=2.0)
plt.plot(sample_sizes, fr_above_200_upper_bounds, '--bo', linewidth=2.0)
plt.plot(sample_sizes, fr_above_200, '--ro', linewidth=2.0)

plt.xlabel("Sampling number (# radon tests per region)")
plt.ylabel("90% Confidence interval of % above 200 Bq/m3")
plt.title('')
plt.grid()
plt.xscale("log")
#plt.xlim(1, 200000)
#plt.ylim(0, 220)

plot_number = 5

for i in range(0,len(sample_sizes)):
 ### D-STATISTIC
 ###################################
 plt.figure(plot_number)
 plt.hist(sample_arrs_ksd[i], bins='auto')  # arguments are passed to np.histogram
 plt.xlabel("KS-test D-statistic (sample size: "+ str(sample_sizes[i])+")")
 plt.ylabel("Counts")
 plt.title('')
 plt.grid()
 plt.xscale("log")
 plt.xlim(1e-4, 1.05)
 #plt.ylim(0, 20000)
 plot_number = plot_number+1
 
 ### P-VALUE
 ###################################
 plt.figure(plot_number)
 plt.hist(sample_arrs_ksp[i], bins='auto')  # arguments are passed to np.histogram
 plt.xlabel("KS-test p-Value (sample size: "+ str(sample_sizes[i])+")")
 plt.ylabel("Counts")
 plt.title('')
 plt.grid()
 #plt.yscale("log")
 plt.xlim(0, 1.05)
 #plt.ylim(0, 20000)
 plot_number = plot_number+1

plt.show()