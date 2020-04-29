import numpy as np
import scipy.stats as stats


def categorize(sample):
	# Restrict sample to likert-range
	sample[sample < 1] = 1
	sample[sample > 7] = 7
	return sample.round()

def restrict(sample):
	sample[sample < 1] = 1
	sample[sample > 7] = 7
	return sample

def sampleBeta(numSamples=100,discrete=True,start=1,end=7):
	sample = np.random.beta(a=0.5,b=0.5,size=numSamples)
	sample = (sample * (end - start)) + start
	if discrete:
		return categorize(sample)
	return restrict(sample)

def sampleUniform(numSamples=100,discrete=False,start=1,end=7):
	"""
	errors are independently from uniform distribution N(0,1)
	"""
	sample = np.random.uniform(1,8, size=numSamples)
	if discrete:
		return categorize(sample)
	return restrict(sample)

def sampleCorrelatedMultivariateNormal(medians,numSamples,within_rho=0,between_rho=0,discrete=True,paired="none"):
	if paired != "none":
		within_rho = 0.3
	sigma2 = 4
	covariance_matrix = [[sigma2 * between_rho for m in medians] for m in medians]
	for i in range(numSamples):
		covariance_matrix[i][i] = sigma2
	if paired == "two":
		for i in range(numSamples-1):
			if i % 2 != 0: continue
			covariance_matrix[i+1][i] = sigma2 * within_rho
			covariance_matrix[i][i+1] = sigma2 * within_rho
	if paired == "three":
		for i in range(numSamples-1):
			if i % 3 != 0: continue
			covariance_matrix[i+1][i] = sigma2 * within_rho
			covariance_matrix[i+2][i] = sigma2 * within_rho
			covariance_matrix[i+1][i+2] = sigma2 * within_rho
			covariance_matrix[i+2][i+1] = sigma2 * within_rho
			covariance_matrix[i][i+1] = sigma2 * within_rho
			covariance_matrix[i][i+2] = sigma2 * within_rho
	# for c in covariance_matrix:
	# 	print(c)
	samples = np.random.multivariate_normal(medians,covariance_matrix)
	if discrete:
		return categorize(samples)
	return restrict(samples)

# def sampleCorrelatedMultivariateNormal(medians,numSamples=100,rho=0,discrete=True,paired=False):
# 	if len(medians) == 2:
# 		covariance_matrix = np.array([
# 			[1, rho],
# 			[rho, 1]
# 		])
# 	else:
# 		covariance_matrix = np.array([
# 			[1, rho, rho],
# 			[rho, 1, rho],
# 			[rho, rho, 1]
# 		])
# 	samples = np.random.multivariate_normal(medians,covariance_matrix,size=numSamples)
# 	if discrete:
# 		return categorize(samples)
# 	return restrict(samples)

def sampleIndependentNormal(numSamples=100, offset=0, discrete=False,error_mean=0):
	"""
	errors are independently from normal distribution N(0,1)
	"""
	errors = np.random.normal(loc=error_mean, scale=1, size=numSamples)
	sample = errors + offset
	if discrete:
		return categorize(sample)
	return restrict(sample)

def sampleIndependentNormalNonConstantVariance(numSamples=100, offset=0, error_mean=0, discrete=False):
	"""
	errors are independently from normal distribution N(0,1)
	"""
	sigmas = np.random.uniform(1,5,size=numSamples)
	errors = np.random.normal(loc=error_mean, scale=sigmas, size=numSamples)
	sample = errors + offset
	if discrete:
		return categorize(sample)
	else:
		return sample

def sampleIndependentContinuousSymmetric(numSamples=100, offset=0, error_mean=0, discrete = False):
	"""
	errors are independent continuous symmetric: mean=median=0
	Logistic(0,1)
	"""
	errors = np.random.logistic(loc=error_mean, scale=1, size=numSamples)
	samples = offset + errors
	if discrete:
		return categorize(samples)
	return restrict(samples)


def sampleIndependentContinuousAsymmetric(numSamples=100, offset=0, discrete=False):
	"""
	errors are independent, continuous, median=0, but Asymmetric
	Gumbel Distribution:
		parameters: mu: location
					beta: scale
		constant: gamma - Euler- Mascheroni constant ~= 0.577216

		mean := mu + beta * gamma
		median := mu - beta * ln(ln(2))
	"""
	# Define Parameters so that median is 0 and mean > 0
	beta = 1
	mu = np.log(np.log(2))
	errors = stats.gumbel_r.rvs(loc=mu, scale=beta, size=numSamples)
	samples = offset + errors
	if discrete:
		return categorize(samples)
	return restrict(samples)

def generateDependentSamplesLatentNormal(numSamples=100, offset=0, clusterScale=0.25, numClusters=2, discrete=False):
	"""
	draw cluster means from normal distribution
	Then draw samples from each of those clusters
	"""
	clusterMeans = np.random.normal(loc=offset,scale=1,size = numClusters)
	samples = []
	for c in clusterMeans:
	    c_samp = np.random.normal(loc=c,scale=clusterScale,size=int(numSamples/numClusters))
	    samples = np.concatenate((samples,c_samp))
	if discrete:
		return categorize(samples)
	return restrict(samples)

def generateDependentSamplesLatentLogistic(numSamples=100, offset=0, numClusters=2, discrete=False):
	"""
	draw cluster means from logistic distribution
	Then draw samples from Normal distribution centered around each cluster mean
	"""
	clusterMeans = np.random.logistic(loc=offset,scale=1,size = numClusters)
	samples = []
	for c in clusterMeans:
	    c_samp = np.random.normal(loc=c,scale=0.25,size=numSamples / numClusters)
	    samples = np.concatenate((samples,c_samp))
	if discrete:
		return categorize(discrete)
	return restrict(samples)

if __name__ == "__main__":
	sampleNormalCorrelatedBetweenSubj(within_rho=0.8,medians=[1,1,1,1],numSamples=4,paired="two")
	sampleNormalCorrelatedBetweenSubj(within_rho=0.8,between_rho=0.2,medians=[1,1,1,1,1,1],numSamples=6,paired="three")
