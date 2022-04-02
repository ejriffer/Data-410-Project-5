# Project-5

## Comparison of Different Regularization and Variable Selection Techniques

This project applies and compares different regularization techniques including Ridge, LASSO, ElasticNet, SCAD, and Square Root LASSO.

## Simulating Data

The data used was 100 sets with 1200 geatures, 200 observations and a toeplitz correlation structure such that the correlation between features *i* and *j* is approximatley ![CodeCogsEqn-7](https://user-images.githubusercontent.com/74326062/161297590-38c6bbd7-d7cb-41d7-95d0-1b3443e576f2.svg) with *p* = 0.8. The dependent variable *y* has the functional relationship ![CodeCogsEqn-8](https://user-images.githubusercontent.com/74326062/161297894-20f41078-da06-4ec0-b825-59953d22b860.svg) where ![CodeCogsEqn-9](https://user-images.githubusercontent.com/74326062/161297960-5449a073-24d8-4b88-889d-9862c0604962.svg) = 3.5, ![CodeCogsEqn-10](https://user-images.githubusercontent.com/74326062/161298209-e3fb8415-efba-472a-849e-86b47941e6d9.svg) is a column vector with ![CodeCogsEqn-11](https://user-images.githubusercontent.com/74326062/161298824-bb712222-c3bc-44c9-a532-4d63afdb81c3.svg) and ![CodeCogsEqn-12](https://user-images.githubusercontent.com/74326062/161299094-df0db8e0-4b14-4bba-b007-fbf317ec85c1.svg)

In Python this data can be simulated with the following code:

```
# how many observation
n = 200 # number of columns
p = 1200 # number of rows

# to generate groundtruth data
beta_star = np.concatenate(([1]*7,[0]*25, [0.25]*5, [0]*50, [0.7]*15, [0]*1098))

# generate data
# we need something like toeplitz([1,0.8,0.8**2,0.8**3,0.8**4,0.8**1199])
# use a loop:
v = []
for i in range(p):
  v.append(0.8**i)

# Generate the random samples
mu = [0]*p
np.random.seed(123)
sigma = 3.5

x = np.random.multivariate_normal(mu, toeplitz(v), size = n)
y = np.matmul(x,beta_star) + sigma*np.random.normal(0,1,n)
y = y.reshape(-1,1)
```

## Comparison

The points of comparison between the different regularization techniques include the acerage number of true non-zero coefficients, the L2 (Eucledian) distance to the ideal solution and the root mean squared error (RMSE). 

### Ridge



### LASSO

### ElasticNet

### SCAD

### Square Root LASSO
