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

For all of the following regularization techniques KFold validation was used in order to ensure coefficients, L2 distance and RMSE were as close to the actual value as possible and not due to a certain data split.

### Ridge

The follwing function runs KFold validation on the data. This function DoKFold_SK_FULL is used for Ridge, LASSO, and ElasticNet. 

```
def DoKFold_SK_FULL(X,y,model,k):
  avg_pos = []
  PE = []
  L2 = []
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    model.fit(X_train,y_train)
    beta_hat = model.coef_
    pos = np.where(beta_star != 0)
    pos_model = np.where(beta_hat != 0)
    avg_pos.append(np.intersect1d(pos,pos_model).shape[0])
    yhat_test = model.predict(X_test)
    PE.append(np.sqrt(MSE(y_test,yhat_test)))
    L2.append(norm(np.array(beta_star)-np.array(beta_hat)))
  return ('avg num of 0:', np.mean(avg_pos), 
          'avg RMSE:', np.mean(PE),
          'avg L2 distance:', np.mean(L2))
```

Before we run the Ridge regularization we need to find the best/most accurate *alpha* hyperparameter value for this specific technique. This can be done in a multitude of ways including GridSearchCV, or with a simple for loop. THe following code shows the tuning of the *alpha* hyperparameter. 

(The DoKFold_SK_pe function is the same as the DoKFold_SK_FULL as shown above but only returns the avg RMSE.)

```
# Find the best RIDGE alpha value
alpha = np.arange(0.01,2,0.001)
# pe = prediction error
PE = []
for a in alpha:
  model = Ridge(alpha=a,fit_intercept = False,max_iter=5000) 
  PE.append(DoKFold_SK_pe(x,y,model,10))
 alpha[np.argmin(PE)]
```

Running the above code we see that the best *alpha* value for Ridge is 0.01. We then run the DoKFold_SK_Full to see how Ridge performs. 

```
model = Ridge(alpha = 0.01, fit_intercept = False, max_iter = 10000)
DoKFold_SK_FULL(x,y,model,100)
```

The above code outputs:

('avg num of 0:',
 27.0,
 'avg RMSE:',
 5.068442072527193,
 'avg L2 distance:',
 3.0034425668074736)

### LASSO

### ElasticNet

### SCAD

### Square Root LASSO
