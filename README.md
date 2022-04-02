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

Like Ridge, the first step for LASSO is to find the best *alpha* hyperparameter value. The code is very similar to Ridge.

```
alpha = np.arange(0.01,2,0.001)
# pe = prediction error
PE = []
for a in alpha:
  model = Lasso(alpha=a,fit_intercept = False, max_iter=5000) 
  PE.append(DoKFold_SK_pe(x,y,model,10))
alpha[np.argmin(PE)]
```

Running the above code we see that the best *alpha* value for LASSO is 0.1759. We then run the DoKFold_SK_Full to see how LASSO performs. 

```
model = Lasso(alpha = 0.1759, fit_intercept = False, max_iter = 10000)
DoKFold_SK_FULL(x,y,model,100)
```
The above code outputs:

('avg num of 0:',
 20.89,
 'avg RMSE:',
 3.3828730256081316,
 'avg L2 distance:',
 3.7209964150325554)
 
### ElasticNet

Like Ridge and LASSO the first step for ElasticNet is to find the best *alpha* hyperparameter value. The code is very similar to above.

```
alpha = np.arange(0.01,2,0.001)
# pe = prediction error
PE = []
for a in alpha:
  model = ElasticNet(alpha=a,fit_intercept = False,max_iter=5000) 
  PE.append(DoKFold_SK_pe(x,y,model,10))
alpha[np.argmin(PE)]
```

Running the above code we see that the best *alpha* value for ElasticNet is 0.1269. We then run the DoKFold_SK_Full to see how ElasticNet performs. 

```
model = ElasticNet(alpha = 0.1269, fit_intercept = False, max_iter = 10000)
DoKFold_SK_FULL(x,y,model,100)
```

The above code outputs:

('avg num of 0:',
 25.32,
 'avg RMSE:',
 3.4403095071645513,
 'avg L2 distance:',
 2.5332132230042226)

### SCAD

SCAD (or smoothly ciplled absolute deviation) is another regularization technique that attempts to alleviate bias issues that can arrise from the ones seen above. The code below shows how to impelement SCAD as well as a KFold validation function similar to the one used for 3 techniques seen above.

```
# SCAD

@jit
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part
    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))

def scad_model(X,y,lam,a):
  n = X.shape[0]
  p = X.shape[1]
  # we add aan extra columns of 1 for the intercept
  #X = np.c_[np.ones((n,1)),X]
  def scad(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return 1/n*np.sum((y-X.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))
  
  def dscad(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.array(-2/n*np.transpose(X).dot(y-X.dot(beta))+scad_derivative(beta,lam,a)).flatten()
  b0 = np.ones((p,1))
  output = minimize(scad, b0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 1e7,'maxls': 25,'disp': True})
  return output.x
  
def scad_predict(X,y,lam,a):
  beta_scad = scad_model(X,y,lam,a)
  n = X.shape[0]
  p = X.shape[1]
  # we add aan extra columns of 1 for the intercept
  X = np.c_[np.ones((n,1)),X]
  return X.dot(beta_scad)
  
def DoKFoldScad_FULL(X,y,lam,a,k):
  avg_pos = []
  PE = []
  L2 = []
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    beta_scad = scad_model(X_train,y_train,lam,a)
    n = X_test.shape[0]
    p = X_test.shape[1]
    # we add an extra columns of 1 for the intercept
    #X1_test = np.c_[np.ones((n,1)),X_test]
    yhat_scad = X_test.dot(beta_scad)
    PE.append(MSE(y_test,yhat_scad))

    pos = np.where(beta_star != 0)
    pos_model = np.where(beta_scad != 0)
    avg_pos.append(np.intersect1d(pos,pos_model).shape[0])
    L2.append(norm(np.array(beta_star)-np.array(beta_scad)))

  return ('avg num of 0:', np.mean(avg_pos), 
          'avg RMSE:', np.mean(PE),
          'avg L2 distance:', np.mean(L2))
```

Like the above regularization techniques SCAD has an *alpha* hyperparameter that needs to be optimized, but SCAD also has a second important hyperparameter *lambda*.

```
alpha = np.arange(0.01,2,0.01)
lambd = np.arange(0.01,2,0.01)
# pe = prediction error
PE = []
for a,l in zip(alpha, lambd): 
  PE.append(DoKFoldScad(x,y,l,a,10))
print('alpha:',alpha[np.argmin(PE)])
print('lambda:',lambd[np.argmin(PE)])
```

The above code shows that the optimal *alpha* value is 0.88 and the optimal *lambda* value is 0.88. We then run the DoKFoldScad_Full to see how SCAD performs. 

`DoKFoldScad_FULL(x,y,0.88,0.88,100)`

The above code outputs:

('avg num of 0:',
 27.0,
 'avg RMSE:',
 74.31705501598313,
 'avg L2 distance:',
 7.544019557110627)

### Square Root LASSO
