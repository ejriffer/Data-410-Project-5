# Project-5

## Comparison of Different Regularization and Variable Selection Techniques

In this project, I applied and compared the different regularization techniques including Ridge, LASSO, Elastic Net, SCAD, and Square Root Lasso.

### Simulating Data

The data used was 100 sets with 1200 geatures, 200 observations and a toeplitz correlation structure such that the correlation between features *i* and *j* is approximatley ![CodeCogsEqn-7](https://user-images.githubusercontent.com/74326062/161297590-38c6bbd7-d7cb-41d7-95d0-1b3443e576f2.svg) with *p* = 0.8. The dependent variable *y* has the functional relationship ![CodeCogsEqn-8](https://user-images.githubusercontent.com/74326062/161297894-20f41078-da06-4ec0-b825-59953d22b860.svg) where ![CodeCogsEqn-9](https://user-images.githubusercontent.com/74326062/161297960-5449a073-24d8-4b88-889d-9862c0604962.svg) = 3.5, ![CodeCogsEqn-10](https://user-images.githubusercontent.com/74326062/161298209-e3fb8415-efba-472a-849e-86b47941e6d9.svg) is a column vector with ![CodeCogsEqn-11](https://user-images.githubusercontent.com/74326062/161298824-bb712222-c3bc-44c9-a532-4d63afdb81c3.svg) and ![CodeCogsEqn-12](https://user-images.githubusercontent.com/74326062/161299094-df0db8e0-4b14-4bba-b007-fbf317ec85c1.svg)
