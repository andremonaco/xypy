from xypy import Xy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# simulate data
my_sim = Xy(n = 1000, # number of observations
            numvars = [2, 0],  # number of variables (10 linear 0 nonlinear)
            noisevars = 2,
            catvars = [1, 2], # one categorical variable with two levels
            stn = 100.0) # signal to noise ratio 10:1

# look at the true effects
#my_sim.varimp()

# extract the design matrix and 
X, y = my_sim.data(add_noise = False)

# build linear model
linreg = LinearRegression(fit_intercept = False)

# fit the model
mod = linreg.fit(X=X,y=y)

# extract the linear regression weights
mod.coef_

# extract the true model weights
my_sim.coef_

# compare
print(mod.coef_ - my_sim.coef_)