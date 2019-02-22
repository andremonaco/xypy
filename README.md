Simulating Supervised Learning Data <img src="/img/Xy.png" alt="drawing" width="150px" align="right"/> 
===================================


With `Xy()` you can convienently simulate supervised learning data. The simulation can be
very specific, since there are many degrees of freedom for the user. For instance,
the functional shape of the nonlinearity is user-defined as well. Interactions can be formed and (co)variances altered. For a more specific motivation you can visit our [blog](https://www.statworx.com/de/blog/simulating-regression-data-with-xy-in-python/). I have adapted this package from my R version, which you can check out [here](https://www.github.com/andrebeleier/Xy).

### Simulate data 

You can simulate regression and classification data with interactions and a user-specified non-linearity. With the <code>stn</code> argument you can alter the signal to noise ratio of your simulation. I strongly encourage you to read this [blog post](https://www.statworx.com/de/blog/pushing-ordinary-least-squares-to-the-limit-with-xy/), where I've analyzed OLS coefficients with different signal to noise ratios.

```
# load the library
from Xy import Xy
# simulate regression data
my_sim = Xy(n = 1000, 
            numvars = [10,10], 
            catvars = [3, 2], 
            noisevars = 50, 
            stn = 100.0)
```

### Feature Selection

You can extract a feature importance of your simulation. For instance, to benchmark feature selection algorithms. You can read up on a small benchmark I did with this feature on our [blog](https://www.statworx.com/de/blog/benchmarking-feature-selection-algorithms-with-xy/). You can perform the same analysis easily in Python as well.

```
# Feature Importance 
my_sim.varimp()
```
<img src="/img/imp.png" alt="drawing"/> 

Feel free to [contact](mailto:andre.bleier@live.de) me with input and ideas.
