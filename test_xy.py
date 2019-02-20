from Xy import Xy
import seaborn as sns
import matplotlib.pyplot as plt
my_sim = Xy(numvars = [50,50], catvars = [2,4],
            nlfun= lambda x: x**2,
            weights = [-10,10], stn=800.0,
            cor=[.1,.9], interactions = 1, noisevars = 5)

X, y = my_sim.data(add_noise = False)
my_sim.weights()
#my_sim.plot()
varimp = my_sim.varimp()


import timeit

n_sim = 10
out = timeit.timeit('Xy(n = '+ str(n_sim) +',  numvars = [50,0])',
                    setup = 'from Xy import Xy', number = 10)