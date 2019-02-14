from Xy import Xy
import seaborn as sns
import matplotlib.pyplot as plt
my_sim = Xy(numvars = [2,3], catvars = [2,4],
            nlfun= lambda x: x**2,
            weights = [-10,10], stn=800.0,
            cor=[.1,.9], interactions = 2, noisevars = 5)

X, y = my_sim.data(add_noise = False)
my_sim.weights()
#my_sim.plot()
varimp = my_sim.varimp()