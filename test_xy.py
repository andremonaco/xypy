from Xy import Xy
my_sim = Xy(numvars = [2,3], catvars = [2,4], interactions = 1, noisevars = 5)
X, y = my_sim.data(add_noise = False)
my_sim.weights()
my_sim.plot()