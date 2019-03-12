# Imports ----------------------------------------------------------------------
import inspect
import random
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels import robust
from copy import copy
from functools import reduce
from types import FunctionType
from scipy.linalg import block_diag
from sklearn.preprocessing import OneHotEncoder, scale

# Class definition -------------------------------------------------------------
class Xy:
    """ Artificial supervised learning class """

    # Check input types --------------------------------------------------------
    def checktypes(function):
        def _f(*arguments, **kwargs):
            for index, argument in enumerate(inspect.getfullargspec(function)[0]):
                if kwargs.get(argument) is None:
                    if index > len(arguments)-1 or argument is 'self':
                        continue
                    if not isinstance(arguments[index], function.__annotations__[argument]):
                        raise TypeError("{} is not of type {}, it has value {}".format(argument, 
                                        function.__annotations__[argument], 
                                        arguments[index]))
                else:
                    if not isinstance(kwargs.get(argument), function.__annotations__[argument]):
                        raise TypeError("{} is not of type {}, it has value {}".format(argument, 
                                        function.__annotations__[argument], 
                                        kwargs.get(argument)))
            return function(*arguments, **kwargs)
        _f.__doc__ = function.__doc__
        return _f

    # Simulation ---------------------------------------------------------------
    @checktypes
    def __init__(self: '__main__.__init__',
                 n: int = 1000,
                 numvars: list = [2, 2],
                 catvars: list = [1, 2],
                 noisevars: int = 5,
                 nlfun: FunctionType = lambda x: x**2,
                 type: str = 'reg',
                 link: FunctionType = lambda x: x,
                 cutoff: float = 0.5,
                 interactions: int = 1,
                 sig: list = [1, 6],
                 cor: list = [-.5, .5],
                 weights: list = [-5, 5],
                 cormat: np.ndarray = np.array(0),
                 stn: float = 4.0,
                 seed: int = 1337,
                 noise_coll: bool = False,
                 intercept: bool = True):
            
        """  A function which simulates linear and nonlinear X and a corresponding
             target. The composition of the target is highly customizable.
             Furthermore, the polynomial degree as well as the functional shape of
             nonlinearity can be specified by the user. Additionally coviarance structure
             of the X can either be sampled by the function or specifically
             determined by the user.

            :param n: an integer specifying the number of observations.
            :param numvars: a numeric list specifying the number of linear and nonlinear
                            X For instance, [5, 10] corresponds to
                            five linear and ten non-linear X.
            :param catvars: a numeric vector determining the amount of categorical predictors.
                            With this vector you can choose how many categorical predictors should
                            enter the equation and secondly the respective amount of categories.
                            For instance, catvars = [2, 5] would correspond to creating
                            two categorical variables with five categories.
            :param noisevars: an integer determining the number of noise variables.
            :param nlfun: a lambda function transforming nonlinear variables.
            :param type: a character specifying the supervised learning task either 'reg' or 'class'.
            :param link: a lambda link function to be used to transform the target.
                         Will perform log-link if type is 'class'.
            :param cutoff: a float indicating the cutoff probability. Only relevant for 'type = 'class''.
            :param interactions: a vector of integer specifying the interaction depth of
                                 of regular X and autoregressive X if
                                 applicable.
            :param sig: a list [min, max] indicating the scale parameter to sample from.
            :param cor: a list [min, max] determining correlation to sample from.
            :param weights: a list [min, max] specifying the multiplication magnitude to sample from.
            :param cormat: a covariance np.array for the linear and nonlinear simulation.
                          Defaults to None which means the structure
                          will be sampled from argument 'cor'.
            :param stn: an integer value determining the signal to noise ratio.
                        Higher values lead to more signal and less noise.
            :param seed: an integer specifying the random number generator seed.
            :param noise_coll: a boolean determining noise collinearity with X
            :param intercept: a boolean indicating whether an intercept should enter the model
        """
        # save input
        self.input = locals()

        # set seed
        random.seed(seed)

        # Functions ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def collapse(x, char=''):
            """ Collapse a list of strings by a character  """
            return reduce(lambda x, y: str(x) + char + str(y), x)

        # Input handling +++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # coerce to integer/float if character list
        coercetoint = lambda x: [int(i) for i in x]
        coercetofloat = lambda x: [float(i) for i in x]
        sig = coercetoint(sig)
        numvars = coercetoint(numvars)
        catvars = coercetoint(catvars)
        cor = coercetofloat(cor)
        
        # handle categorical values
        if len(catvars) is 1 and catvars is 0:
            catvars = [0, 0]
        if len(catvars) is 1 and catvars is not 0:
            raise ValueError('catvars has to be either zero ([0]) for no '\
                             'categorical effects or a list with two integers ([2, 2]).')
        
        # handle interaction depth
        if interactions >= sum(numvars):
            raise ValueError('the interaction depth is greater than the'\
                             ' amount of numerical features recude'\
                             ' the interaction depth to at least'\
                             ' to: ' + str(sum(numvars)-1) + '.')
        
        # TODO: try nlfun

        # TODO: try link function

        # Preliminaries ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # dictionary
        mapping = {'NLIN': numvars[1], 
                   'LIN': numvars[0], 
                   'NOISE': noisevars}

        # total number of variables
        vars = sum(mapping.values())
        
        if vars+catvars[0] is 0:
            raise ValueError('The total amount of variables you selected is 0')

        if noise_coll:
            sub_noise = 0
            if len(cormat.shape) > 0 and not vars:
               raise ValueError('You have specified collinearity between the\
                                 the features and the noise. Your prespecified\
                                 correlation matrix lacks' + vars-cormat.shape[0] +
                                 ' columns/rows.')
        else:
            sub_noise = noisevars

        # covariance handling ++++++++++++++++++++++++++++++++++++++++++++++++++
        if len(cormat.shape) is 0:
            cormat = np.random.uniform(low = min(cor), high = max(cor), 
                                      size = (vars-sub_noise, vars-sub_noise))
        
        # force symmetric correlation matrix
        CORR = cormat + cormat.T - np.diag(cormat.diagonal())
        np.fill_diagonal(CORR, 1)
        
        # handle noise between X and E
        if not noise_coll:
            noisecor = np.random.uniform(low = min(cor), high = max(cor), 
                                         size = (noisevars, noisevars))
            
            CORR = block_diag(CORR, noisecor)
            np.fill_diagonal(noisecor, 1)

        # sample standard deviations
        sds = np.random.uniform(low = min(sig), 
                                high = max(sig), 
                                size = vars)
  
        # create covariance matrix
        SIGMA = np.diag(sds) @ CORR @ np.diag(sds)
        SIGMA = SIGMA.transpose() @ SIGMA 

        # X sampling +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # sample features (and noise)
        X = np.random.multivariate_normal(mean = np.repeat(0, vars),
                                          cov = SIGMA,
                                          size = n)

        # center X
        X = scale(X, with_std=True)

        # split noise and  X
        X, E = np.hsplit(X, [vars-noisevars])

        # form column names
        all = sum([numvars, [catvars[0]], [noisevars]], [])
        chars = len(str(max(all)))
        
        # create design matrix names
        def feature_names(name, var, chars, catvars):
            """ Create column names from mapper """
            name_cont = []
            for k in range(0, len(name)):
                if var[k] is 0:
                    continue
                if name[k] is 'DUMMY':
                    name_cont.append(['DUMMY_' + str(i+1).zfill(chars) + '__' + str(j+2).zfill(chars) for i in range(0, var[k]) for j in range(0, catvars[1]-1)])
                else:
                    name_cont.append([name[k] + "_" + str(i+1).zfill(chars) for i in range(0, var[k])])
            return sum(name_cont, [])
            
        colnames = feature_names(['LIN', 'NLIN', 'DUMMY', 'NOISE'], all, 2, catvars)
        
        # transform nonlinear variables
        X_TRANS = copy(X)
        if numvars[1] > 0:
           nlin_ind = range(numvars[0], sum(numvars))
           X_TRANS[:, nlin_ind] = np.array([nlfun(X_TRANS[:, i]) for i in nlin_ind]).transpose()
        
        # handle categorical features
        X_DUM = [np.random.randint(catvars[1], size = n) for i in range(0, catvars[0])]
        X_DUM = np.array(X_DUM).transpose() 
        enc = OneHotEncoder(handle_unknown='error', categories='auto')
        if X_DUM.shape[0] > 0:
            enc.fit(X_DUM)
            X_DUM = enc.transform(X_DUM).toarray()
            X_DUM = np.delete(X_DUM, np.arange(0, np.prod(catvars), catvars[1]), 1)
        else: 
            X_DUM = [] # TODO: fix this right
        
        # handle interactions ++++++++++++++++++++++++++++++++++++++++++++++++++

        # sample interactions
        def sample_interactions(x, weights, interactions):
            """ Sample interactions """
            for c in range(0, x.shape[1]):
                sample_value = np.random.choice(np.append(0, np.round(np.random.uniform(-1, 1, 1), 2)), interactions-1)
                pos = np.arange(x.shape[1])
                pos = pos[np.arange(len(pos)) != c]
                sample_pos = np.random.choice(pos, replace = False, size = interactions-1)
                x[sample_pos, c] = sample_value
            return x      

        # build interaction matrix (raw)
        INT = np.diag(np.round(np.random.uniform(low = min(weights), 
                                                 high = max(weights), 
                                                 size = sum(numvars)+catvars[0]*(catvars[1]-1)), 2))
        
        INT = sample_interactions(INT, weights, interactions)
        
        # TODO: too hacky | filter out irrelevant entries
        X_COMBINE = [X_TRANS, X_DUM]
        X_COMBINE = [e for e in X_COMBINE if len(e) > 0]
        if len(X_COMBINE) is 2:
            X_COMBINE = np.concatenate(X_COMBINE, axis=1)
        else:
            X_COMBINE = X_COMBINE[0]
        target = X_COMBINE @ INT @ np.array([1]*INT.shape[1])
        
        # handle noise +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        noise = np.random.normal(np.mean(target)/stn, np.var(target)/stn, size=n)

        target = target + noise
        noise_paste = 'e ~ N(0, ' + str(np.round(np.std(noise), 2)) + ')'
        
                # handle intercept +++++++++++++++++++++++++++++++++++++++++++++++++++++
        if intercept: 
            i_cept = float(np.mean(noise)) + float(np.random.normal(size=1))
            i_cept_paste = 'y = ' + str(np.round(i_cept, 2))
            I = np.ones((n,1))
            target = target + i_cept
        else:
            i_cept_paste = 'y = '
            I = np.zeros((n,0))

        # handle link/cutoff +++++++++++++++++++++++++++++++++++++++++++++++++++
        # TODO: redundant? because tested above?
        if 5 is link(5) and type is 'class':
               link = lambda x: np.exp(x)/ (1 + np.exp(x))

        try:
            target = link(target)
        except:
            ValueError('Could not apply link function.')
        
        if type is 'class':
            target = np.where(target>=cutoff, 1, 0)

        # transformation matrix ++++++++++++++++++++++++++++++++++++++++++++++++
        psi = []
        # add intercept
        if intercept:
            psi.append(np.array(i_cept))

        # linear and nonlinear variabels
        psi.append(INT)

        # noise variables
        if noisevars > 0:
            psi.append(np.identity(noisevars))

        # block diagonlize transformation matrix
        self.psi = block_diag(*psi)

        # form tgp +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def extract_names(x, var):
            """ Extracts the name out of the interaction matrix """
            psi = copy(x)
            psi[psi == 0] = 1
            weights = [np.round(np.prod(psi[:, i]), 2) for i in range(0, psi.shape[1])]
            out = []
            out_names = []
            for i in range(0, x.shape[1]):
                names = var[x[:, i] != 0]
                tmp = [names[i] for i in range(0, len(names))]
                out_names.append(collapse(tmp, ':'))
                out.append(str(np.where(weights[i] >= 0, ' + ', ' - ')) +
                           str(weights[i]) + collapse(tmp, ':'))
                                    

            return collapse(out, ''), out_names

        varnames = np.array(colnames)[['NOISE' not in i for i in colnames]]
        process, effect_names = extract_names(x = INT, var = varnames)  
        tgp = i_cept_paste + collapse([process, noise_paste], " + ")
        self.tgp = re.sub(" \\- \\-", " - ", tgp)

        # pandas frames ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def cbind(x):
            """ Column bind a list of matrices and remove Nones """
            x = [e for e in x if len(e) > 0]
            sub = [x[i].shape[1] is not 0 for i in range(0, len(x))]
            out = np.concatenate([i for (i, v) in zip(x, sub) if v], axis = 1)
            return out

        self.X = pd.DataFrame(cbind([I, X, X_DUM, E]))

        if intercept:
            colnames.insert(0,'Intercept')
            effect_names.insert(0, 'Intercept')
        self.X.columns = colnames
        self.names = effect_names
        self.y = pd.DataFrame({'y': target})
        
        # true weights ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        weights = self.psi @ np.ones((self.psi.shape[1], 1))
        psi = copy(self.psi)
        psi[psi == 0] = 1
        names = self.names
        select = [not bool(re.search('DUMMY.*__.*1$', str(i))) for i in names]
        weights = np.array([np.round(np.prod(psi[:, i]), 2) for i in np.where(select)[0].tolist()])
        self.coef_ = pd.DataFrame(np.reshape(weights, (1,len(weights))), columns=np.array(names)[select])

        return None
    

    # Print method -------------------------------------------------------------
    def __repr__(self):
        """ Print representation """ 
        if self.input['type'] is 'reg':
            type = 'regression'
        else:
            type = 'classification'
         
        # Type description
        out =  'Xy Simulation \n' \
                ' \t | \n' \
                ' \t | + task "' + type + '"\n' \
                ' \t | + observations ' + str(self.input['n']) + '\n' \
                ' \t | + interactions ' + str(self.input['interactions']) + 'D\n' \
                ' \t | + signal to noise ratio ' + str(self.input['stn']) + '\n' \

        # Effect description
        out = out +  ' \t | + effects \n' \
                     ' \t   | - linear ' + str(self.input['numvars'][0]) + '\n' \
                     ' \t   | - nonlinear ' + str(self.input['numvars'][1]) + '\n' \
                     ' \t   | - categorical ' + str(self.input['catvars'][0]) + '\n' \
                     ' \t   | - noise ' + str(self.input['noisevars']) + '\n' \

        # Interval description
        out = out + ' \t | + intervals \n' 
        if len(self.input['cor']) > 1:
            out = out + ' \t   | - correlation [' + str(self.input['cor'][0]) + ', ' + str(self.input['cor'][1]) + ']\n' 
        else:
            out = out + ' \t   | - correlation ' + str(self.input['cor'][0]) + '\n' 

        if len(self.input['weights']) > 1:
            out = out + ' \t   | - weights [' + str(self.input['weights'][0]) + ', ' + str(self.input['weights'][1]) + ']\n' 
        else:
            out = out + ' \t   | - weights ' + str(self.input['weights'][0]) + '\n' 
        
        if len(self.input['sig']) > 1:
            out = out + ' \t   | - sd [' + str(self.input['sig'][0]) + ', ' + str(self.input['sig'][1]) + ']\n' 
        else:
            out = out + ' \t   | - sd ' + str(self.input['sig'][0]) + '\n' 
        
        out = out + '\n\nTarget generating process: \n'

        out = out + self.tgp
        return out

    # Extract data method ------------------------------------------------------
    def data(self, add_noise=True):
        """ Return training data """
        if add_noise:
            return self.X, self.y
        else:
            select = [not bool(re.search('NOISE', str(i))) for i in self.X.columns]
            X = copy(self.X)
            X = X.loc[:,self.X.columns[select]]
            return X, self.y

    # Transform data method ----------------------------------------------------
    def transform(self):
        """ Transform design matrix to get true effects """
        X = copy(self.X)
        nlins = [bool(re.search('NLIN', str(i))) for i in self.X.columns]
        if sum(nlins) > 0:
            X_NLIN = X.loc[:,nlins]
            nlfun = self.input['nlfun']
            X.loc[:, nlins] =  np.array([nlfun(X_NLIN.iloc[:, i]) for i in range(0, sum(nlins))]).transpose()
        names = X.columns    
        TRANS = X @ self.psi
        TRANS.columns = names
        return TRANS

    # Plotting method ----------------------------------------------------------
    def plot(self):
        """ Plot the true underlying effects """
        df = pd.concat([self.X, self.y], axis = 1)
        df = df.iloc[:, [bool(re.search('LIN|y', str(i))) for i in df.columns]]
        df_melt = pd.melt(df, id_vars='y', var_name = 'feature', value_name = 'size')
        g = sns.FacetGrid(df_melt, col='feature')
        g.map(plt.scatter, 'size', 'y', color='#13235B', s=50, alpha=.7, linewidth=.5, edgecolor='white')
        g.map(sns.regplot, 'size', 'y', color='#00A378', lowess=True, scatter = False)
        plt.show()
        return None

    # Variable importance method -----------------------------------------------
    def varimp(self, plot=True):
        """ Plot feature importance """ 
        # transform X back
        X = copy(self.transform())
        X = X.iloc[:, [not bool(re.search('NOISE|y', str(i))) for i in X.columns]]
        feature_sum = pd.DataFrame(X.sum(axis= 1))
        E = self.y.subtract(np.array(feature_sum))
        E.columns = ['NOISE']
        df_plot = pd.concat([X, E], axis=1)
        imp_raw = df_plot.apply(lambda x, df_plot: np.abs(x) / df_plot.abs().sum(axis=1), 
                                df_plot=df_plot, axis = 0)
        out = imp_raw.aggregate([np.mean, np.median, np.std, robust.mad], axis=0).transpose()
        out = out.sort_values('median', ascending=False)

        if plot:
            imp_melt = pd.melt(imp_raw, var_name = 'feature', value_name = 'size')
            op = dict(markerfacecolor='#C62F4B',linestyle='none', marker = 'o',
                      markersize=5, alpha = .7, markeredgecolor='white')
            bp =   dict(linestyle='-', facecolor='white',
                        edgecolor='#13235B')
            mp = dict(linestyle='-', linewidth=1.5, color='#C62F4B')
            wp = dict(color = '#13235B')
            imp_plot = sns.boxplot(y='feature', x='size', 
                                   data=imp_melt, order=out.index.tolist(),
                                   flierprops=op, boxprops=bp, 
                                   medianprops=mp, whiskerprops=wp)
            imp_plot.set_title('Feature Importance')
            imp_plot.set(xlabel='importance', ylabel='')
            plt.show()

        return out
