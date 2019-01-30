# Imports ----------------------------------------------------------------------
import inspect
import random
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import copy
from functools import reduce
from types import FunctionType
from scipy.linalg import block_diag
from sklearn.preprocessing import OneHotEncoder


# Class definition -------------------------------------------------------------
class Xy:
    """ Artificial supervised learning class """

    # Check input types --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- 
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

    # Simulation --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- -
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
                 sig: list = [1, 1], 
                 cor: list = [0, 0.1], 
                 weights: list = [-5, 5], 
                 sigma: np.ndarray = np.array(0),
                 stn: int = 4, 
                 noise_coll: bool = False, 
                 intercept: bool = True
    ):
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
            :param sigma: a covariance np.array for the linear and nonlinear simulation.
                          Defaults to None which means the structure
                          will be sampled from argument 'cor'.
            :param stn: an integer value determining the signal to noise ratio.
                        Higher values lead to more signal and less noise.
            :param noise_coll: a boolean determining noise collinearity with X
            :param intercept: a boolean indicating whether an intercept should enter the model
        """    
        # save input
        self.input = locals()

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

         # handle noise collinearity
        if not noise_coll:
            # handle wrong dimensionality due to noise variables
            sub_noise = noisevars
        else:
            # handle wrong dimensionality due to noise variables
            sub_noise = 0

        # total number of variables
        vars = sum(mapping.values())

        # covariance handling ++++++++++++++++++++++++++++++++++++++++++++++++++
        if len(sigma.shape) is 0:
            sigma = np.random.uniform(low = min(cor), high = max(cor), 
                                      size = (vars-sub_noise, vars-sub_noise))
            np.fill_diagonal(sigma, 1)

            # force symmetric (necessary?)
            # sigma = np.maximum( sigma, sigma.transpose() )
            chol = np.linalg.cholesky(sigma)
            np.fill_diagonal(chol, 1)
        else: 
            # handle false misspecified sigma matrix
            if sigma.shape[0] is not vars:
                raise ValueError('your specified sigma matrix has not the expected'\
                             ' dimension of ' + str(vars) + 'x' + str(vars) +
                             '.')
            try:
                np.fill_diagonal(sigma, 1)
                chol = np.linalg.cholesky(sigma)
            except np.linalg.linalg.LinAlgError:
                raise np.linalg.linalg.LinAlgError('could not cholesky '\
                                                   'decompose your sigma matrix.')

        # X sampling +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        def sample_feature(n, sig, cat, catvars = [0,0]):
            """
            Sample from normal distribution with sampled scale parameter
                :param n: the number of observations.
                :param sig: the scale parameter interval e.g. [0, 0.1]
            """
            if not cat:    
                x = np.random.normal(loc = 0, scale = float(np.random.uniform(low = min(sig), 
                                                                    high = max(sig), 
                                                                    size = 1)), 
                            size = n)
            else:
                if catvars[1] is not 0:
                    x = np.random.randint(catvars[1], size = n)
                else:
                    x = []
            return x

        # sample features
        X = [sample_feature(n, sig, cat = False) for i in range(0, vars)]
        X = np.array(X).transpose()
        
        # split noise from features if there is no correlation between E and X
        if not noise_coll:
            chol = block_diag(chol, np.identity(sub_noise))

        # rotate X
        X = X @ chol

        # normalize X
        #X = (X - X.min(0)) / X.ptp(0)
        
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
        if numvars[1] > 0:
           nlin_ind = range(numvars[0], sum(numvars))
           X_TRANS = copy(X)
           X_TRANS[:, nlin_ind] = np.array([nlfun(X[:, i]) for i in nlin_ind]).transpose()
        
        # handle categorical features
        X_DUM = [sample_feature(n, sig, cat = True, catvars = catvars) for i in range(0, catvars[0])]
        X_DUM = np.array(X_DUM).transpose()
        enc = OneHotEncoder(handle_unknown='error')
        if X_DUM.shape[0] > 0:
            enc.fit(X_DUM)
            X_DUM = enc.transform(X_DUM).toarray()
            X_DUM = np.delete(X_DUM, np.arange(0, np.prod(catvars), catvars[1]), 1)
        
        
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
        
        target = np.concatenate([X_TRANS, X_DUM], axis=1) @ INT @ np.array([1]*INT.shape[1])

        # handle intercept +++++++++++++++++++++++++++++++++++++++++++++++++++++
        if intercept: 
            i_cept = abs(max(target)-min(target))*0.3
            i_cept_paste = 'y = ' + str(np.round(i_cept, 2))
            I = np.ones((n,1))
            target = target + i_cept
        else:
            i_cept_paste = 'y = '
            I = np.zeros((n,0))

        # handle noise +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        noise_n = np.random.normal(size=n)
        noise = noise_n * np.sqrt(np.var(target)/(stn*np.var(noise_n)))
        target = target + noise
        noise_paste = 'e ~ N(0, ' + str(np.round(np.std(noise), 2)) + ')'
        
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
            X = copy(self.X.loc[:,self.X.columns[select]])
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
        TRANS = np.concatenate([X, self.y]) @ self.psi
        return TRANS

    # Extract true weights method ----------------------------------------------
    def weights(self):
        """ Extract the true underlying model weights """
        weights = self.psi @ np.ones((self.psi.shape[1], 1))
        psi = copy(self.psi)
        psi[psi == 0] = 1
        names = self.names
        select = [not bool(re.search('DUMMY.*__.*1$', str(i))) for i in names]
        weights = np.array([np.round(np.prod(psi[:, i]), 2) for i in np.where(select)[0].tolist()])
        out = pd.DataFrame(np.reshape(weights, (1,len(weights))), columns=np.array(names)[select])
        return out

    # Plotting method ----------------------------------------------------------
    def plot(self):
        """ Plot the true underlying effects """
        df = pd.concat([self.X, self.y], axis = 1)
        df = df.iloc[:, [bool(re.search('LIN|y', str(i))) for i in df.columns]]
        df_melt = pd.melt(df, id_vars='y', var_name = 'feature', value_name = 'size')
        g = sns.FacetGrid(df_melt, col='feature')
        g.map(plt.scatter, 'size', 'y', color = "#13235B", s=50, alpha=.7, 
              linewidth=.5, edgecolor="white")
        return None

    # Variable importance method -----------------------------------------------
    def varimp(self):
        return None
