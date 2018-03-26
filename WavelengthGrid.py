import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from matplotlib import rcParams

rcParams['image.cmap'] = 'viridis'
rcParams['figure.figsize'] = (5, 5)
rcParams['image.interpolation'] = 'none'

from scipy.stats import norm


class WavelengthSolution:
    """Best fit solutions for lines within slit-spectra arcs, and their wavelength solutions."""

    def __init__(self, arcdata, w1, dw, verbose=True, toplot=True, randomseed=0):
        """
        Args:
            arcdata (ndarray): 2D spectra array of arc to be identified
            w1 (float): wavelength at 1st x-pixel - initial guess solution
            dw (float): wavelength / pixel - initial guess solution
            verbose (bool): T/F verbose output
            toplot (bool): T/F show plots
            randomseed (int): numpy random number generator seed value
        """

        self.arcdata = arcdata
        self.w1 = w1
        self.dw = dw

        self.verbose = verbose
        self.toplot = toplot
        self.randomseed = randomseed

        self.fit_lines_fit = []
        self.fit_lines_x_o = []
        self.fit_lines_wavelength = []

        if self.toplot:
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(20, 10)
            ax.imshow(self.arcdata)

    def calculatewavelengths(self, wavelengths, pixel_widths, toplot=True, verbose=False):

        # inputs wavelength and outputs pixel
        self.wave_to_pixel = lambda wave : round(
            (wave - self.w1) / self.dw + 1, 3)

        self.wavelengths = wavelengths
        self.extract_centers = list(map(self.wave_to_pixel, self.wavelengths))
        self.extract_widths = pixel_widths

        for wavelength, extract_center, extract_width in zip(self.wavelengths, self.extract_centers, self.extract_widths):
            ylo = float(int(extract_center - extract_width / 2))
            yhi = float(int(extract_center + extract_width / 2))

            if self.verbose:
                print(ylo, yhi)

            returns = self.AddFitLine(ylo=ylo, yhi=yhi,
                                      initargs={'amplitude': 4000.0, 'x_o': None,  'sigma': 1.0, 'a': 0.0, 'b': 0.0, 'background': 230.0})
            x, y, p_solution, z_solution = returns

            if p_solution.amplitude >= 10:
                p_solution.x_o += ylo

                if self.verbose:
                    print(wavelength, 'added')

                self.fit_lines_fit.append(p_solution)
                self.fit_lines_x_o.append(p_solution.x_o)
                self.fit_lines_wavelength.append(wavelength)

    def addmultfitlines(self, allinitargs, allbounds={}):
        default_initargs = {'amplitude': 1000.0, 'x_o': None,
                            'sigma': 1.0, 'a': 0.0, 'b': 0.0, 'background': 0.0}

        pass




    def addfitline(self, ylo, yhi, xlo=None, xhi=None, plottitle='', addline=None,
                   initargs={'amplitude': 1000.0, 'x_o': None,
                             'sigma': 1.0, 'a': 0.0, 'b': 0.0, 'background': 0.0},
                   bounds={'a': (-0.001, 0.001),  'b': (-0.05, 0.05), 'amplitude': (0, np.inf)}):
        """Fits Poly-Gaussian Model to Slit Spectra Data within ylo, yhi and xlo, xhi limits.

        Args:
            ylo (int): y low of sub-region to fit.
            yhi (int): Y hi of sub-region to fit.
            xlo (int): X low of sub-region to fit.
            xhi (int): Y hi of sub-region to fit.
            initargs (dict): dictionary of init parameters
                amplitude (float): gaussian init model input, value must be zero or positive.
                x_o (float): gaussian init model input. If otherwise unspecified, x_o will be set to the mean of ylo and yhi.
                sigma (float): gaussian init model input.
                a (float): gaussian init model input.
                b (float): gaussian init model input.
                background (float): background value.
            bounds (dict): boundary arguments for fit function (see astropy.modelling).

        Returns:
            x (ndarray): x coord of np.mgrid.
            y (ndarray): y coord of np.mgrid.
            p_solution (astropy.modeling.core.polyGausian): fit model of region.
            z_solution (ndarray): fit model output z.
        """

        @custom_model
        def polyGausian(x, y, amplitude=1.0, x_o=0.0, sigma=1.0, a=0.0, b=0.0, background=0.0):
            def loc(x): return a * (x)**2 + b * (x) + x_o
            return amplitude * norm.pdf(y, loc(x), sigma) + background

        if initargs['x_o'] is None:
            initargs['x_o'] = (yhi - ylo) / 2

        z = self.arcdata[xlo:xhi, ylo:yhi]

        # building grid for specified area
        nx, ny = z.shape
        x, y = np.mgrid[:nx, :ny]

        # initial guess fit for astropy.modeling
        p_init = polyGausian(**initargs)

        if bounds is not {}:
            for bound_it in bounds.keys():
                p_init.bounds[bound_it] = bounds[bound_it]

        fit_p = LevMarLSQFitter()
        p = fit_p(p_init, x, y, z)

        def loc_func(x_all, a=initargs['a'], b=initargs['b'], x_o=initargs['x_o']):
            def loc(x): return a * (x)**2 + b * (x) + x_o
            return list(map(loc, x_all))

        # plotting line of best fit locations
        x_all = np.linspace(0, nx, 10)

        loc_all_init = loc_func(x_all)
        loc_all_pred = loc_func(
            x_all, a=p.a.value, b=p.b.value, x_o=p.x_o.value)

        if self.verbose:
            print('fit function', p)
            print ('fit funtion bounds', p_init.bounds)

        if self.toplot:
            # Plot of Modelled Data, Model & Residual
            title = 'ylo : yhi   ' + str(ylo) + ' : ' + str(yhi) + plottitle

            fig, ax = plt.subplots(1, 3)
            fig.set_size_inches(8 * 2, 2.5 * 2)
            fig.suptitle(title)

            cax = ax[0].imshow(z, origin='lower', interpolation='nearest')
            ax[0].set_title("Data")
            ax[0].set_aspect('auto')
            ax[0].scatter(loc_all_pred, x_all, c='r',
                          marker='x', label='fit function')
            plt.colorbar(cax, ax=ax[0])

            cax = ax[1].imshow(p(x, y), origin='lower',
                               interpolation='nearest')
            ax[1].set_title("Model")
            ax[1].set_aspect('auto')
            ax[1].scatter(loc_all_pred, x_all, c='r',
                          marker='x', label='fit function')
            ax[1].scatter(loc_all_init, x_all, c='b',
                          marker='x', label='init function')
            ax[1].legend(loc=1)
            plt.colorbar(cax, ax=ax[1])

            cax = ax[2].imshow(z - p(x, y), origin='lower',
                               interpolation='nearest')
            ax[2].set_title("Residual")
            ax[2].set_aspect('auto')
            ax[2].scatter(loc_all_pred, x_all, c='r',
                          marker='x', label='fit function')
            fig.colorbar(cax, ax=ax[2])
            plt.show()

        if addline != None:

            if self.verbose:
                print(addline, 'added')

            p.x_o += ylo

            self.fit_lines_fit.append(p)
            self.fit_lines_x_o.append(p.x_o.value)
            self.fit_lines_wavelength.append(addline)


        return x, y, p, p(x, y)


class SimulateLineFit:
    """Simulations of Model 2D PolyGaussian, and their astropy.modelling best-fit solutions."""

    def __init__(self, verbose=True, toplot=True, randomseed=0):
        self.verbose = verbose
        self.toplot = toplot
        self.randomseed = randomseed

    def Model2DLineFit(self, nx=128, ny=128,
                       modelargs={'amplitude': 1.0, 'x_o': 0.0, 'sigma': 1.0,
                                  'a': 0.0, 'b': 0.0, 'background': 0.0},
                       initargs={'amplitude': 1.0, 'x_o': 0.0, 'sigma': 1.0, 'a': 0.0, 'b': 0.0, 'background': 0.0}):
        """Models 2D Line Fit using a Polynomial (vertical) Gaussian (horizontal) combo function. Plots modelled data, fit, and residuals.

        Args:
            nx (int): X dimension length.
            ny (int): Y dimension length.
            modelargs (dict): dictionary of model parameters
                amplitude (float): gaussian model input.
                x_o (float): gaussian model input.
                sigma (float): gaussian model input.
                a (float): gaussian model input.
                b (float): gaussian model input.
                background (float): background value.
            initargs (dict): dictionary of init parameters
                amplitude (float): gaussian init model input.
                x_o (float): gaussian init model input.
                sigma (float): gaussian init model input.
                a (float): gaussian init model input.
                b (float): gaussian init model input.
                background (float): background value.

        Returns:
            x (ndarray): x coord of np.mgrid.
            y (ndarray): y coord of np.mgrid.
            p_model (astropy.modeling.core.polyGausian): reference model.
            z_model (ndarray): reference model output z.
            p_solution (astropy.modeling.core.polyGausian): fit model.
            z_solution (ndarray): fit model output z.
        """

        np.random.seed(self.randomseed)

        @custom_model
        def polyGausian(x, y, amplitude=1.0, x_o=0.0, sigma=1.0, a=0.0, b=0.0, background=0.0):
            def loc(x): return a * (x)**2 + b * (x) + x_o
            return amplitude * norm.pdf(y, loc(x), sigma) + background

        x, y = np.mgrid[:nx, :ny]

        m_ref = polyGausian(**modelargs)
        z = m_ref(x, y)
        z += np.random.normal(0., z.max() / 10, z.shape)

        # Fit the data using astropy.modeling
        p_init = polyGausian(**initargs)
        fit_p = LevMarLSQFitter()
        p = fit_p(p_init, x, y, z)

        def loc_func(x_all, a=modelargs['a'], b=modelargs['b'], x_o=modelargs['x_o']):
            def loc(x): return a * (x)**2 + b * (x) + x_o
            return list(map(loc, x_all))

        # plotting line of best fit locations
        x_all = np.linspace(0, nx, 10)

        loc_all = loc_func(
            x_all, a=modelargs['a'], b=modelargs['b'], x_o=modelargs['x_o'])
        loc_all_pred = loc_func(
            x_all, a=p.a.value, b=p.b.value, x_o=p.x_o.value)

        if self.verbose:
            print('init function', m_ref)
            print('fit function', p)
            print ('d(loc)', np.subtract(loc_all, loc_all_pred))

        if self.toplot:
            # Plot of Modelled Data, Model & Residual
            plt.figure(figsize=(8 * 2, 2.5 * 2))
            plt.subplot(1, 3, 1)
            plt.imshow(z, origin='lower', interpolation='nearest')
            plt.title("Data")
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(p(x, y), origin='lower', interpolation='nearest')
            plt.title("Model")
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(z - p(x, y), origin='lower', interpolation='nearest')
            plt.title("Residual")
            plt.colorbar()
            plt.show()

            # best fit line plot
            plt.imshow(p(x, y))
            plt.colorbar()
            plt.scatter(loc_all, x_all, c='red', marker='x')
            plt.scatter(loc_all_pred, x_all, c='red', marker='x')
            plt.gca().invert_yaxis()
            plt.show()

        return x, y, m_ref, z, p, p(x, y)


class BuildModel:

    def __init__(self, verbose=True, toplot=True, randomseed=0):
        self.verbose = verbose
        self.toplot = toplot
        self.randomseed = randomseed

        self.model = None

    def PolyGaus(self, modelargs={'amplitude': 1.0, 'x_o': 0.0, 'sigma': 1.0, 'a': 0.0, 'b': 0.0, 'background': 0.0},
                 bounds={'a': (-0.001, 0.001),  'b': (-0.05, 0.05), 'amplitude': (0, np.inf)}):

        @custom_model
        def polyGausian(x, y, amplitude=1.0, x_o=0.0, sigma=1.0, a=0.0, b=0.0, background=0.0):
            def loc(x): return a * (x)**2 + b * (x) + x_o
            return amplitude * norm.pdf(y, loc(x), sigma)

        p_init = polyGausian(**modelargs)

        if bounds is not {}:
            for bound_it in bounds.keys():
                p_init.bounds[bound_it] = bounds[bound_it]

        if self.model is not None:
            self.model += p_init
        else:
            self.model = p_init
        return self.model

    def __add__(self, other):
        return self.model + other.model

    def __radd__(self, other):
        return [self if (other == 0) else self.__add__(other)]
        # if other == 0:
        #     return self
        # else:
        #     return self.__add__(other)

    def plot(self, x, y):
        # Plot of Modelled Data, Model & Residual
        plt.figure(figsize=(8 * 2, 2.5 * 2))
        plt.subplot(1, 3, 2)
        plt.imshow(self.model(x, y), origin='lower', interpolation='nearest')
        plt.title("Model")
        plt.colorbar()
        plt.show()

    def testModel(self, modelargs, initargs, nx=128, ny=128):
        np.random.seed(self.randomseed)

        x, y = np.mgrid[:nx, :ny]
        m_ref = self.model(**modelargs)
        z = m_ref(x, y)
        z += np.random.normal(0., z.max() / 10, z.shape)

        # Fit the data using astropy.modeling
        p_init = self.model(**initargs)
        fit_p = LevMarLSQFitter()
        p = fit_p(p_init, x, y, z)

        def loc_func(x_all, a=modelargs['a'], b=modelargs['b'], x_o=modelargs['x_o']):
            def loc(x): return a * (x)**2 + b * (x) + x_o
            return list(map(loc, x_all))

        # plotting line of best fit locations
        x_all = np.linspace(0, nx, 10)

        loc_all = loc_func(
            x_all, a=modelargs['a'], b=modelargs['b'], x_o=modelargs['x_o'])
        loc_all_pred = loc_func(
            x_all, a=p.a.value, b=p.b.value, x_o=p.x_o.value)

        if self.verbose:
            print('init function', m_ref)
            print('fit function', p)
            print('d(loc)', np.subtract(loc_all, loc_all_pred))

        if self.toplot:
            # Plot of Modelled Data, Model & Residual
            plt.figure(figsize=(8 * 2, 2.5 * 2))
            plt.subplot(1, 3, 1)
            plt.imshow(z, origin='lower', interpolation='nearest')
            plt.title("Data")
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(p(x, y), origin='lower', interpolation='nearest')
            plt.title("Model")
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(z - p(x, y), origin='lower', interpolation='nearest')
            plt.title("Residual")
            plt.colorbar()
            plt.show()

            # best fit line plot
            plt.imshow(p(x, y))
            plt.colorbar()
            plt.scatter(loc_all, x_all, c='red', marker='x')
            plt.scatter(loc_all_pred, x_all, c='red', marker='x')
            plt.gca().invert_yaxis()
            plt.show()

        return x, y, m_ref, z, p, p(x, y)

#      - remove? redundancy in inputs (a, b vs. defaults inputs into polygaus function?)
#      - self. implemented (but also storage method for test fits?) (put it inside a 'test'  dictionary?)
# DONE - two dictionaries as inputs? (the init guess  parameters and the test model parameters) or not?
# DONE - dictionary key order (python 3 automatically does this)
