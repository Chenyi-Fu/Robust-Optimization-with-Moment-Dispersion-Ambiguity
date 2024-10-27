#!/usr/bin/env python
# coding: utf-8

# # Portfolio Optimization for figures 1-3
# ## Imports
# 
# The numerical experiments require the following packages:
# 1. RSOME (1.2.1 or above).
# 2. SciPy
# 3. NumPy
# 4. Pandas
# 5. Matplotlib 

# In[1]:


from rsome import ro
from rsome import dro
from rsome import msk_solver as msk
from rsome import E
import rsome as rso
import time
from scipy.optimize import golden

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Description
#
# In this numerical study, we consider a portfolio optimization problem that aims to minimize
# the conditional value-at-risk (CVaR) of the portfolio return for $N$ distinct financial assets.
# These assets are characterized by their uncertain returns, denoted as  $\tilde{\pmb{z}} \sim \mathbb{P}$,
# with the distribution $\mathbb{P}\in\mathcal{F}$ defined within an ambiguity set $\mathcal{F} \subseteq \mathcal{P}_0(\mathbb{R}^N)$.
# Given a specific CVaR risk preference parameterized by $\epsilon\in(0, 1)$, the robust portfolio optimization problem is formulated as
#
# $$
# \min\limits_{v\in\mathbb{R}, \pmb{x}\in\mathcal{X}}\left\{v + \frac{1}{\epsilon}\sup\limits_{\mathbb{P}\in\mathcal{F}}\mathbb{E}_{\mathbb{P}}\left[\max\{-\pmb{x}^{\top}\tilde{\pmb{z}} - v, 0\}\right]\right\} = \min\limits_{v\in\mathbb{R}, \pmb{x}\in\mathcal{X}}\left\{\frac{1}{\epsilon}\sup\limits_{\mathbb{P}\in\mathcal{F}}\mathbb{CE}_{\mathbb{P}}^{\infty}\left[\max\{-\pmb{x}^{\top}\tilde{\pmb{z}} - (1-\epsilon)v, \epsilon v\}\right]\right\}.
# $$
#
# The portfolio decision is defined by $\pmb{x}\in\mathcal{X}$,
# where $\mathcal{X}\in\left\{\pmb{x}\in\mathbb{R}_+^N\left|~\pmb{1}^{\top}\pmb{x} = 1\right.\right\}$.
#
# In the following numerical experiments, the portfolio return is constructed as random variable
# $\tilde{\pmb{z}}=\pmb{Q}\tilde{\pmb{xi}}$, where matrix $\pmb{Q}\in\mathbb{R}^{N\times N}$,
# and $\tilde{\pmb{\xi}}$ represents the hidden random factors that influence the portfolio returns.

# ## RO Models
#
# ### RO-MM
#
# The RO-MM model considers an ambiguity set with the marginal moment information:
#
# $$
# \mathcal{F} = \left\{
# \tilde{z} \in \mathcal{P}_0(\mathbb{R}^N)\left|
# \begin{array}
# ~\tilde{\pmb{z}} \sim \mathbb{P} \\
# \mathbb{E}_{\mathbb{P}}[\tilde{\pmb{z}}] = \pmb{\mu} \\
# \mathbb{E}_{\mathbb{P}}[(\tilde{z}_i - \mu_i)^2] = \sigma_i^2 & i\in[N] \\
# \end{array}
# \right.
# \right\}.
# $$
#
# The `ro_mm_cvar()` function defined below is used to solve the RO-MM model.

def ro_mm_cvar(mu, std, epsilon):
    """
    The function ro_mm_cvar solves a robust model with a marginal
    moment ambiguity set that aims to minimize the Conditional 
    Value-at-Risk (CVaR) of the portfolio return.  
    
    Parameters
    ----------
    mu : numpy.ndarray
        A one-dimensional array representing the estimated means
        of n stock returns. 
    std : numpy.ndarray
        A one-dimensional array representing the standard deviations
        of n stock returns. 
    epsilon : float
        The confidence level expressed as 1 - epsilon.
    
    Returns
    -------
    xs : numpy.ndarray
        A one-dimensional array representing the optimal portfolio
        decision.
    objval : float, numpy.float64
        The optimal objective value of the robust model.
    """
    
    n = len(mu)
    
    model = dro.Model()
    
    x = model.dvar(n)
    v = model.dvar()
    
    z = model.rvar(n)
    u = model.rvar(n)
    fset = model.ambiguity()
    fset.suppset(rso.square(z - mu) <= u)
    fset.exptset(E(z) == mu, E(u) == std**2)
    
    model.minsup(v + E((1/epsilon)*rso.maxof(-x@z-v, 0)), fset)
    model.st(x >= 0, x.sum() == 1)
    
    model.solve(msk, display=False)
    
    return x.get(), model.get()


# ### RO-CM
# 
# The RO-CM model extends the RO-MM model by incorporating additional covariance information, so the ambiguity set is written as
# 
# $$
# \mathcal{F} = \left\{
# \tilde{z} \in \mathcal{P}_0(\mathbb{R}^N)\left|
# \begin{array}
# ~\tilde{\pmb{z}} \sim \mathbb{P} \\
# \mathbb{E}_{\mathbb{P}}[\tilde{\pmb{z}}] = \pmb{\mu} \\
# \mathbb{E}_{\mathbb{P}}[(\tilde{\pmb{z}} - \pmb{\mu})(\tilde{\pmb{z}} - \pmb{\mu})^{\top}] \preceq \pmb{\Sigma} \\
# \end{array}
# \right.
# \right\},
# $$
# 
# where the covariance matrix is $\pmb{\Sigma} = \text{diag}\left(\sigma_1^2, ..., \sigma_N^2\right)$.

def ro_cm_cvar(mu, std, epsilon):
    """
    The function ro_cm_cvar solves a robust model with a cross-
    moment ambiguity set that aims to minimize the Conditional 
    Value-at-Risk (CVaR) of the portfolio return.  
    
    Parameters
    ----------
    mu : numpy.ndarray
        A one-dimensional array representing the estimated means
        of n stock returns. 
    std : numpy.ndarray
        A one-dimensional array representing the standard deviations
        of n stock returns. 
    epsilon : float
        The confidence level expressed as 1 - epsilon.
    
    Returns
    -------
    xs : numpy.ndarray
        A one-dimensional array representing the optimal portfolio
        decision.
    objval : float, numpy.float64
        The optimal objective value of the robust model.
    """
    
    n = len(mu)
    
    model = ro.Model()
    
    x = model.dvar(n)
    
    c = ((1-epsilon)/epsilon)**0.5
    model.min(-x@mu + c*rso.norm(std*x))
    model.st(x >= 0, x.sum() == 1)
    
    model.solve(msk, display=False)
    
    return x.get(), model.get()


# ### SAA
# 
# The Sample Average Approximation (SAA) model is written as 
# 
# $$
# \min\limits_{v\in\mathbb{R}, \pmb{x}\in\mathcal{X}}\left\{v + \frac{1}{\epsilon S}\sum\limits_{s\in[S]}\max\{-\pmb{x}^{\top}\check{\pmb{z}}_s - v, 0\}\right\}.
# $$
# 
# where the sample data $\check{\pmb{z}}_s\in\mathbb{R}^N$ are re-sampled from the training data, assuming each stock is independently distributed. 
# 
# The `saa_cvar()` function defined below is used to solve the SAA model.

# In[4]:


def saa_cvar(zhat, epsilon):
    """
    The function saa_cvar solves a Sample Average Approximation
    (SAA) model that aims to minimize the Conditional Value-at-
    Risk (CVaR) of the portfolio return.  
    
    Parameters
    ----------
    zhat : numpy.ndarray
        A two-dimensional array representing the sample data of
        stock returns. The shape of the array is (s, n), where s
        is the sample size, and n is the number of stocks.
    epsilon : float
        The confidence level expressed as 1 - epsilon.
    
    Returns
    -------
    xs : numpy.ndarray
        A one-dimensional array representing the optimal portfolio
        decision.
    objval : float, numpy.float64
        The optimal objective value of the SAA model.
    """
    
    s, n = zhat.shape
    
    model = ro.Model()
    
    x = model.dvar(n)
    v = model.dvar()
    y = model.dvar(s)
    
    model.min(v + (1/s)*y.sum())
    model.st(y >= 0, y >= (1/epsilon)*(-zhat@x - v))
    model.st(x >= 0, x.sum() == 1)
    
    model.solve(msk, display=False)
    
    return x.get(), model.get()


# ### DRO with Wasserstein Ambiguity Sets
# 
# In the distributionally robust optimization (DRO) model defined below
# 
# $$
# \min\limits_{v\in\mathbb{R}, \pmb{x}\in\mathcal{X}}\sup\limits_{\mathbb{P}\in\mathcal{F}^{W}(\theta)}\left\{v + \frac{1}{\epsilon}\mathbb{E}_{\mathbb{P}}\left[\max\{-\pmb{x}^{\top}\tilde{\pmb{z}}_s - v, 0\}\right]\right\},
# $$
# 
# the ambiguity set is defined as below to capture a Wasserstein centered at the training sample data $\check{\pmb{z}}_s\in\mathbb{R}^N$:
# 
# $$
# \mathcal{F}^W(\theta) = \left\{\mathbb{P}\in\mathcal{P}_0(\mathbb{R}^N\times\mathbb{R}\times[N])\left|
# \begin{array}{ll}
# (\tilde{\pmb{z}}, \tilde{u}, \tilde{s})\in\mathbb{P} \\
# \mathbb{E}_{\mathbb{P}}[\tilde{u}] \leq \theta \\
# \mathbb{P}\left[\left.\|\tilde{\pmb{z}} - \check{\pmb{z}}_s\|_1 \leq \tilde{u}\right|\tilde{s}=s\right] = 1 \\
# \mathbb{P}[\tilde{s}=s] = 1/S
# \end{array}
# \right.
# \right\}.
# $$
# 
# The `wass_cvar()` function defined below is used to solve the DRO model.

# In[5]:


def wass_dro_cvar(zhat, epsilon, theta):
    """
    The function wass_cvar solves a Sample Average Approximation
    (SAA) model that aims to minimize the Conditional Value-at-
    Risk (CVaR) of the portfolio return.  
    
    Parameters
    ----------
    zhat : numpy.ndarray
        A two-dimensional array representing the sample data of
        stock returns. The shape of the array is (s, n), where s
        is the sample size, and n is the number of stocks.
    epsilon : float
        The confidence level expressed as 1 - epsilon.
    theta : float
        The radius of the Wasserstein ball centered at the empirical
        distribution.
    
    Returns
    -------
    xs : numpy.ndarray
        A one-dimensional array representing the optimal portfolio
        decision.
    objval : float, numpy.float64
        The optimal objective value of the SAA model.
    """
    
    s, n = zhat.shape
    
    model = dro.Model(s)
    
    z = model.rvar(n)
    u = model.rvar()
    fset = model.ambiguity()
    for k in range(s):
        fset[k].suppset(rso.norm(z - zhat[k], 1) <= u)
    fset.exptset(E(u) <= theta)
    pr = model.p
    fset.probset(pr == 1/s)
    
    
    x = model.dvar(n)
    v = model.dvar()
    
    model.minsup(v + (1/epsilon)*E(rso.maxof(-z@x - v, 0)), fset)
    model.st(x >= 0, x.sum() == 1)
    
    model.solve(msk, display=False)
    
    return x.get(), model.get()


# In[6]:


def wass_cvar(zhat, epsilon, theta):
    """
    The function wass_cvar solves a Sample Average Approximation
    (SAA) model that aims to minimize the Conditional Value-at-
    Risk (CVaR) of the portfolio return.  
    
    Parameters
    ----------
    zhat : numpy.ndarray
        A two-dimensional array representing the sample data of
        stock returns. The shape of the array is (s, n), where s
        is the sample size, and n is the number of stocks.
    epsilon : float
        The confidence level expressed as 1 - epsilon.
    theta : float
        The radius of the Wasserstein ball centered at the empirical
        distribution.
    
    Returns
    -------
    xs : numpy.ndarray
        A one-dimensional array representing the optimal portfolio
        decision.
    objval : float, numpy.float64
        The optimal objective value of the SAA model.
    """
    
    s, n = zhat.shape
    
    model = ro.Model()
    
    x = model.dvar(n)
    v = model.dvar()
    c = model.dvar(s)
    eta = model.dvar()
    
    a1, b1 = 0, 1
    a2, b2 = -1/epsilon, 1 - 1/epsilon
    model.min(eta*theta + (1/s) * c.sum())
    model.st(b1*v + a1*(zhat@x) <= c, 
             b2*v + a2*(zhat@x) <= c)
    model.st(abs(-a1 * x) <= eta, 
             abs(-a2 * x) <= eta)
    model.st(x >= 0, x.sum() == 1)
    
    model.solve(msk, display=False)
    
    return x.get(), model.get()



# ### RO-$\Gamma$
# 
# #### Function for calculating the forward/backward deviations
# 
# According to DEFINITION 4 in the paper, the forward and backward deviations are defined as
# 
# $$
# \begin{cases}
# \overline{\sigma} = \sup\limits_{v > 0}\frac{1}{v}(2D_{\mathbb{P}}(v))^{1/2} \\
# \underline{\sigma} = \sup\limits_{v > 0}\frac{1}{v}(2D_{\mathbb{P}}(-v))^{1/2}
# \end{cases}
# $$
# 
# where the $D_{\mathbb{P}}(v)$ is the dispersion generating function, defined in DEFINITION 1 as
# 
# $$
# D_{\mathbb{P}}(v) := \ln \mathbb{E}_{\mathbb{P}}\left[\pmb{v}^{\top}(\tilde{\pmb{z}}- \mathbb{E}_{\mathbb{P}}[\tilde{\pmb{z}}])\right].
# $$
# 
# In the following numerical experiments, the forward and backward deviations are estimated based on the sample dataset using the `devs()` function, defined as follows. 



def devs(zhat):
    """
    The function devs calcualtes the forward/backward deviations of
    the given sample data. See DEFINITION 4 in the paper. 
    
    Parameters
    ----------
    zhat : np.ndarray
        A two-dimenional array as the training sample data. The shape
        of the array is (s, n), where s is the sample size, and n is
        the number of varaibles.
    
    Returns
    -------
    output : dict
        A dictionary where the forward deviation is given as the value
        of 'forward', and the backward deviation is given as the value
        of 'backward'. Both deviations are presented as one-dimensional
        arrays.
    
    Notes
    -----
    The value of v that maximizes the deviations are obtained using the
    golden section method. See the documentation of scipy.optimize.golden.
    """
    
    def pos(v, zvec):
        
        if v == 0:
            return zvec.std(ddof=0)
        else:
            return - (2*np.log(np.exp(zvec*v).mean())) ** 0.5 / v
    
    def neg(v, zvec):
        
        if v == 0:
            return zvec.std(ddof=0)
        else:
            return - (2*np.log(np.exp(-zvec*v).mean())) ** 0.5 / v
    
    fdev = [-golden(pos, args=(zhat[:, j],), brack=(0.0, 5.0), 
                    full_output=True)[1]
            for j in range(zhat.shape[1])]
    print(dict(forward=np.array(fdev)))
    bdev = [-golden(neg, args=(zhat[:, j],), brack=(0.0, 5.0), 
                    full_output=True)[1]
            for j in range(zhat.shape[1])]
    # print(dict(forward=np.array(fdev)))
    return dict(forward=np.array(fdev), 
                backward=np.array(bdev))


# #### `GMModel` class for the RO-$\Gamma$ model
# 
# In the code segment below, we create a subclass named `GMModel`. Besides inheriting all attributes and operations of the `rsome.ro.Model` class, the `GMModel` objects have several extra methods:
# 
# 1. `in_Phi()` enforces that the given variables $t$, $\kappa$, and $v$, are defined within the dispersion characteristic set $\Phi$. See <span style="font-variant:small-caps;"><b>Theorem 3</b></span> for more details. 
# 2. `in_Phi_Gamma()` enforces that the given variables $t$, $\kappa$, and $\pmb{v}$ are defined within the characteristic function $\Phi_{\Gamma}$, introduced in <span style="font-variant:small-caps;"><b>Theorem 7</b></span> (Equation (13)). 
# 3. `min_ce_infty()` defines the objective of the created model to be minimizing the worst-case expectation
# 
# $$\sup_{\mathbb{P}\in\mathcal{F}}\mathbb{E}_{\mathbb{P}}\left[f(\pmb{x}, \tilde{\pmb{z})}\right] = \sup_{\mathbb{P}\in\mathcal{F}}\mathbb{CE}_{\mathbb{P}}^{\infty}\left[f(\pmb{x}, \tilde{\pmb{z}})\right],$$
# 
# where $f(\pmb{x}, \pmb{z})=\max\limits_{k\in[K]}\{\pmb{a}_k(\pmb{x})^{\top}\pmb{z} + \pmb{b}_k(\pmb{x})\}$ is a convex piecewise affine function. See <span style="font-variant:small-caps;"><b>Theorem 9</b></span> (Equation (19)) for more information.

class GMModel(ro.Model):
    """
    GMModel is a subclas of rsome.ro.Model where the Phi sets and
    the entropic risk measures can be explicitly defined by methods
    defined below.
    """
    
    def in_Phi(self, t, kappa, v, fdev, bdev):
        """
        Define the dispersion characteristic set for the given model
        object, i.e., (t, \kappa, v) \in \Phi.
        
        Parameters
        ----------
        t : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        kappa : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        v : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        fdev : {float, numpy.float64}
            The forward deviation of the random variable.
        bdev : {float, numpy.float64}
            The backward deviation of the random variable.
        
        Notes
        -----
        The dispersion characteristic set Phi is defined in THEOREM 3
        as a second-order cone representable set. 
        """
        
        u = self.dvar()
        
        self.st(t >= 0, kappa >= 0,
                rso.rsocone(u, kappa, 2*t),
                u >= -bdev*v,
                u >= fdev*v)
    def in_Phi_infcon(self, t, kappa, v):
        """
        Define the dispersion characteristic set for the given model
        object, i.e., (t, \kappa, v) \in \Phi.

        Parameters
        ----------
        t : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        kappa : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        v : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        fdev : {float, numpy.float64}
            The forward deviation of the random variable.
        bdev : {float, numpy.float64}
            The backward deviation of the random variable.

        Notes
        -----
        The dispersion characteristic set Phi is defined in THEOREM 3
        as a second-order cone representable set.
        """

        u = self.dvar()

        self.st(t >= 0, kappa >= 0,
                rso.rsocone(v, kappa, 2*t))
    def in_Phi_data(self, t, kappa, v, fdev, bdev, covar):
        """
        Define the dispersion characteristic set for the given model
        object, i.e., (t, \kappa, v) \in \Phi.

        Parameters
        ----------
        t : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        kappa : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        v : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        fdev : {float, numpy.float64}
            The forward deviation of the random variable.
        bdev : {float, numpy.float64}
            The backward deviation of the random variable.

        Notes
        -----
        The dispersion characteristic set Phi is defined in THEOREM 3
        as a second-order cone representable set.
        """

        u = self.dvar()

        self.st(t >= 0, kappa >= 0,
                rso.rsocone(u, kappa, 2*t),
                u >= -bdev * covar @ v,
                u >= fdev * covar @ v)
    
    def in_Phi_Gamma(self, t, kappa, v, sigmas, Gamma):
        """
        Define the joint dispersion characteristic set for the given 
        model object, i.e., (t, \kappa, v) \in \Phi_{\Gamma}.
        
        Parameters
        ----------
        t : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        kappa : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        v : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A one-dimensional array of variables.
        sigmas : dict
            A dictionary as the output of the devs() function. The
            forward deviation is given as the value of 'forward',
            and the backward deviation is given as the value of 
            'backward'. Both deviations are presented as one-
            dimensional arrays.
        Gamma : {int, float, numpy.float64}
            The value of the independence propensity hyperparameter 
            Gamma. It is between 1 and I, where I is the number of
            random variables.
        
        Notes
        -----
        The characteristic set Phi_Gamma is defined in THEOREM 7
        (Equation (13)).
        """
        
        I = v.size
        
        ti = self.dvar(I)
        ki = self.dvar(I)
        
        for i in range(I):
            fdev = sigmas['forward'][i]
            bdev = sigmas['backward'][i]
            self.in_Phi(ti[i], ki[i], v[i], fdev, bdev)
        self.st(ti.sum() <= t, ti >= 0)
        self.st(ki.sum() <= Gamma * kappa)
        self.st(ki >= 0, ki <= kappa)
    def in_Phi_Gamma_data(self, t, kappa, v, sigmas, Gamma):
        """
        Define the joint dispersion characteristic set for the given
        model object, i.e., (t, \kappa, v) \in \Phi_{\Gamma}.

        Parameters
        ----------
        t : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        kappa : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        v : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A one-dimensional array of variables.
        sigmas : dict
            A dictionary as the output of the devs() function. The
            forward deviation is given as the value of 'forward',
            and the backward deviation is given as the value of
            'backward'. Both deviations are presented as one-
            dimensional arrays.
        Gamma : {int, float, numpy.float64}
            The value of the independence propensity hyperparameter
            Gamma. It is between 1 and I, where I is the number of
            random variables.

        Notes
        -----
        The characteristic set Phi_Gamma is defined in THEOREM 7
        (Equation (13)).
        """

        I = v.size

        ti = self.dvar(I)
        ki = self.dvar(I)

        for i in range(I):
            fdev = sigmas['forward'][i]
            bdev = sigmas['backward'][i]
            covar = sigmas['cov'][i]
            self.in_Phi_data(ti[i], ki[i], v[i], fdev, bdev, covar)
        self.st(ti.sum() <= t, ti >= 0)
        self.st(ki.sum() <= Gamma * kappa)
        self.st(ki >= 0, ki <= kappa)
    def in_Phi_Gamma_infcon(self, t, kappa, v, Gamma):
        """
        Define the joint dispersion characteristic set for the given
        model object, i.e., (t, \kappa, v) \in \Phi_{\Gamma}.

        Parameters
        ----------
        t : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        kappa : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A scalar variable.
        v : {rsome.lp.Vars, rsome.lp.VarSub, rsome.lp.Affine}
            A one-dimensional array of variables.
        sigmas : dict
            A dictionary as the output of the devs() function. The
            forward deviation is given as the value of 'forward',
            and the backward deviation is given as the value of
            'backward'. Both deviations are presented as one-
            dimensional arrays.
        Gamma : {int, float, numpy.float64}
            The value of the independence propensity hyperparameter
            Gamma. It is between 1 and I, where I is the number of
            random variables.

        Notes
        -----
        The characteristic set Phi_Gamma is defined in THEOREM 7
        (Equation (13)).
        """

        I = v.size

        ti = self.dvar(I)
        ki = self.dvar(I)

        for i in range(I):
            self.in_Phi_infcon(ti[i], ki[i], v[i])
        self.st(ti.sum() <= t, ti >= 0)
        self.st(ki.sum() <= Gamma * kappa)
        self.st(ki >= 0, ki <= kappa)
    def min_ce_infty(self, pieces, z, mu, sigmas, Gamma):
        """
        Define the objective of the model as the worst-case expectation
        (the entropic risk with \kappa \rightarrow \infty) over the 
        moment-dispersion ambiguity set.
        
        Parameters
        ----------
        pieces : Iterables
            The piecewise affine function expressed as a collection of
            affine functions.
        z : rsome.lp.Vars
            Random variables involved in the model.
        mu : numpy.ndarray
            A one-dimensional array representing the means of random
            variables.
        sigmas : dict
            A dictionary as the output of the devs() function. The
            forward deviation is given as the value of 'forward',
            and the backward deviation is given as the value of 
            'backward'. Both deviations are presented as one-
            dimensional arrays.
        Gamma : {int, float, numpy.float64}
            The value of the independence propensity hyperparameter 
            Gamma. It is between 1 and I, where I is the number of
            random variables.
        
        Notes
        -----
        The equivalent formulation of the entropic risk assessing 
        function is presented in THEOREM 9 (Equation (19)).
        """
        
        K = len(pieces)
        Nz = z.size
        
        t = self.dvar(2)
        q = self.dvar()
        u = self.dvar(Nz)
        w = self.dvar(K)
        r = self.dvar(K)
        eta = self.dvar(K)
        v = self.dvar((K, Nz))
        vbar = self.dvar(K)
        
        self.min(t.sum())
        
        self.st(eta.sum() <= q, q >= 0)
        self.st(rso.pexp(w + r - t[0], q) <= eta)
        self.st(v@mu + vbar <= w)
        for k in range(K):
            self.in_Phi_Gamma(r[k], q, v[k] - u, sigmas, Gamma)
            self.st((pieces[k] - v[k]@z - vbar[k] <= t[1]).forall())
    def min_ce_infty_data(self, pieces, z, mu, sigmas, Gamma, fset):
        """
        Define the objective of the model as the worst-case expectation
        (the entropic risk with \kappa \rightarrow \infty) over the
        moment-dispersion ambiguity set.

        Parameters
        ----------
        pieces : Iterables
            The piecewise affine function expressed as a collection of
            affine functions.
        z : rsome.lp.Vars
            Random variables involved in the model.
        mu : numpy.ndarray
            A one-dimensional array representing the means of random
            variables.
        sigmas : dict
            A dictionary as the output of the devs() function. The
            forward deviation is given as the value of 'forward',
            and the backward deviation is given as the value of
            'backward'. Both deviations are presented as one-
            dimensional arrays.
        Gamma : {int, float, numpy.float64}
            The value of the independence propensity hyperparameter
            Gamma. It is between 1 and I, where I is the number of
            random variables.

        Notes
        -----
        The equivalent formulation of the entropic risk assessing
        function is presented in THEOREM 9 (Equation (19)).
        """

        K = len(pieces)
        Nz = z.size

        t = self.dvar(2)
        q = self.dvar()
        u = self.dvar(Nz)
        w = self.dvar(K)
        r = self.dvar(K)
        eta = self.dvar(K)
        v = self.dvar((K, Nz))
        vbar = self.dvar(K)

        self.min(t.sum())

        self.st(eta.sum() <= q, q >= 0)
        self.st(rso.pexp(w + r - t[0], q) <= eta)
        self.st((v@mu + vbar <= w).forall(fset))
        for k in range(K):
            self.in_Phi_Gamma_data(r[k], q, v[k] - u, sigmas, Gamma)
            self.st((pieces[k] - v[k]@z - vbar[k] <= t[1]).forall())

    def min_ce_infty_infcon(self, pieces, z, mu, Gamma, fset):
        """
        Define the objective of the model as the worst-case expectation
        (the entropic risk with \kappa \rightarrow \infty) over the
        moment-dispersion ambiguity set.

        Parameters
        ----------
        pieces : Iterables
            The piecewise affine function expressed as a collection of
            affine functions.
        z : rsome.lp.Vars
            Random variables involved in the model.
        mu : numpy.ndarray
            A one-dimensional array representing the means of random
            variables.
        sigmas : dict
            A dictionary as the output of the devs() function. The
            forward deviation is given as the value of 'forward',
            and the backward deviation is given as the value of
            'backward'. Both deviations are presented as one-
            dimensional arrays.
        Gamma : {int, float, numpy.float64}
            The value of the independence propensity hyperparameter
            Gamma. It is between 1 and I, where I is the number of
            random variables.

        Notes
        -----
        The equivalent formulation of the entropic risk assessing
        function is presented in THEOREM 9 (Equation (19)).
        """

        K = len(pieces)
        Nz = z.size

        t = self.dvar(2)
        q = self.dvar()
        u = self.dvar(Nz)
        w = self.dvar(K)
        r = self.dvar(K)
        eta = self.dvar(K)
        v = self.dvar((K, Nz))
        vbar = self.dvar(K)

        self.min(t.sum())

        self.st(eta.sum() <= q, q >= 0)
        self.st(rso.pexp(w + r - t[0], q) <= eta)
        self.st((v@mu + vbar <= w).forall(fset))
        for k in range(K):
            self.in_Phi_Gamma_infcon(r[k], q, v[k] - u, Gamma)
            self.st((pieces[k] - v[k]@z - vbar[k] <= t[1]).forall())
# #### Function for solving the RO-$\Gamma$ model
# 
# The `ro_gamma_cvar_data()` function defined below is used to solve the RO-$\Gamma$ model. Note that the objective as the entropic risk assessing function $\rho(\infty, \pmb{x})$ over the moment-dispersion ambiguity set can be easily defined using the `min_ce_infty()` method of the `GMModel` object.

def ro_gamma_cvar(mu, sigmas, Gamma, epsilon, label='default'):
    """
    The function ro_gamma_cvar solves a robust model with a moment-
    dispersion ambiguity set that aims to minimize the Conditional 
    Value-at-Risk (CVaR) of the portfolio return.  
    
    Parameters
    ----------
    sigmas : dict
        A dictionary as the output of the devs() function. The
        forward deviation is given as the value of 'forward',
        and the backward deviation is given as the value of 
        'backward'. Both deviations are presented as one-
        dimensional arrays.
    Gamma : {int, float, numpy.float64}
        The value of the independence propensity hyperparameter 
        Gamma. It is between 1 and I, where I is the number of
        random variables.
    epsilon : float
        The confidence level expressed as 1 - epsilon.
    
    Returns
    -------
    xs : numpy.ndarray
        A one-dimensional array representing the optimal portfolio
        decision.
    objval : float, numpy.float64
        The optimal objective value of the SAA model.
    """
    
    n = len(mu)
    
    model = GMModel()
    
    z = model.rvar(n)
    
    x = model.dvar(n)
    v = model.dvar()
    
    pieces = [(1/epsilon)*(-x@z - (1-epsilon)*v), v]
    model.min_ce_infty(pieces, z, mu, sigmas, Gamma)
    model.st(x >= 0, x.sum() == 1)
    st = time.perf_counter()
    model.solve(msk, display=True)
    ed = time.process_time()
    if label=='default':
        return x.get(), model.get()
    else:
        return x.get(), model.get(), ed-st


def ro_gamma_cvar_data(mu, sigmas, Gamma, epsilon, gamma):
    """
    The function ro_gamma_cvar solves a robust model with a moment-
    dispersion ambiguity set that aims to minimize the Conditional
    Value-at-Risk (CVaR) of the portfolio return under real data.

    Parameters
    ----------
    sigmas : dict
        A dictionary as the output of the devs() function. The
        forward deviation is given as the value of 'forward',
        and the backward deviation is given as the value of
        'backward'. Both deviations are presented as one-
        dimensional arrays.
    Gamma : {int, float, numpy.float64}
        The value of the independence propensity hyperparameter
        Gamma. It is between 1 and I, where I is the number of
        random variables.
    epsilon : float
        The confidence level expressed as 1 - epsilon.

    Returns
    -------
    xs : numpy.ndarray
        A one-dimensional array representing the optimal portfolio
        decision.
    objval : float, numpy.float64
        The optimal objective value of the SAA model.
    """

    n = len(mu)

    model = GMModel()

    z = model.rvar(n)
    u = model.rvar(n)
    fset = (rso.norm(u-mu, 1) <= gamma)

    x = model.dvar(n)
    v = model.dvar()

    pieces = [(1 / epsilon) * (-x @ z - (1 - epsilon) * v), v]
    model.min_ce_infty_data(pieces, z, u, sigmas, Gamma, fset)
    model.st(x >= 0, x.sum() == 1)
    model.solve(msk, display=True)
    return x.get(), model.get()

def ro_gamma_cvar_infcon(mu, uup, ulow, Gamma, epsilon):
    n = len(mu)

    model = GMModel()

    z = model.rvar(n)
    u = model.rvar(n)
    fset = (u <= uup, u>=ulow)

    x = model.dvar(n)
    v = model.dvar()

    pieces = [(1 / epsilon) * (-x @ z - (1 - epsilon) * v), v]
    model.min_ce_infty_infcon(pieces, z, u, Gamma, fset)
    model.st(x >= 0, x.sum() == 1)
    st = time.perf_counter()
    model.solve(msk, display=True)
    ed = time.process_time()
    return x.get(), model.get(), ed-st

# ### Out-sample Test
# 
# The `out_sample_test()` function defined below is used to conduct out-sample tests for assessing the performance of a given portfolio decision. 

# In[10]:


def out_sample_test(x_sol, zhat, epsilon):
    """
    The out_sample_test function returns the Conditional Value-at-
    Risk of a given portfolio decision, based on a out-of-sample
    dataset.
    
    Parameters
    ----------
    x_sol : np.ndarray
        A one-dimensional array representing the portfolio decision.
    zhat : np.ndarray
        A two-dimensional array representing the out-of-sample data
        of stock returns. The shape of the array is (s, n), where s
        is the sample size, and n is the number of stocks.
    epsilon : float
        The confidence level expressed as 1 - epsilon.
        
    Return
    ------
    The CVaR of returns of the given portfolio decision.
    """
    
    model = ro.Model()
    
    s = zhat.shape[0]
    v = model.dvar()
    y = model.dvar(s)
    
    model.min(v + (1/s) * y.sum())
    model.st(y >= 0, y >= (1/epsilon) * (-zhat@x_sol - v))
    
    model.solve(msk, display=False)
    
    return model.get()


# ## Numerical Experiments
# 
# ### Parameters
# 
# Parameters of the numerical experiments are defined below.

# In[11]:


# I = J = 20
# epsilon = 0.05
#
# in_size = 100
# saa_size = 1000
# out_size = 10000
#
# repeats = 50


# ### Data Generation Functions
# 
# Functions below are used to randomly generate samples for portfolio returns. 

# In[12]:


def norm_return(mu, Q, size, seed=None):
    """
    The function norm_return generates random samples of stock
    returns, denoted by z, following the equation below.
    
    z = mu + xi@Q.T
    
    where mu is the constant term, and Q is a matrix of linear
    coefficients. The vector xi follows a Gaussian (normal)
    distribution.
    
    Parameter
    ---------
    mu : np.ndarray
        A one-dimensional array representing the constant term.
    Q : np.ndarray
        A two-dimensional array as the linear coefficients.
    size : int
        The sample size of the generated dataset.
    seed : int
        The seed for generating random numbers.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    J = Q.shape[1]
    xi = np.random.normal(size=(size, J))

    zhat = mu + xi@Q.T
    
    return pd.DataFrame(zhat)


# In[13]:
def uniform_return(mu, Q, size, seed=None):
    if seed is not None:
        np.random.seed(seed)

    J = Q.shape[1]
    xi = np.random.uniform(-1,1,size=(size, J))

    zhat = mu + xi @ Q.T

    return pd.DataFrame(zhat)

def beta_return(mu, Q, size, seed=None):
    """
    The function norm_return generates random samples of stock
    returns, denoted by z, following the equation below.
    
    z = mu + xi@Q.T
    
    where mu is the constant term, and Q is a matrix of linear
    coefficients. The vector xi follows a Beta distribution with
    alpha = 0.05*j, and beta = 0.10*j.
    
    Parameter
    ---------
    mu : np.ndarray
        A one-dimensional array representing the constant term.
    Q : np.ndarray
        A two-dimensional array as the linear coefficients.
    size : int
        The sample size of the generated dataset.
    seed : int
        The seed for generating random numbers.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    J = Q.shape[1]
    xi = np.array([np.random.beta(0.05*j, 0.1*j, size=size)
                   for j in range(1, J+1)]).T

    zhat = mu + xi@Q.T
    
    return pd.DataFrame(zhat)


# ### Function for One Numerical Test
# 
# The function `one_test()` defined below is used to solve all models mentioned above and evaluate their performance using out-of-sample tests. The results are saved in a data frame. Users could use the function arguments to specify:
# 
# 1. How the stock returns are correlated (`corr='default'`, `'weak'`, or `'neg'`)
# 2. The distribution of the random factor $\xi$ (`factor='gaussian'` or `'beta'`)

# In[14]:


def one_test(r, corr='default', factor='gaussian'):
    """
    The function one_test conducts one test for solving the following
    portfolio optimization models:
    
    1. RO-CM
    2. RO-MM
    3. SAA
    4. Clairvoyance
    5. RO-Gamma with Gamma values ranging from 1 to 20,
    
    and assessing their performance using out-of-sample tests.
    
    Parameters
    ----------
    r : int
        The seed for generating random parameters and in-sample/out-of-
        sample datasets.
    corr : {'default', 'weak', 'neg'}
        The option for generating the linear coefficient matrix Q. If
        corr = 'default', Q is randomly generated from the independent
        uniform distribution on the interval [0, 1]. If corr = 'weak',
        the diagonal components of the matrix Q is set to ones, while
        the remaining components are randomly generated from a uniform
        distribution over the range [0, 0.4]. If corr = 'neg', the
        diagonal components of the matrix Q is set to ones, while
        the remaining components are randomly generated from a uniform
        distribution over the range [-0.2, 0].
    factor : {'gaussian', 'beta'}
        The option for generating the random vector xi. If factor is
        'gaussian', then xi follows a standard normal distribution. If
        factor is 'beta', then each xi follows a beta distribution. 
    
    Returns
    -------
    cvar_out : dict
        A dictionary where each element represents the out-of-sample CVaR
        of every model. 
    """
    
    np.random.seed(2000+r)
    
    mu0 = np.zeros(I)
    if corr.lower() == 'default':
        Q = np.random.rand(I, J)
    elif corr.lower() == 'weak':
        Q = np.random.rand(I, J) * 0.4
        dim = np.minimum(I, J)
        Q[range(dim), range(dim)] = 1
    elif corr.lower() == 'neg':
        Q = - np.random.rand(I, J) * 0.2
        dim = np.minimum(I, J)
        Q[range(dim), range(dim)] = 1
    else:
        raise ValueError('Unknonw option for correlation type.')
    
    if factor.lower() == 'gaussian':
        samp_func = norm_return
    elif factor.lower() == 'beta':
        samp_func = beta_return
    else:
        raise ValueError('Unknown option for factor.')
        
    z_in = samp_func(mu0, Q, in_size).values
    z_out = samp_func(mu0, Q, out_size).values
    
    z_saa = np.array([np.random.choice(z_in[:, i], size=saa_size) 
                      for i in range(z_in.shape[1])]).T
    mu = z_in.mean(axis=0)
    std = z_in.std(axis=0)
    
    
    x_cm, obj_cm = ro_cm_cvar(mu, std, epsilon)
    x_mm, obj_mm = ro_mm_cvar(mu, std, epsilon)
    x_sa, obj_sa = saa_cvar(z_saa, epsilon)
    
    cvar_out = {}
    cvar_out['RO-CM'] = out_sample_test(x_cm, z_out, epsilon)
    cvar_out['RO-MM'] = out_sample_test(x_mm, z_out, epsilon)
    cvar_out['SAA'] = out_sample_test(x_sa, z_out, epsilon)
    x_cla, cvar_out['Clairvoyance'] = saa_cvar(z_out, epsilon)
    
    mu_out = z_out.mean(axis=0)
    mean_out = {}
    mean_out['RO-CM'] = x_cm @ mu_out
    mean_out['RO-MM'] = x_mm @ mu_out
    mean_out['SAA'] = x_sa @ mu_out
    mean_out['Clairvoyance'] = x_cla @ mu_out
    
    delta = devs(z_in - mu)
    cvar_fb_gamma = []
    for Gamma in range(1, 21): 
        x_fb, obj_fb = ro_gamma_cvar(mu, delta, Gamma, epsilon)
        cvar_out[f'RO-Gamma={Gamma}'] = out_sample_test(x_fb, z_out, epsilon)
        mean_out[f'RO-Gamma={Gamma}'] = x_fb @ mu_out
        
    for k, theta in enumerate(10**np.linspace(-5, -1, 15)):
        x_was, obj_was = wass_cvar(z_saa, epsilon, theta)
        cvar_out[f'Wass-{k}'] = out_sample_test(x_was, z_out, epsilon)
        mean_out[f'Wass-{k}'] = x_was @ mu_out
        
    return cvar_out, mean_out

def one_test_infty(r, epsilon):
    """
    The function one_test conducts one test for solving the following
    portfolio optimization models:

    1. RO-CM
    2. RO-MM
    3. SAA
    4. Clairvoyance
    5. RO-Gamma with Gamma values ranging from 1 to 20,

    and assessing their performance using out-of-sample tests.

    Parameters
    ----------
    r : int
        The seed for generating random parameters and in-sample/out-of-
        sample datasets.
    corr : {'default', 'weak', 'neg'}
        The option for generating the linear coefficient matrix Q. If
        corr = 'default', Q is randomly generated from the independent
        uniform distribution on the interval [0, 1]. If corr = 'weak',
        the diagonal components of the matrix Q is set to ones, while
        the remaining components are randomly generated from a uniform
        distribution over the range [0, 0.4]. If corr = 'neg', the
        diagonal components of the matrix Q is set to ones, while
        the remaining components are randomly generated from a uniform
        distribution over the range [-0.2, 0].
    factor : {'gaussian', 'beta'}
        The option for generating the random vector xi. If factor is
        'gaussian', then xi follows a standard normal distribution. If
        factor is 'beta', then each xi follows a beta distribution.

    Returns
    -------
    cvar_out : dict
        A dictionary where each element represents the out-of-sample CVaR
        of every model.
    """
    if epsilon == 0.01:
        x_infty = np.array([0.120905241,0.065463769,0.046983279,0.037743033,0.032198886,0.028502788,0.025862718,
                            0.023882665,0.022342624,0.021110592,0.020102565,0.019262543,0.018551755,0.017942508,
                            0.017414494,0.016952481,0.016544823,0.016182461,0.015858242,0.015566445,0.015302438,
                            0.015062431,0.014843295,0.01464242,0.014457615,0.014287026,0.014129073,0.013982402,
                            0.013845847,0.013718395,0.013599166,0.013487389,0.013382387,0.01328356,0.013190381,
                            0.013102379,0.013019134,0.01294027,0.01286545,0.012794371,0.012726759,0.012662367,
                            0.01260097,0.012542364,0.012486363,0.012432796,0.012381509,0.012332359,0.012285215,0.012239956])
    elif epsilon == 0.02:
        x_infty = np.array([0.108702206,0.059965582,0.04372004,0.035597269,0.030723607,0.027474498,0.025153707,
                            0.023413113,0.022059318,0.020976282,0.020090161,0.019351727,0.018726899,0.018191332,
                            0.017727173,0.017321035,0.016962677,0.016644137,0.016359128,0.016102619,0.01587054,
                            0.015659559,0.015466924,0.015290342,0.015127887,0.014977928,0.014839077,0.014710144,
                            0.014590103,0.014478065,0.014373255,0.014274996,0.014182692,0.014095817,0.014013907,
                            0.013936547,0.013863369,0.013794042,0.013728271,0.013665788,0.013606353,0.013549748,
                            0.013495776,0.013444258,0.013395029,0.01334794,0.013302856,0.013259649,0.013218207,0.013178422])
    elif epsilon == 0.05:
        x_infty = np.array([0.082767039,0.048280257,0.036784663,0.031036866,0.027588188,0.025289069,0.023646842,
                            0.022415171,0.021457205,0.020690832,0.020063799,0.019541272,0.019099134,0.018720159,
                            0.018391713,0.018104323,0.017850744,0.01762534,0.017423663,0.017242154,0.017077931,
                            0.016928637,0.016792326,0.016667374,0.016552418,0.016446305,0.016348052,0.016256817,
                            0.016171874,0.016092594,0.016018429,0.015948899,0.015883583,0.01582211,0.015764149,
                            0.015709408,0.015657626,0.015608569,0.015562028,0.015517815,0.015475758,0.015435703,
                            0.015397512,0.015361056,0.015326221,0.015292901,0.015260998,0.015230425,0.015201099,0.015172947])
    elif epsilon == 0.08:
        x_infty = np.array([0.058683342,0.03742913,0.030344392,0.026802023,0.024676602,0.023259655,0.022247549,
                            0.02148847,0.020898076,0.02042576,0.02003932,0.019717286,0.019444796,0.019211233,
                            0.019008812,0.018831694,0.018675413,0.018536497,0.018412203,0.018300339,0.018199128,
                            0.018107119,0.01802311,0.017946102,0.017875254,0.017809857,0.017749304,0.017693075,
                            0.017640725,0.017591865,0.017546157,0.017503306,0.017463051,0.017425165,0.017389444,
                            0.017355707,0.017323794,0.01729356,0.017264877,0.017237628,0.017211708,0.017187023,
                            0.017163485,0.017141018,0.017119549,0.017099014,0.017079352,0.01706051,0.017042436,0.017025086])
    elif epsilon == 0.1:
        x_infty = np.array([0.041914122,0.029873606,0.0258601,0.023853348,0.022649296,0.021846595,0.021273237,
                            0.020843219,0.02050876,0.020241193,0.020022275,0.019839843,0.019685477,0.019553164,
                            0.019438492,0.019338154,0.019249621,0.019170925,0.019100513,0.019037141,0.018979806,
                            0.018927682,0.018880091,0.018836466,0.018796331,0.018759283,0.01872498,0.018693127,
                            0.01866347,0.018635791,0.018609897,0.018585622,0.018562818,0.018541356,0.018521119,
                            0.018502007,0.018483929,0.018466801,0.018450552,0.018435116,0.018420432,0.018406448,
                            0.018393114,0.018380386,0.018368224,0.018356591,0.018345452,0.018334778,0.018324539,0.01831471])
    np.random.seed(2000 + r)

    mu0 = np.array([0.004 * (1+i) for i in range(I)])
    sigma = np.array([0.001 * np.sqrt(2 * (i+1) * I * (I + 1)) for i in range(I)])
    Q = np.diag(sigma)
    samp_func = uniform_return
    uup = mu0 + sigma
    ulow = mu0 - sigma


    # z_in = samp_func(mu0, Q, in_size).values
    z_out = samp_func(mu0, Q, out_size).values
    mu_out = z_out.mean(axis=0)

    cvar_out = {}
    mean_out = {}
    obj_in = {}
    time_in = {}

    x_cla, cvar_out['Clairvoyance'] = saa_cvar(z_out, epsilon)
    mean_out['Clairvoyance'] = x_cla @ mu_out
    # delta = devs(z_in - mu0)
    for Gamma in range(I, I+1):
        x_fb, obj_fb, time_fb = ro_gamma_cvar_infcon(mu0, uup, ulow, Gamma, epsilon)

        cvar_out[f'RO-Gamma={Gamma}'] = out_sample_test(x_fb, z_out, epsilon)
        mean_out[f'RO-Gamma={Gamma}'] = x_fb @ mu_out
        obj_in[f'RO-Gamma={Gamma}'] = obj_fb
        time_in[f'RO-Gamma={Gamma}'] = time_fb
    cvar_out[f'RO-infty'] = out_sample_test(x_infty, z_out, epsilon)
    mean_out[f'RO-infty'] = x_infty @ mu_out


    return cvar_out, mean_out, obj_in, time_in
# ### Function for Visualizing Out-Sample Performance
# 
# The function `plot_results()` defined below is used to visualize the results generated from all repeated tests.

# In[15]:


def plot_results(cvar_results):
    """
    The function plot_results visualizes the out-of-sample CVaR of
    each model. See Figure 4, Figure 5, and Figure 6 in the paper.
    
    Parameters
    ----------
    out_results : pandas.DataFrame
        A data frame that summarizes the out-of-sample CVaR of each
        model. Every row is an independent experiement.
    """
    
    cvar_avg = cvar_results.mean()
    
    theta_values = (10**np.linspace(-3, -1, 15)).round(5)
    #min_wass_index = np.argmin(cvar_avg['Wass-0':].values)
    
    
    plt.figure(figsize=(6, 4), dpi=100)
    plt.plot(np.arange(1, 21), cvar_avg['RO-Gamma=1':'RO-Gamma=20'], 
             linewidth=1, marker='o', color='k', markerfacecolor='none',
             label=r'RO-$\Gamma$')
    plt.hlines(cvar_avg['SAA'], 1, 20, 
               linewidth=1, linestyle='--', color='r', label='SAA')
    plt.hlines(cvar_avg['RO-CM'], 1, 20, 
               linewidth=1, linestyle='-.', color='y', label='RO-CM')
    plt.hlines(cvar_avg['RO-MM'], 1, 20, 
               linewidth=1, linestyle=':', color='m', label='RO-MM')
    #plt.hlines(cvar_avg[f'Wass-theta={theta_values[min_wass_index]:.5f}'], 1, 20, 
    #           linewidth=1, color='b', label='Minimum Wasserstein')
    plt.hlines(cvar_avg['Clairvoyance'], 1, 20, 
               linewidth=1, color='g', label='Clairvoyance')
    plt.legend(prop={'family': 'times', 'math_fontfamily': 'cm'}, 
               bbox_to_anchor=(1.01, 1.02))
    plt.xlabel(r'$\Gamma$', fontsize=12, math_fontfamily='cm')
    plt.ylabel('Average CVaR', fontsize=12, fontfamily='times')
    # plt.show()

    plt.figure(figsize=(4.5, 4), dpi=100)
    cvar_avg = cvar_results.mean()
    min_ro_index = np.argmin(cvar_avg['RO-Gamma=1':'RO-Gamma=20'].values)
    labels = ['SAA', 'RO-CM', 'RO-MM', 
              #r'DRO Wass($\theta$' + f'={theta_values[min_wass_index]})',
              r'RO-$\Gamma$($\Gamma$'+f'={min_ro_index+1})']
    plt.boxplot([cvar_results['SAA'], cvar_results['RO-CM'], cvar_results['RO-MM'], 
                 #cvar_results.loc[:, f'Wass-theta={theta_values[min_wass_index]:.5f}'],
                 cvar_results.loc[:, f'RO-Gamma={min_ro_index+1}']],
                labels=labels, vert=False)
    plt.xlabel('CVaR Values', fontsize=12, fontfamily='times')
    # plt.show()

def plot_mean_var(mean_results, cvar_results):
    
    mean_avg = mean_results.mean()
    cvar_avg = cvar_results.mean()
    
    plt.figure(figsize=(6, 4), dpi=100)
    plt.plot(mean_avg['RO-Gamma=1':'RO-Gamma=20'], 
             cvar_avg['RO-Gamma=1':'RO-Gamma=20'], marker='s', label=f'RO-$\Gamma$')
    plt.plot(mean_avg['Wass-0':], 
             cvar_avg['Wass-0':], marker='o', label=f'Wasserstein')
    
    plt.legend(prop={'family': 'times', 'math_fontfamily': 'cm'}, 
               bbox_to_anchor=(1.01, 1.02))
    plt.xlabel('Average Return', fontsize=12, math_fontfamily='cm')
    plt.ylabel('CVaR Values', fontsize=12, fontfamily='times')
    # plt.show()


# ### Figures in the Paper
#
I = J = 20
epsilon = 0.05

in_size = 5
saa_size = 10
out_size = 10000

repeats = 50
# #### Figure 1. Out-of-sample CVaR comparison of the four models under random $Q$ and Gaussian factors

cvar_results = []
mean_results = []
for r in range(repeats):
    if (r+1)%10 == 0:
        print(f'Repeat {r+1} ...')
    cvar_out, mean_out = one_test(r)
    cvar_results.append(cvar_out)
    mean_results.append(mean_out)

cvar_results = pd.DataFrame(cvar_results)
mean_results = pd.DataFrame(mean_results)
resultPath = 'portfolio_gaussian_v3.xlsx'
writer = pd.ExcelWriter(resultPath)
cvar_results.to_excel(writer,sheet_name = "cvar",index = False,na_rep = 0,inf_rep = 0)
mean_results.to_excel(writer,sheet_name = "mean",index = False,na_rep = 0,inf_rep = 0)
writer.close()

# # #### Figure 2. Out-of-sample CVaR comparison of the models under random $Q$ and Beta factors
cvar_results = []
mean_results = []
for r in range(repeats):
    if (r+1)%10 == 0:
        print(f'Repeat {r+1} ...')
    cvar_out, mean_out = one_test(r, factor='beta')
    cvar_results.append(cvar_out)
    mean_results.append(mean_out)

cvar_results = pd.DataFrame(cvar_results)
mean_results = pd.DataFrame(mean_results)
plot_results(cvar_results)
resultPath = 'portfolio_beta.xlsx'
writer = pd.ExcelWriter(resultPath)
cvar_results.to_excel(writer,sheet_name = "cvar",index = False,na_rep = 0,inf_rep = 0)
mean_results.to_excel(writer,sheet_name = "mean",index = False,na_rep = 0,inf_rep = 0)
writer.close()

# # #### Figure 3. Out-of-sample CVaR comparison of the four models under random $Q$ (weak correlations) and Gaussian factors
cvar_results = []
mean_results = []
for r in range(repeats):
    if (r+1)%10 == 0:
        print(f'Repeat {r+1} ...')
    cvar_out, mean_out = one_test(r, corr='weak')
    cvar_results.append(cvar_out)
    mean_results.append(mean_out)

cvar_results = pd.DataFrame(cvar_results)
mean_results = pd.DataFrame(mean_results)
resultPath = 'portfolio_week.xlsx'
writer = pd.ExcelWriter(resultPath)
cvar_results.to_excel(writer,sheet_name = "cvar",index = False,na_rep = 0,inf_rep = 0)
mean_results.to_excel(writer,sheet_name = "mean",index = False,na_rep = 0,inf_rep = 0)
writer.close()

# #### Figure 7 ############################################

# for epsilon in [0.01,0.02,0.05,0.08,0.1]:
#     cvar_results = []
#     mean_results = []
#     obj_results = []
#     time_results = []
#     for r in range(repeats):
#         if (r+1)%10 == 0:
#             print(f'Repeat {r+1} ...')
#         cvar_out, mean_out, obj_in, time_in = one_test_infty(r,epsilon)
#         cvar_results.append(cvar_out)
#         mean_results.append(mean_out)
#         obj_results.append(obj_in)
#         time_results.append(time_in)
#
#     cvar_results = pd.DataFrame(cvar_results)
#     mean_results = pd.DataFrame(mean_results)
#     obj_results = pd.DataFrame(obj_results)
#     time_results = pd.DataFrame(time_results)
#     resultPath = f'portfolio_infty3_{epsilon}.xlsx'
#     writer = pd.ExcelWriter(resultPath)
#     cvar_results.to_excel(writer,sheet_name = "cvar",index = False,na_rep = 0,inf_rep = 0)
#     mean_results.to_excel(writer,sheet_name = "mean",index = False,na_rep = 0,inf_rep = 0)
#     obj_results.to_excel(writer,sheet_name = "obj",index = False,na_rep = 0,inf_rep = 0)
#     time_results.to_excel(writer,sheet_name = "time",index = False,na_rep = 0,inf_rep = 0)
#     writer.close()