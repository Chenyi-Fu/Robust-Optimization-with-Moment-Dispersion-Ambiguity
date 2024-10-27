#!/usr/bin/env python
# coding: utf-8
import numpy
# # Portfolio Optimization
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

from scipy.optimize import golden

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Description
#
# In this numerical study, we consider a portfolio optimization problem that aims to minimize
# the conditional value-at-risk (CVaR) of the portfolio return for $N$ distinct financial assets. These assets are characterized by their uncertain returns, denoted as  $\tilde{\pmb{z}} \sim \mathbb{P}$, with the distribution $\mathbb{P}\in\mathcal{F}$ defined within an ambiguity set $\mathcal{F} \subseteq \mathcal{P}_0(\mathbb{R}^N)$. Given a specific CVaR risk preference parameterized by $\epsilon\in(0, 1)$, the robust portfolio optimization problem is formulated as
#
# $$
# \min\limits_{v\in\mathbb{R}, \pmb{x}\in\mathcal{X}}\left\{v + \frac{1}{\epsilon}\sup\limits_{\mathbb{P}\in\mathcal{F}}\mathbb{E}_{\mathbb{P}}\left[\max\{-\pmb{x}^{\top}\tilde{\pmb{z}} - v, 0\}\right]\right\} = \min\limits_{v\in\mathbb{R}, \pmb{x}\in\mathcal{X}}\left\{\frac{1}{\epsilon}\sup\limits_{\mathbb{P}\in\mathcal{F}}\mathbb{CE}_{\mathbb{P}}^{\infty}\left[\max\{-\pmb{x}^{\top}\tilde{\pmb{z}} - (1-\epsilon)v, \epsilon v\}\right]\right\}.
# $$
#
# The portfolio decision is defined by $\pmb{x}\in\mathcal{X}$, where $\mathcal{X}\in\left\{\pmb{x}\in\mathbb{R}_+^N\left|~\pmb{1}^{\top}\pmb{x} = 1\right.\right\}$.
#
# In the following numerical experiments, the portfolio return is constructed as random variable $\tilde{\pmb{z}}=\pmb{Q}\tilde{\pmb{xi}}$, where matrix $\pmb{Q}\in\mathbb{R}^{N\times N}$, and $\tilde{\pmb{\xi}}$ represents the hidden random factors that influence the portfolio returns.

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

# In[2]:


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
    fset.exptset(E(z) == mu, E(u) == std ** 2)

    model.minsup(v + E((1 / epsilon) * rso.maxof(-x @ z - v, 0)), fset)
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

# In[3]:

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

    c = ((1 - epsilon) / epsilon) ** 0.5
    model.min(-x @ mu + c * rso.norm(std * x))
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

    model.min(v + (1 / s) * y.sum())
    model.st(y >= 0, y >= (1 / epsilon) * (-zhat @ x - v))
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
    fset.probset(pr == 1 / s)

    x = model.dvar(n)
    v = model.dvar()

    model.minsup(v + (1 / epsilon) * E(rso.maxof(-z @ x - v, 0)), fset)
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
    a2, b2 = -1 / epsilon, 1 - 1 / epsilon
    model.min(eta * theta + (1 / s) * c.sum())
    model.st(b1 * v + a1 * (zhat @ x) <= c,
             b2 * v + a2 * (zhat @ x) <= c)
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
            return - (2 * np.log(np.exp(zvec * v).mean())) ** 0.5 / v

    def neg(v, zvec):

        if v == 0:
            return zvec.std(ddof=0)
        else:
            return - (2 * np.log(np.exp(-zvec * v).mean())) ** 0.5 / v



    sigma = np.cov(zhat.T)
    v, P = np.linalg.eigh(sigma)
    v = (np.abs(v) > 1e-14) * v
    v_half_inv = np.array([np.abs(i) ** (-0.5) if np.abs(i) > 1e-14 else 0 for i in v])
    v_half = np.array([np.abs(i) ** (0.5) if np.abs(i) > 1e-14 else 0 for i in v])
    covar_half_inv = P @ np.diag(v_half_inv) @ P.T
    covar_half = P @ np.diag(v_half) @ P.T
    zeta = covar_half_inv @ zhat.T
    zeta = zeta.T

    fdev = [-golden(pos, args=(zeta[:, j],), brack=(0.0, 50.0),
                    full_output=True)[1]
            for j in range(zhat.shape[1])]
    bdev = [-golden(neg, args=(zeta[:, j],), brack=(0.0, 50.0),
                    full_output=True)[1]
            for j in range(zhat.shape[1])]

    # print([-golden(pos, args=(zeta[:, j],), brack=(0.0, 50.0),
    #                 full_output=True)[0]
    #         for j in range(zhat.shape[1])])
    # print([-golden(neg, args=(zeta[:, j],), brack=(0.0, 50.0),
    #                 full_output=True)[0]
    #         for j in range(zhat.shape[1])])
    # print()
    return dict(forward=np.array(fdev),
                backward=np.array(bdev),
                cov=covar_half,
                zeta=zeta)


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
        #
        # self.st(t >= 0, kappa >= 0,
        #         rso.rsocone(u, kappa, 2 * t),
        #         u >= -bdev * covar @ v,
        #         u >= fdev * covar @ v)

        self.st(t >= 0, kappa >= 0,
                rso.rsocone(u, kappa, 2 * t),
                u >= covar @ v,
                u >= -covar @ v)

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
            sigma_pow = sigmas['cov'][i]
            self.in_Phi_data(ti[i], ki[i], v, fdev, bdev, sigma_pow)
        self.st(ti.sum() <= t, ti >= 0)
        self.st(ki.sum() <= Gamma * kappa)
        self.st(ki >= 0, ki <= kappa)

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
        mu : rsome.lp.Vars
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
        self.st((v @ mu + vbar <= w).forall(fset))
        for k in range(K):
            self.in_Phi_Gamma_data(r[k], q, v[k] - u, sigmas, Gamma)
            self.st((pieces[k] - v[k] @ z - vbar[k] <= t[1]).forall(fset))

    # def min_ce_infty_data2(self, x, vhat, mu, sigmas, Gamma, gamma, epsilon):
    #     """
    #     Define the objective of the model as the worst-case expectation
    #     (the entropic risk with \kappa \rightarrow \infty) over the
    #     moment-dispersion ambiguity set.
    #
    #     Parameters
    #     ----------
    #     pieces : Iterables
    #         The piecewise affine function expressed as a collection of
    #         affine functions.
    #     z : rsome.lp.Vars
    #         Random variables involved in the model.
    #     mu : rsome.lp.Vars
    #         A one-dimensional array representing the means of random
    #         variables.
    #     sigmas : dict
    #         A dictionary as the output of the devs() function. The
    #         forward deviation is given as the value of 'forward',
    #         and the backward deviation is given as the value of
    #         'backward'. Both deviations are presented as one-
    #         dimensional arrays.
    #     Gamma : {int, float, numpy.float64}
    #         The value of the independence propensity hyperparameter
    #         Gamma. It is between 1 and I, where I is the number of
    #         random variables.
    #
    #     Notes
    #     -----
    #     The equivalent formulation of the entropic risk assessing
    #     function is presented in THEOREM 9 (Equation (19)).
    #     """
    #
    #     K = 2
    #     Nz = x.size
    #
    #     t = self.dvar(2)
    #     q = self.dvar()
    #     u = self.dvar(Nz)
    #     w = self.dvar(K)
    #     r = self.dvar(K)
    #     eta = self.dvar(K)
    #     v = self.dvar((K, Nz))
    #     vbar = self.dvar(K)
    #     ubar = self.rvar(Nz)
    #     fset = (rso.norm(ubar - mu, 1) <= gamma)
    #
    #     self.minmax(t.sum(), fset)
    #
    #     self.st(eta.sum() <= q, q >= 0)
    #     self.st(rso.pexp(w + r - t[0], q) <= eta)
    #     self.st(v @ ubar + vbar <= w)
    #     self.st(-(1 / epsilon) * x - v[0] == 0)
    #     self.st((1 - 1 / epsilon) * vhat - vbar[0] <= t[1])
    #     self.st(v[1] == 0)
    #     self.st(vhat - vbar[1] <= t[1])
    #     for k in range(K):
    #         self.in_Phi_Gamma_data(r[k], q, v[k] - u, sigmas, Gamma)
    #         # self.st((pieces[k] - v[k] @ z - vbar[k] <= t[1]).forall())
# #### Function for solving the RO-$\Gamma$ model
#
# The `ro_gamma_cvar_data()` function defined below is used to solve the RO-$\Gamma$ model. Note that the objective
# as the entropic risk assessing function $\rho(\infty, \pmb{x})$ over the moment-dispersion ambiguity set can be easily
# defined using the `min_ce_infty()` method of the `GMModel` object.
def ro_gamma_cvar_data(mu, sigmas, Gamma, epsilon, gamma, z_upper=[], z_lower=[]):
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
    # print(z_upper, z_lower)
    # fset = (rso.norm(u - mu, 1) <= gamma, z<=z_upper, z>=z_lower)
    fset = (rso.norm(u - mu, 1) <= gamma)

    x = model.dvar(n)
    v = model.dvar()

    pieces = [(1 / epsilon) * (-x @ z - (1 - epsilon) * v), v]
    model.min_ce_infty_data(pieces, z, u, sigmas, Gamma, fset)
    model.st(x >= 0, x.sum() == 1)

    model.solve(msk, display=False)

    return x.get(), model.get()

# def ro_gamma_cvar_data2(mu, sigmas, Gamma, epsilon, gamma):
#     """
#     The function ro_gamma_cvar solves a robust model with a moment-
#     dispersion ambiguity set that aims to minimize the Conditional
#     Value-at-Risk (CVaR) of the portfolio return under real data.
#
#     Parameters
#     ----------
#     sigmas : dict
#         A dictionary as the output of the devs() function. The
#         forward deviation is given as the value of 'forward',
#         and the backward deviation is given as the value of
#         'backward'. Both deviations are presented as one-
#         dimensional arrays.
#     Gamma : {int, float, numpy.float64}
#         The value of the independence propensity hyperparameter
#         Gamma. It is between 1 and I, where I is the number of
#         random variables.
#     epsilon : float
#         The confidence level expressed as 1 - epsilon.
#
#     Returns
#     -------
#     xs : numpy.ndarray
#         A one-dimensional array representing the optimal portfolio
#         decision.
#     objval : float, numpy.float64
#         The optimal objective value of the SAA model.
#     """
#
#     n = len(mu)
#
#     model = GMModel()
#
#     # z = model.rvar(n)
#     # u = model.rvar(n)
#     # fset = (rso.norm(u - mu, 1) <= gamma)
#
#     x = model.dvar(n)
#     v = model.dvar()
#
#     # pieces = [(1 / epsilon) * (-x @ z - (1 - epsilon) * v), v]
#     # model.min_ce_infty_data(pieces, z, u, sigmas, Gamma, fset)
#     model.min_ce_infty_data2(x, v, mu, sigmas, Gamma, gamma, epsilon)
#     model.st(x >= 0, x.sum() == 1)
#
#     model.solve(msk, display=False)
#
#     return x.get(), model.get()

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

    model.min(v + (1 / s) * y.sum())
    model.st(y >= 0, y >= (1 / epsilon) * (-zhat @ x_sol - v))

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

    zhat = mu + xi @ Q.T

    return pd.DataFrame(zhat)


# In[13]:


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
    xi = np.array([np.random.beta(0.5 * j, 1 * j, size=size)
                   for j in range(1, J + 1)]).T

    zhat = mu + xi @ Q.T

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

    np.random.seed(2000 + r)

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
        raise ValueError('Unknown option for correlation type.')

    if factor.lower() == 'gaussian':
        samp_func = norm_return
    elif factor.lower() == 'beta':
        samp_func = beta_return
    else:
        raise ValueError('Unknown option for factor.')

    z_in = samp_func(mu0, Q, in_size).values
    z_out = samp_func(mu0, Q, out_size).values
    # z_out = numpy.append(z_out,z_in,axis=0)
    print(z_out.shape)
    # z_out = z_in

    # z_saa = np.array([np.random.choice(z_in[:, i], size=saa_size)
    #                   for i in range(z_in.shape[1])]).T
    z_saa = z_in
    mu = z_in.mean(axis=0)
    # z_upper = z_in.max(axis=0)*15
    # z_lower = z_in.min(axis=0)*15
    # print(mu)

    obj = {}
    x = {}
    cvar_out = {}
    x_sa, obj_sa = saa_cvar(z_saa, epsilon)

    x[f'SAA,{r}'] = x_sa
    cvar_out['SAA'] = out_sample_test(x_sa, z_out, epsilon)
    x_cla, cvar_out['Clairvoyance'] = saa_cvar(z_out, epsilon)

    mu_out = z_out.mean(axis=0)
    mean_out = {}
    mean_out['SAA'] = x_sa @ mu_out
    mean_out['Clairvoyance'] = x_cla @ mu_out

    # gamma = 0.01
    Gamma = 20
    delta = devs(z_in - mu)
    # print(delta['cov'])
    # cvar_fb_gamma = []
    for gamma in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]:
        # for gamma in [0]:
        x_fb, obj_fb = ro_gamma_cvar_data(mu, delta, Gamma, epsilon, gamma)
        cvar_out[f'RO-gamma={gamma}'] = out_sample_test(x_fb, z_out, epsilon)
        mean_out[f'RO-gamma={gamma}'] = x_fb @ mu_out
        obj[f'RO-gamma={gamma}'] = obj_fb
        x[f'RO-gamma={gamma},{r}'] = x_fb
        # print(x_fb.shape)
    theta = 0
    x_was, obj_was = wass_cvar(z_saa, epsilon, theta)
    cvar_out[f'Wass-{theta}'] = out_sample_test(x_was, z_out, epsilon)
    mean_out[f'Wass-{theta}'] = x_was @ mu_out
    x[f'Wass-{theta},{r}'] = x_was
    Theta = np.linspace(-5, -1, 15)
    for k, theta in enumerate(10 ** Theta):
        x_was, obj_was = wass_cvar(z_saa, epsilon, theta)
        cvar_out[f'Wass-{theta}'] = out_sample_test(x_was, z_out, epsilon)
        mean_out[f'Wass-{theta}'] = x_was @ mu_out
        x[f'Wass-{theta},{r}'] = x_was
    theta = 1
    x_was, obj_was = wass_cvar(z_saa, epsilon, theta)
    cvar_out[f'Wass-{theta}'] = out_sample_test(x_was, z_out, epsilon)
    mean_out[f'Wass-{theta}'] = x_was @ mu_out
    x[f'Wass-{theta},{r}'] = x_was

    return cvar_out, mean_out, obj, x

def one_test_cm(r, corr='default', factor='gaussian'):
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

    np.random.seed(2000 + r)

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
        raise ValueError('Unknown option for correlation type.')

    if factor.lower() == 'gaussian':
        samp_func = norm_return
    elif factor.lower() == 'beta':
        samp_func = beta_return
    else:
        raise ValueError('Unknown option for factor.')

    z_in = samp_func(mu0, Q, in_size).values
    z_out = samp_func(mu0, Q, out_size).values
    # z_out = numpy.append(z_out,z_in,axis=0)
    # print(z_out.shape)
    # z_out = z_in

    # z_saa = np.array([np.random.choice(z_in[:, i], size=saa_size)
    #                   for i in range(z_in.shape[1])]).T
    z_saa = z_in
    mu = z_in.mean(axis=0)
    std = z_in.std(axis=0)
    # z_upper = z_in.max(axis=0)*15
    # z_lower = z_in.min(axis=0)*15
    # print(mu)

    obj = {}
    x = {}
    cvar_out = {}
    mean_out = {}

    x[f'RO-CM-{r}'], obj['RO-CM'] = ro_cm_cvar(mu, std, epsilon)
    x[f'RO-MM-{r}'], obj['RO-MM'] = ro_mm_cvar(mu, std, epsilon)
    cvar_out['RO-CM'] = out_sample_test(x[f'RO-CM-{r}'], z_out, epsilon)
    cvar_out['RO-MM'] = out_sample_test(x[f'RO-MM-{r}'], z_out, epsilon)

    mu_out = z_out.mean(axis=0)
    mean_out['RO-CM'] = x[f'RO-CM-{r}'] @ mu_out
    mean_out['RO-MM'] = x[f'RO-MM-{r}'] @ mu_out

    return cvar_out, mean_out, obj, x

def one_test_data_generation(r, corr='default', factor='gaussian'):
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

    np.random.seed(2000 + r)

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
        raise ValueError('Unknown option for correlation type.')

    if factor.lower() == 'gaussian':
        samp_func = norm_return
    elif factor.lower() == 'beta':
        samp_func = beta_return
    else:
        raise ValueError('Unknown option for factor.')

    z_in = samp_func(mu0, Q, in_size).values
    mu = z_in.mean(axis=0)
    delta = devs(z_in - mu)
    Sigma = delta['cov']
    zeta = delta['zeta']

    return mu, Sigma, zeta.max(axis=0)*1.5, zeta.min(axis=0)*1.5

def one_test_sp(z):
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
    n,m = z.shape
    z_in = z[:int(q*n),:]
    z_out = z[int(q*n):,:]
    z_cumu = z_out + 1
    z_cumu = z_cumu.cumprod(axis=0)

    z_saa = z_in
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

    cumu_out = {}
    cumu_out['RO-CM'] = x_cm @ z_cumu[-1,:]
    cumu_out['RO-MM'] = x_mm @ z_cumu[-1,:]
    cumu_out['SAA'] = x_sa @ z_cumu[-1,:]
    cumu_out['Clairvoyance'] = x_cla @ z_cumu[-1,:]

    gamma = 0.01
    Gamma = 20
    delta = devs(z_in - mu)
    cvar_fb_gamma = []
    for gamma in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]:
        x_fb, obj_fb = ro_gamma_cvar_data(mu, delta, Gamma, epsilon, gamma)
        cvar_out[f'RO-gamma={gamma}'] = out_sample_test(x_fb, z_out, epsilon)
        mean_out[f'RO-gamma={gamma}'] = x_fb @ mu_out
        cumu_out[f'RO-gamma={gamma}'] = x_fb @ z_cumu[-1,:]
    Theta = np.linspace(-5, -1, 15)
    Theta = np.append(Theta, [0, 1])
    for k, theta in enumerate(10 ** Theta):
        x_was, obj_was = wass_cvar(z_saa, epsilon, theta)
        cvar_out[f'Wass-{k}'] = out_sample_test(x_was, z_out, epsilon)
        mean_out[f'Wass-{k}'] = x_was @ mu_out
        cumu_out[f'Wass-{k}'] = x_was @ z_cumu[-1,:]

    return cvar_out, mean_out, cumu_out

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

    theta_values = (10 ** np.linspace(-3, -1, 15)).round(5)
    # min_wass_index = np.argmin(cvar_avg['Wass-0':].values)

    plt.figure(figsize=(6, 4), dpi=100)
    plt.plot(np.arange(1, 21), cvar_avg['RO-Gamma=1':'RO-Gamma=20'],
             linewidth=1, marker='o', color='k', markerfacecolor='none',
             label=r'RO-$\Gamma$')
    plt.hlines(cvar_avg['SAA'], 1, 20,
               linewidth=1, linestyle='--', color='r', label='SAA')
    # plt.hlines(cvar_avg['RO-CM'], 1, 20,
    #            linewidth=1, linestyle='-.', color='y', label='RO-CM')
    # plt.hlines(cvar_avg['RO-MM'], 1, 20,
    #            linewidth=1, linestyle=':', color='m', label='RO-MM')
    # plt.hlines(cvar_avg[f'Wass-theta={theta_values[min_wass_index]:.5f}'], 1, 20,
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
    labels = ['SAA',
              r'RO-$\Gamma$($\Gamma$' + f'={min_ro_index + 1})']
    plt.boxplot([cvar_results['SAA'],
                 cvar_results.loc[:, f'RO-Gamma={min_ro_index + 1}']],
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
# #### Figures 5, 6 and 7
#
I = J = 20
epsilon = 0.05
# change the value of the following three parameters
in_size = 20
saa_size = 20
out_size = 10000

repeats = 50

cvar_results = []
mean_results = []
obj_results = []
x_results = []
for r in range(repeats):
    if (r + 1) % 10 == 0:
        print(f'Repeat {r + 1} ...')
    # use either one_test or one_test_cm
    cvar_out, mean_out, obj_out, x = one_test(r,factor='gaussian')  # change the value of 'factor' equaling to gaussian or beta
    # cvar_out, mean_out, obj_out, x = one_test_cm(r)
    cvar_results.append(cvar_out)
    mean_results.append(mean_out)
    obj_results.append(obj_out)
    x_results.append(x)
# print(x_results)
# print(obj_results)

combined_dict = {k: v for d in x_results for k, v in d.items()}

# Convert the combined dictionary to a pandas DataFrame
df = pd.DataFrame(combined_dict)

#
# for d in x_results:
#     temp_df = pd.DataFrame({k: pd.Series(v) for k, v in d.items()})
#     df = pd.concat([df, temp_df], ignore_index=True)
# print(df)

cvar_results = pd.DataFrame(cvar_results)
mean_results = pd.DataFrame(mean_results)
obj_results = pd.DataFrame(obj_results)
# x_results = pd.DataFrame(x_results)

resultPath = f'portfolio_data_driven_guassian_{in_size}_{epsilon}.xlsx'
writer = pd.ExcelWriter(resultPath)
cvar_results.to_excel(writer, sheet_name="cvar", index=False, na_rep=0, inf_rep=0)
mean_results.to_excel(writer, sheet_name="mean", index=False, na_rep=0, inf_rep=0)
obj_results.to_excel(writer, sheet_name="obj", index=False, na_rep=0, inf_rep=0)
df.to_excel(writer, sheet_name="x", index=False, na_rep=0, inf_rep=0)
writer.close()

# #### Figures 6 and 7
#
I = J = 20
epsilon = 0.2
# change the value of the following three parameters
in_size = 20000
saa_size = 20000
out_size = 10000

repeats = 50

cvar_results = []
mean_results = []
obj_results = []
x_results = []
for r in range(repeats):
    if (r + 1) % 10 == 0:
        print(f'Repeat {r + 1} ...')
    # use either one_test or one_test_cm
    cvar_out, mean_out, obj_out, x = one_test(r,factor='gaussian')  # change the value of 'factor' equaling to gaussian or beta
    # cvar_out, mean_out, obj_out, x = one_test_cm(r,corr='weak')
    cvar_results.append(cvar_out)
    mean_results.append(mean_out)
    obj_results.append(obj_out)
    x_results.append(x)
# print(x_results)
# print(obj_results)

combined_dict = {k: v for d in x_results for k, v in d.items()}

# Convert the combined dictionary to a pandas DataFrame
df = pd.DataFrame(combined_dict)

#
# for d in x_results:
#     temp_df = pd.DataFrame({k: pd.Series(v) for k, v in d.items()})
#     df = pd.concat([df, temp_df], ignore_index=True)
# print(df)

cvar_results = pd.DataFrame(cvar_results)
mean_results = pd.DataFrame(mean_results)
obj_results = pd.DataFrame(obj_results)
# x_results = pd.DataFrame(x_results)

resultPath = f'portfolio_data_driven_guassian_large_{in_size}_{epsilon}.xlsx'
writer = pd.ExcelWriter(resultPath)
cvar_results.to_excel(writer, sheet_name="cvar", index=False, na_rep=0, inf_rep=0)
mean_results.to_excel(writer, sheet_name="mean", index=False, na_rep=0, inf_rep=0)
obj_results.to_excel(writer, sheet_name="obj", index=False, na_rep=0, inf_rep=0)
df.to_excel(writer, sheet_name="x", index=False, na_rep=0, inf_rep=0)
writer.close()



# ## Figures 8 and 9
step = 50
epsilon = 0.2
period_type = 'W'
q = 0.8
repeats = 1
combine = 50
df = pd.read_csv('data/daily_returns.csv', index_col=0)
df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))

df = df + 1
period_stock = df.resample(period_type).prod() - 1
name = period_stock.index
period_stock = period_stock.drop(name[-1])
# print(period_stock)
n, m = period_stock.shape
cvar_results = []
mean_results = []
cumu_results = []
for s in range(combine):
    np.random.seed(2000+s)
    sample = period_stock.sample(n=I, axis=1)
    # print(sample.index.values[-50:])
    sample = sample.values
    for r in range(repeats):

        z = sample[n - step - r:n - r, :]
        if (r + 1) % 10 == 0:
            print(f'Repeat {r + 1} ...')

        cvar_out, mean_out, cumu_out = one_test_sp(z)
        cvar_results.append(cvar_out)
        mean_results.append(mean_out)
        cumu_results.append(cumu_out)


cvar_results = pd.DataFrame(cvar_results)
mean_results = pd.DataFrame(mean_results)
cumu_results = pd.DataFrame(cumu_results)
resultPath = f'portfolio_sp_40_10_week_cumu_cm_{epsilon}.xlsx'
writer = pd.ExcelWriter(resultPath)
cvar_results.to_excel(writer, sheet_name="cvar", index=False, na_rep=0, inf_rep=0)
mean_results.to_excel(writer, sheet_name="mean", index=False, na_rep=0, inf_rep=0)
cumu_results.to_excel(writer, sheet_name="cumu", index=False, na_rep=0, inf_rep=0)
writer.close()



# #### Figure 4.
# step = 1000
# q = 0.8
# repeats = 1
# combine = 50
# df = pd.read_csv('daily_returns.csv', index_col=0)
# n, m = df.shape
# cvar_results = []
# mean_results = []
# cumu_results = []
# for s in range(combine):
#     np.random.seed(2000+s)
#     sample = df.sample(n=I, axis=1)
#     print(sample)
#     sample = sample.values
#     for r in range(repeats):
#
#         z = sample[n - step - r:n - r, :]
#         if (r + 1) % 10 == 0:
#             print(f'Repeat {r + 1} ...')
#
#         cvar_out, mean_out, cumu_out = one_test_sp(z)
#         cvar_results.append(cvar_out)
#         mean_results.append(mean_out)
#         cumu_results.append(cumu_out)
#
#
# cvar_results = pd.DataFrame(cvar_results)
# mean_results = pd.DataFrame(mean_results)
# cumu_results = pd.DataFrame(cumu_results)
# resultPath = 'portfolio_sp_800_200_cumu.xlsx'
# writer = pd.ExcelWriter(resultPath)
# cvar_results.to_excel(writer, sheet_name="cvar", index=False, na_rep=0, inf_rep=0)
# mean_results.to_excel(writer, sheet_name="mean", index=False, na_rep=0, inf_rep=0)
# cumu_results.to_excel(writer, sheet_name="cumu", index=False, na_rep=0, inf_rep=0)
# writer.close()




### Figures in the Paper

#### data generation for infinite constraint ambiguity set
# repeats = 50
#
# mu_results = []
# Sigma_results = []
# for r in range(repeats):
#     if (r + 1) % 10 == 0:
#         print(f'Repeat {r + 1} ...')
#     mu, Sigma, upper, lower = one_test_data_generation(r)
#     mu = pd.DataFrame(mu)
#     Sigma = pd.DataFrame(Sigma)
#     upper = pd.DataFrame(upper)
#     lower = pd.DataFrame(lower)
#
#     resultPath = f'parameter//random_Gaussian//portfolio_parameter_{r}.xlsx'
#     writer = pd.ExcelWriter(resultPath)
#     mu.to_excel(writer, sheet_name="mu", index=False, na_rep=0, inf_rep=0)
#     Sigma.to_excel(writer, sheet_name="Sigma", index=False, na_rep=0, inf_rep=0)
#     upper.to_excel(writer, sheet_name="upper", index=False, na_rep=0, inf_rep=0)
#     lower.to_excel(writer, sheet_name="lower", index=False, na_rep=0, inf_rep=0)
#     writer.close()

