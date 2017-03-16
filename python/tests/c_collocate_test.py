import sympy as sp
import numpy as np
import hydrocyclone
import bayesian_pdes as bpdes

def test_c_operators():
    s_x, s_xbar, s_y, s_ybar = sp.symbols('x,xbar,y,ybar')
    a, a_x, a_y = sp.symbols('a,a_x,a_y')
    a_bar, a_x_bar, a_y_bar = sp.symbols('abar,a_xbar,a_ybar')
    s_length_scale, s_variance = sp.symbols('l,sigma')
    kernel = s_variance*sp.exp(-((s_x-s_xbar)**2 + (s_y-s_ybar)**2) / (2.*s_length_scale**2))
    symbols = [[s_x, s_y, a, a_x, a_y], [s_xbar, s_ybar, a_bar, a_x_bar, a_y_bar], [s_length_scale, s_variance]]

    def A(k):
        return sp.exp(a)*(k.diff(s_x,s_x) + k.diff(s_y,s_y) + k.diff(s_x)*a_x + k.diff(s_y)*a_y)
    def A_bar(k):
        return sp.exp(a_bar)*(k.diff(s_xbar,s_xbar) + k.diff(s_ybar,s_ybar) + k.diff(s_xbar)*a_x_bar + k.diff(s_ybar)*a_y_bar)
    def B(k):
        return sp.exp(a)*(k.diff(s_x)*s_x + k.diff(s_y)*s_y)
    def B_bar(k):
        return sp.exp(a_bar)*(k.diff(s_xbar)*s_xbar + k.diff(s_ybar)*s_ybar)

    op_system = bpdes.operator_compilation.sympy_gram.compile_sympy(
        [A, B], [A_bar, B_bar],
        kernel,
        symbols,
        parallel=False
    )

    n_in_shell = 8
    n_bdy = 64
    n_sensor = 8
    grid = hydrocyclone.grids.construct_circular(n_in_shell, n_bdy, n_sensor)

    kappa = np.random.normal(0, 1, 3*len(grid.all) - 2*len(grid.sensors))
    proposal_dot_mat = np.eye(kappa.shape[0])
    fun_args = np.array([0.3, 1e-2])

    x = np.hstack([grid.sensors, np.empty((8,3))])
    p = hydrocyclone.pcn_kernel.construct_posterior(grid, op_system, kappa, fun_args, proposal_dot_mat)
    mu, cov = p.no_obs_posterior(x)

    x = np.hstack([grid.sensors, np.empty((8,3))])
    mu2, cov2 = hydrocyclone.pcn_kernel.construct_c_posterior(x, grid, kappa, fun_args, proposal_dot_mat)

    assert np.abs((mu2-mu) / mu).mean() < 1e-4
    assert np.abs((cov2-cov) / cov).mean() < 1e-4