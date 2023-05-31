# %%
# Leo Simpson, University of Freiburg, Tool-Temp AG, 2023

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib notebook

# %%
import sys, os
from os.path import join, dirname
main_dir = dirname(os.getcwd())
sys.path.append(main_dir)

# %%
import numpy as np
from time import time
import matplotlib.pyplot as plt # type: ignore
import casadi as ca # type: ignore
from src import ProblemParser, ModelParser # main objects to use the present algorithms
from src import plot_data, plot_est, plot_res # plotting tools
rng = np.random.default_rng(seed=0)

# %%
dt = 1.
beta_min = 1e-6
Text = 0.
alpha_max = 0.5

def dynamic(x, u, alpha, beta):
    T1 = u[0]
    T2 = x[0]
    T3 = x[1]
    T4 = x[2]
    a_middle = alpha[0]
    aext = alpha[1]
    nx = 3

    # equations of the system
    T2_plus = T2 + dt * a_middle * (T1 - T2)
    T3_plus = T3 + dt * a_middle * (T2 - T3)
    T4_plus = T4 + dt *( a_middle * (T3 - T4) + aext * (Text - T4))
    
    # construction of the output
    x_plus = ca.vcat([T2_plus, T3_plus, T4_plus])
    y = ca.vcat([T2, T4])
    ny = 2
    
    # noise model
    Q =  beta[0] *  ca.DM.eye(nx)
    R =  beta[1] *  ca.DM.eye(ny)
    
    # inequality constraints on the form h > 0
    h = ca.vertcat(alpha, alpha_max - alpha, beta - beta_min)
    return x_plus, y, Q, R, h


# %%
# Define the model with Casadi symbolics
x_symbol = ca.SX.sym("x", 3)
u_symbol = ca.SX.sym("u", 1)
alpha_symbol = ca.SX.sym("alpha", 2)
beta_symbol = ca.SX.sym("beta", 2)
xplus_symbol, y_symbol, Q_symbol, R_symbol, h_symbol = dynamic(x_symbol, u_symbol, alpha_symbol, beta_symbol)

# %%
# Define a Casadi functions associated with the model
xplus_fn = ca.Function("xplus", [x_symbol, u_symbol, alpha_symbol], [xplus_symbol])
y_fn = ca.Function("y", [x_symbol], [y_symbol])
Q_fn = ca.Function("Q", [beta_symbol], [Q_symbol])
R_fn = ca.Function("R", [beta_symbol], [R_symbol])
h_fn = ca.Function("h", [alpha_symbol, beta_symbol], [h_symbol])

# %% [markdown]
# # Problem definition

# %%
Ntrain = 3000

model_true = ModelParser(xplus_fn, y_fn, Q_fn, R_fn)
model_true.Ineq = h_fn
model = ModelParser(xplus_fn, y_fn, Q_fn, R_fn)
model.Ineq = h_fn

x0 = np.zeros(model_true.nx)
P0 = np.eye(model_true.nx) * 0.

# %% [markdown]
# # Data generation 

# %%
umax = 50
us_train = model_true.generate_u(rng, Ntrain, umax=umax, step = 10)

# %%
# noise in the true data
alpha_true, beta_true = model_true.draw(rng) # choose a "true parameter randomly"

# %%
ys_train, _ = model_true.simulation(x0, us_train, alpha_true,  beta_true, rng)

assert model_true.feasible(alpha_true, beta_true), "Constraints should be satisfied for the true parameters"

# %% [markdown]
# # Estimation

# %%
alpha_true, beta_true

# %%
# the flag lti allow to speed things up for LTI systems 
problemTrain =ProblemParser(
    model, ys_train, us_train, x0, P0, lti=True)
formulation = "MLE"  # can be "MLE", "PredErr" (remark: to apply PredErr, one needs a different parameterization)

# %%
dict_opts = {
    "SQP":{"pen_step":1e-4, "maxiter":20, "tol.direction":0., "tol.kkt":1e-8}, # parameters of the SQP method
    "IPOPT":{} # use default parameters of IPOPT
}
Ns = [200, 500, 1000, 1500]
def diff(x1, x2):
    return np.sum((x1-x2)**2)


# %%
## Optimize over the Kalman filter
res = {}

alpha0 = np.ones(model.nalpha) * 0.2
beta0 = np.ones(model.nbeta) * 0.2
for algorithm, einsum in [("SQP", True), ("SQP", False), ("IPOPT", None)]: # the second element is wether we use einsum or not
    infos = {}
    for N in Ns:
        name_algo = algorithm + " with einsum = " + str(einsum)
        print("Algorithm : {}, N = {}".format(name_algo, N))
        problemTrain.cut(N)
        t0 = time()
        dict_opts[algorithm]["einsum"] = einsum
        alpha, beta, stats = problemTrain.solve(alpha0, beta0,
                                                formulation, algorithm,
                                                opts=dict_opts[algorithm], verbose=False)
        rtime = time() - t0
        error_alpha = diff(alpha, alpha_true)
        error_beta = diff(beta, beta_true)
        error = error_alpha + error_beta
        if algorithm == "SQP":
            niter = stats['niter']
        else:
            niter = 1
        info = {
                "rtime": rtime,
                "status":stats["return_status"],
                "rtime-per-iter": rtime/niter,
                "alpha": alpha.copy(),
                "beta": beta.copy(),
                "alpha_true": alpha_true.copy(),
                "beta_true": beta_true.copy(),
                "error": error,
                "error_alpha":error_alpha,
                "error_beta":error_beta
            }
        infos[N] = info
        print("rtime : {:.2e}  status : {}".format(rtime, stats["return_status"]))
        
        alpha0 = alpha.copy() # for making faster, to remove
        beta0 = beta.copy()
        
    res[name_algo] = infos

# %%
for name_algo, infos in res.items():
    print("solution for", name_algo)
    for N, info in infos.items():
        print("N = {}".format(N))
        print("Running time : {:.2e}".format(info["rtime"]))
        print("Error : {:.2e}".format(info["error"]))
        print("Status : {}".format(info["status"]))
        print("beta_true : {}".format(info["beta_true"]))
        print("beta : {}".format(info["beta"]))
        print("alpha_true : {}".format(info["alpha_true"]))
        print("alpha : {}".format(info["alpha"]))

        print(" ")
    print(" ")

# %%
fig = plot_res(res, "rtime", scale="lin")
# fig = plot_res(res, "rtime-per-iter", scale="lin")

# %%
fig = plot_res(res, "error_alpha")

# %%
fig = plot_res(res, "error_beta")

# %%

# %%
