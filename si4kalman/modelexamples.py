import casadi as ca  # type: ignore
import numpy as np  # type: ignore
from .modelparser import ModelParser

def minimal_model(dt, uscale, Text=0., offset_free=False, min_beta=0.):
    T1 = ca.SX.sym("T1") 
    T2 = ca.SX.sym("T2") 
    T3 = ca.SX.sym("T3") 
    T4 = ca.SX.sym("T4")

    a_middle = ca.SX.sym("amed") 
    aext = ca.SX.sym("aext")
    
    u = ca.vcat([T1])
    x = ca.vcat([T2, T3, T4])
    y = ca.vcat([T2, T4])
    alpha = ca.vcat([a_middle, aext])

    T2_dot = a_middle * (T1 - T2)
    T3_dot = a_middle * (T2 - T3)
    T4_dot = a_middle * (T3 - T4) + aext * (Text - T4)

    xdot = ca.vcat([T2_dot, T3_dot, T4_dot])
    F = ca.Function("dynamics", [x, u, alpha], [xdot])
    G = ca.Function("measurement", [x], [y])
    Fdiscr = discretize(F, dt)
    nx = x.shape[0]
    ny = y.shape[0]
    sigma_y = ca.SX.sym("sigma_y")
    sigma_x = ca.SX.sym("sigma_x")
    if offset_free:
        sigma_d = ca.SX.sym("sigma_d")
        diagSx =  ca.vertcat(sigma_d * ca.DM.ones(ny), sigma_x * ca.DM.ones(nx - ny))
        beta = ca.vertcat(sigma_x, sigma_d, sigma_y)
    else:
        diagSx = sigma_x * ca.DM.ones(nx)
        beta = ca.vertcat(sigma_x, sigma_y)
    Sx = ca.diag(diagSx)
    Sy =  sigma_y *  ca.DM.eye(ny)
    Q_func = ca.Function("Sx_func", [beta], [Sx])
    R_func = ca.Function("Sy_func", [beta], [Sy])

    model = ModelParser(Fdiscr, G, Q_func, R_func)
    if offset_free:
        model.augment_model()
    x0 = np.zeros(model.nx)
    A = model.Fdiscr.jacobian()(x0, uscale * np.ones(model.nu), alpha, 0)
    diagA = ca.diag(A)
    ineq = ca.vertcat(alpha, diagA, beta - min_beta)
    model.Ineq = ca.Function("ineq", [alpha, beta], [ineq])
    return model

def rw_model(min_beta=0.):
    
    u = ca.SX.sym("u", 1)
    x = ca.SX.sym("x", 1)
    y = x
    xplus = x
    alpha = ca.SX.sym("alpha", 1)
    beta = ca.SX.sym("beta", 2)

    F = ca.Function("dynamics", [x, u, alpha], [xplus])
    G = ca.Function("measurement", [x], [y])
    

    diagSx = ca.vcat([beta[0]])
    diagSy = ca.vcat([beta[1]])
    Sx = ca.diag(diagSx)
    Sy = ca.diag(diagSy)
    Q_func = ca.Function("Sx_func", [beta], [Sx])
    R_func = ca.Function("Sy_func", [beta], [Sy])

    model = ModelParser(F, G, Q_func, R_func)

    # ineq = ca.vertcat(alpha, diagA, beta - min_beta)
    max_alpha = 1.
    max_beta = 1.
    ineq = ca.vertcat(alpha, max_alpha - alpha, beta - min_beta, max_beta - beta)
    model.Ineq = ca.Function("ineq", [alpha, beta], [ineq])

    return model

def condconv_model(nx=3, measurements=[1], qu=None, q=None, os=False, scales=None, min_beta=0.):
    if scales is None:
        scales = {"r":1., "os":1., "q":1.}
    u = ca.SX.sym("u", 2)
    x_ = ca.SX.sym("x", nx)
    alpha = ca.SX.sym("alpha", 3)
    y_ = ca.vcat([x_[i-1] for i in measurements])
    ny = y_.shape[0]
    # dynamical system
    a = (alpha[1] + u[1] * alpha[2] ) / 10
    b = a * alpha[0]
    xplus_list = [ (1.-a)* x_[0] + b * u[0] ]
    for i in range(nx-1):
        xplus_list.append(
            (1.-a)* x_[i+1] + a * x_[i]
        )
    # noise model
    r = ca.SX.sym("diagR", ny)
    diagR = scales["r"] * r
    beta_q = []
    if qu is None:
        qu = ca.SX.sym("qu", 1)
        beta_q.append(qu)
    if q is None:
        q = ca.SX.sym("q", 1)
        beta_q.append(q)
    diagQ_list = [scales["q"] * q]*nx
    diagQ_list[0] = qu
    diagQ = ca.vcat(diagQ_list)
    if os:
        # offset free part
        nd = 1
        d = ca.SX.sym('d', nd)
        y = y_ + d
        x = ca.vcat([x_, d])
        xplus_list.append(d)
        qd = ca.SX.sym("qd", 1)
        diagQd = scales["os"] * ca.vcat([qd]*nd)
        diagQ = ca.vcat([diagQ, diagQd])
        beta_q.append(qd)
    else:
        x = x_
        y = y_    
    beta = ca.vcat(beta_q + [r])
    xplus = ca.vcat(xplus_list)
    F = ca.Function("dynamics", [x, u, alpha], [xplus])
    G = ca.Function("measurement", [x], [y])
    Q = ca.diag(diagQ)
    R = ca.diag(diagR)
    Q_func = ca.Function("Q_func", [beta], [Q])
    R_func = ca.Function("R_func", [beta], [R])
    model = ModelParser(F, G, Q_func, R_func)
    # ineq = ca.vertcat(alpha, diagA, beta - min_beta)
    max_alpha = 1.
    max_beta = 1.
    ineq = ca.vertcat(alpha, max_alpha - alpha, beta - min_beta, max_beta - beta)
    model.Ineq = ca.Function("ineq", [alpha, beta], [ineq])
    return model

# misc functions 
def discretize(F, dt, method="Euler"):
    x = ca.SX.sym("x", F.size1_in(0))
    u = ca.SX.sym("x", F.size1_in(1))
    alpha = ca.SX.sym("x", F.size1_in(2))

    if method == "Euler":
        xplus = x + dt * F(x, u, alpha)
    elif method == "RK4":
        dt1, dt2, dt3, dt4 = dt, 0.0, 0.0, 0.0
        x1 = x + dt1 * F(x, u, alpha)

    Fdiscr = ca.Function("Fdiscr", [x, u, alpha], [xplus])
    return Fdiscr
