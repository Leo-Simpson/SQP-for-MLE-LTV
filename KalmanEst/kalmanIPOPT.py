import casadi as ca 
import numpy as np 
import KalmanEst.misc as misc

class NLP:
    def __init__(self):
        self.nlp = {
            "f": 0., 
            "x": [],
            "g": []
            }
    
        self.solver_dict = {
            "x0": np.array([]),
            "lbx": np.array([]),
            "ubx": np.array([]),
            "lbg": np.array([]),
            "ubg": np.array([]),
        }

        self.keep = [] # list for not putting to the trash some objects

    def direct_init(self, nlp,  lbg, ubg, X0):
        self.nlp = nlp
        self.solver_dict: dict[str, np.ndarray] = {
            "x0": X0,
            "lbx": -np.inf * np.ones_like(X0),
            "ubx": np.inf * np.ones_like(X0),
            "lbg": lbg,
            "ubg": ubg,
        }

    def add_var(self, var, var0, ubx=np.inf, lbx=-np.inf):
        self.nlp["x"].append(var.T.reshape((-1, 1)))
        if type(var0) is not np.ndarray:
            var0 = var0.full()
        var0_ = var0.reshape(-1)

        self.solver_dict["x0"] = np.append(self.solver_dict["x0"], var0_)
        self.solver_dict["lbx"] = np.append(self.solver_dict["lbx"], np.ones(len(var0_))*lbx)
        self.solver_dict["ubx"] = np.append(self.solver_dict["ubx"], np.ones(len(var0_)) * ubx)

    def add_eq(self, eq, ubg=0., lbg=0.):
        eq_ = eq.reshape((-1, 1))
        self.nlp["g"].append(eq_)
        self.solver_dict["lbg"] = np.append(self.solver_dict["lbg"], np.ones(eq_.shape[0])*lbg)
        self.solver_dict["ubg"] = np.append(self.solver_dict["ubg"], np.ones(eq_.shape[0]) * ubg)
    
    def verif_init(self, tol=1e-8):
        x_sym = ca.vcat(self.nlp["x"])
        g_sym = ca.vcat(self.nlp["g"])
        g0 = misc.dm2np(ca.Function("constr", [x_sym], [g_sym])(self.solver_dict["x0"]))
        cond = np.all(g0 <= self.solver_dict["ubg"]+tol) and np.all(g0 >= self.solver_dict["lbg"]-tol)
        if not cond:
            raise ValueError("The initial point does not satisfy the constraints")

    def _solver_options(self, opts):
        self.nlpsolver_options = {}
        self.nlpsolver_options["expand"] = False
        self.nlpsolver_options["ipopt.max_iter"] = 500
        self.nlpsolver_options["ipopt.max_cpu_time"] = 3600.0
        self.nlpsolver_options["ipopt.linear_solver"] = "mumps"  # suggested: ma57
        self.nlpsolver_options["ipopt.mumps_mem_percent"] = 10000
        self.nlpsolver_options["ipopt.mumps_pivtol"] = 0.001
        self.nlpsolver_options["ipopt.print_level"] = 5
        self.nlpsolver_options["ipopt.print_frequency_iter"] = 10

        for key, value in opts.items():
            self.nlpsolver_options[key] = value

    def presolve(self, opts={}):
        self._solver_options(opts)
        nlp_stack = {
            "x": ca.vcat(self.nlp["x"]),
            "f": self.nlp["f"],
            "g": ca.vcat(self.nlp["g"])
        }
        self.nlpsol = ca.nlpsol("S", "ipopt", nlp_stack, self.nlpsolver_options)

    def solve(self) -> dict:
        r = self.nlpsol(
            x0=self.solver_dict["x0"],
            lbx=self.solver_dict["lbx"],
            ubx=self.solver_dict["ubx"],
            lbg=self.solver_dict["lbg"],
            ubg=self.solver_dict["ubg"],
        )
        return r

    def add_value(self, value):
        self.nlp["f"] = self.nlp["f"] + value

    def remember(self, obj):
        self.keep.append(obj)

def nlp_kalman(problem, alpha0, beta0, formulation):
    us = problem.us[0]
    ys = problem.ys[0] # TODO : that takes only the first dataset, so I have to make it for several datasets
    model = problem.model
    x0 = problem.x0[0]
    P0 = problem.P0[0]
    N = ys.shape[0] - 1
    nP = int(model.nx * (model.nx + 1) / 2)
    nS = int(model.ny * (model.ny + 1) / 2)
    alpha = misc.sym("alpha", model.nalpha)
    beta = misc.sym("beta", model.nbeta)

    if model.Ineq is None:
        ineq = np.array([0])
    else:
        ineq = model.Ineq(alpha, beta)

    if P0 is None:
        P0 = init_P0(us, ys, x0, model, alpha0, beta0)

    nlp = NLP()
    nlp.add_var(alpha, alpha0)
    nlp.add_eq(ineq, ubg=np.inf)
    nlp.add_eq(beta, ubg=np.inf) # seems useless, but on the example, divides by 100 the running time 

    x_k = x0.copy()
    P_k = P0.copy()
    x_k0 = x0
    P_k0 = P0
    Afn = model.Fdiscr.jacobian()
    C = model.G.jacobian()(x0, 0)

    def get_S(x, P, R):
        S = C @ P @ C.T + R
        return S

    def update(x, P, Q, S, a, u, y, M=None):
        A_ = Afn(x, u, a, 0)
        A = misc.select_jac(A_, model.nx)
        if M is None:
            M = ca.inv(S)
        K = P @ C.T @ M
        P_est = P - K @ C @ P
        P_next = A @ P_est @ A.T + Q
        x_est = x + K @ (y - model.G(x))
        x_next = model.Fdiscr(x_est, u, a)
        return P_next, x_next
    
    nlp.add_var(beta, beta0, lbx=0.)

    Q = model.Q_fn(beta)
    Q0 = model.Q_fn(beta0)
    R = model.R_fn(beta)
    R0 = model.R_fn(beta0)
    
    for k in range(N+1):
        # Equation for S
        S_vec = misc.sym("S{}".format(k+1), nS)
        S_k = misc.vec2sym(S_vec, model.ny)
        Sk_new = get_S(x_k, P_k, R)
        Sk0 = get_S(x_k0, P_k0, R0)

        nlp.add_var(S_vec, misc.sym2vec(Sk0))
        nlp.add_eq( misc.sym2vec( S_k - Sk_new ) )

        M_k = ca.inv(S_k)
        
        innov = ys[k] - model.G(x_k)
        logdet_term = ca.log( ca.fabs(ca.det(S_k))  )
        if formulation == "MLE":
            stage_cost = innov.T @(M_k @ innov) +  logdet_term
        elif formulation == "PredErr":
            stage_cost = innov.T @ innov
        else:
            raise ValueError("Formulation {} is unknown. Choose between 'MLE' or 'PredError'".format(formulation))
        nlp.add_value(stage_cost)
        if k == N:
            break
        P_next, x_next = update(x_k, P_k, Q, S_k, alpha, us[k], ys[k], M=M_k)
        P_k0, x_k0 = update(x_k0, P_k0, Q0, Sk0, alpha0, us[k], ys[k])
        
        x_k = misc.sym("x{}".format(k+1), model.nx)
        P_vec = misc.sym("P{}".format(k+1), nP)
        P_k = misc.vec2sym(P_vec, model.nx)
        
        nlp.add_var(x_k, x_k0)
        nlp.add_var(P_vec, misc.sym2vec(P_k0))
        nlp.add_eq((x_next - x_k))
        nlp.add_eq(misc.sym2vec(P_k - P_next))

    return nlp

def nlp_kalman_solve(problem, alpha0, beta0, formulation, opts={}, rescale=False):
    nlp = nlp_kalman(problem, alpha0, beta0, formulation)
    nlp.presolve(opts=opts)
    x = misc.dm2np(nlp.solve()["x"])
    alpha_found = x[:problem.nalpha]
    beta_found = x[problem.nalpha:problem.nalpha+problem.nbeta]
    stats = nlp.nlpsol.stats()
    if rescale:
        beta_found = beta_found * problem.scale(alpha_found, beta_found)
    return alpha_found, beta_found, stats

def init_P0(us, ys, x0, model, alpha, beta):
    p0s = model.kalman(us, ys, x0, np.zeros((model.nx, model.nx)), alpha, beta, save_pred=True)["Ps_pred"]
    return p0s[-1]
