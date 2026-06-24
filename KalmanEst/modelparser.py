import casadi as ca #type: ignore
import numpy as np
import numpy.linalg as LA
import contextlib
from math import ceil
from numpy.linalg import norm
from scipy.linalg import sqrtm
import KalmanEst.misc as misc

class ModelParser:
    def __init__(self, Fdiscr, G, Q_fn, R_fn):
        self.Fdiscr = Fdiscr
        self.G =  G
        self.Q_fn = Q_fn
        self.R_fn = R_fn
        self.Ineq: ca.Function | None = None
        self.get_dim()

    def get_dim(self):
        self.nalpha = self.Fdiscr.size1_in(2)
        self.nbeta = self.Q_fn.size1_in(0)
        self.nx = self.Fdiscr.size1_in(0)
        self.nu = self.Fdiscr.size1_in(1)
        self.ny = self.G.size1_out(0)
    
    def get_QR(self, beta):
        return self.Q_fn(beta), self.R_fn(beta)

    def get_S(self, beta, P):
        R = self.R_fn(beta).full()
        C = self.G.jacobian()(np.zeros(self.nx), 0).full()
        return C @ (P @ C.T) + R
    
    def propP(self, P, u, alpha, Q):
        A_ = self.Fdiscr.jacobian()(np.zeros(self.nx), u, alpha, 0)
        A = misc.select_jac(A_, self.nx).full()
        P_next = A @ P @ A.T + Q
        return misc.symmetrize(P_next)

    def feasible(self, alpha, beta):
        if self.Ineq is None:
            return True
        else:
            return misc.dm2np(self.Ineq(alpha, beta)).min() > 0.


    def trajectory(self, x0, us, alpha):
        x = x0.copy()
        xs = [x]
        for i in range(len(us)):
            x = self.Fdiscr(x, us[i], alpha)
            xs.append(x)
        return ca.hcat(xs)
    
    def trajectory_P(self, P0, us, alpha, Q):
        P = P0.copy()
        Ps = [P]
        for i in range(len(us)):
            P = self.propP(P, us[i], alpha, Q)
            Ps.append(P)
        return Ps

    def sim_predictions(self, us, xs, alpha, npred, Npred=None):
        N = len(us)
        ts = np.arange(N+1, dtype=int)
        ys_pred, ts_pred = [], []
        if Npred is None:
            Npred = int(float(N) / npred)  # this is the number of time point in one interval of prediction
        interval = int(  float(N - Npred) / (npred - 1) )
        G_map  = self.G.map(Npred+1)
        for i in range(npred):
            idx = i * interval
            xstart = xs[idx]
            upred = us[idx:idx+Npred]
            tpred = ts[idx:idx+Npred+1]
            xpred = self.trajectory(xstart, upred, alpha)
            ypred = misc.dm2np(G_map(xpred), shape="matrix").T
            ys_pred.append(ypred)
            ts_pred.append(tpred)

        return ts_pred, ys_pred

    def predictions_S(self, us, Ps, alpha, beta, npred, Npred=None):
        N = len(us)
        S_pred = []
        if Npred is None:
            Npred = int(float(N) / npred)  # this is the number of time point in one interval of prediction
        interval = int(  float(N - Npred) / (npred - 1) )
        Q = self.Q_fn(beta).full()
        for i in range(npred):
            idx = i * interval
            Pstart = Ps[idx]
            upred = us[idx:idx+Npred]
            Ppred = self.trajectory_P(Pstart, upred, alpha, Q)
            spred = np.array([ self.get_S(beta, P) for P in Ppred])
            S_pred.append(spred)
        return S_pred

    def generate_u(self, rng, N, umax=1., umin=0., step=None, step_len=None):
        assert not (step is None and step_len is None), "has to specify one of these"
        if step is None:
            step = ceil(N / step_len)
        if step_len is None:
            step_len = ceil(N / step)
        du = umax - umin
        Us = sum(
            [
            [umin + rng.random(self.nu) * du] * step_len
            for i in range(step)
            ], [])
        assert len(Us) >= N, "there should be more U than N"
        Us = Us[:N]
        us = np.array(Us)
        return us

    def simulation(self, x0, us, alpha, beta, rng=None):
        # Q is the covariance of the noise which acts on the discrete equations
        if rng is None:
            rng = np.random.default_rng(seed=42)
        Q, R = self.get_QR(beta)
        sqrR = sqrtm(R)
        sqrQ = sqrtm(Q)
        x = ca.DM(x0.copy())
        Ys = []
        Ytrues = []
        N = len(us)
        for i in range(N+1):
            ytrue = self.G(x)
            y = ytrue + sqrR @ rng.normal(size=self.ny)
            Ys.append(y)
            Ytrues.append(ytrue)
            if i == N:
                break
            x = self.Fdiscr(x, us[i], alpha)      
            noise =  sqrQ @  rng.normal(size=self.nx)
            x = x + noise

        ys = np.array([ y.full().reshape(-1) for y in Ys ])
        ys_true = np.array([ y.full().reshape(-1) for y in Ytrues ])
        return ys, ys_true
  
    def augment_model(self):
        x = misc.sym("x", self.nx)
        u = misc.sym("u", self.nu)
        alpha = misc.sym("a", self.nalpha)

        xplus = self.Fdiscr(x, u, alpha)
        y = self.G(x)

        d = misc.sym("d", self.ny)
        dplus = d
        yaug = y + d
        self.Fdiscr = ca.Function(
            "Faugmeted", [ca.vertcat(x, d), u, alpha], [ca.vertcat(xplus, dplus)]
        )
        self.G = ca.Function("Gaugmented", [ca.vertcat(d, x)], [yaug])
        self.get_dim()

    def draw(self, rng, alphascale=1, betascale=1):
        for i in range(10):
            alpha_true = rng.random(self.nalpha)* alphascale # choose a "true parameter randomly"
            beta_true = rng.random(self.nbeta)* betascale # choose a "true parameter randomly"
            if self.feasible(alpha_true, beta_true):
                return alpha_true, beta_true
        raise ValueError("Failed to draw feasible parameter")

    def gradient_covariances_fn(self):
        beta = misc.sym("betatemp", self.nbeta)
        Q, R = self.get_QR(beta)
        dQ_fn = ca.Function("dQ", [beta], [ca.jacobian(Q, beta)])
        dR_fn = ca.Function("dR", [beta], [ca.jacobian(R, beta)])
        return dQ_fn, dR_fn

class ProblemParser:
    def __init__(self, model, list_ys, list_us, x0, P0, L2pen=0., lti=False, no_u=False, idx_start=0):
        self.model = model
        if type(list_ys) is not list:
            list_ys = [list_ys]
            list_us = [list_us]
        self.ys_tot = list_ys 
        self.us_tot = list_us
        n_data = len(list_ys)
        if type(x0) is not list:
            self.x0 = [x0] * n_data
            self.P0 = [P0] * n_data
        else:
            self.x0 = x0
            self.P0 = P0
        self.L2pen = L2pen
        self.lti = lti
        self.no_u = no_u
        self.idx_start = idx_start # index of the first data point to consider in the cost function
        self.nx = self.model.nx
        self.ny = self.model.ny
        self.nu = self.model.nu
        self.nalpha = self.model.nalpha
        self.nbeta = self.model.nbeta
        
        self.cut()
    
    def cut(self, N=None):
        self.list_N = []
        self.us, self.ys = [], []
        Ny = N
        if Ny is not None:
            Ny = Ny + 1
        for (us_long, ys_long) in zip(self.us_tot, self.ys_tot):
            us = us_long[:N]
            ys = ys_long[:Ny]
            self.us.append(us)
            self.ys.append(ys)
            self.list_N.append(len(us))

    def kalman(self, alpha, beta, data_ind=0, save_pred=False, save_Pest=False):
        x_pred = self.x0[data_ind].copy()
        P_pred = self.P0[data_ind].copy()
        N = self.list_N[data_ind]
        us = self.us[data_ind]
        ys = self.ys[data_ind]
        xs_est, ys_est = np.empty((N+1, self.nx)), np.empty((N+1, self.ny))
        innovs = np.empty((N+1, self.ny))
        if save_pred:
            xs_pred, Ps_pred, Ks = (
                np.empty((N + 1, self.nx)),
                np.empty((N + 1, self.nx, self.nx)),
                np.empty((N, self.nx, self.ny))
            )
            Ps_pred[0, :, :] = P_pred[:, :]
            xs_pred[0, :] = x_pred
        if save_Pest:
            Ps_est = np.empty((N + 1, self.nx, self.nx))
            Ps_est[0, :, :] = P_pred[:, :]
        Q, R = self.model.get_QR(beta)
        Q = Q.full()
        R = R.full()
        dF = self.model.Fdiscr.jacobian()
        for k in range(N+1):
            C = self.model.G.jacobian()(x_pred, 0).full()
            M = LA.inv(C @ P_pred @ C.T + R)
            K = P_pred @ C.T @ M
            innovs[k, :] = ys[k] - self.model.G(x_pred).full().squeeze()
            x_est = x_pred + K @ innovs[k, :]
            P_est = P_pred - K @ C @ P_pred
            P_est = misc.symmetrize(P_est)
            xs_est[k, :] = x_est
            ys_est[k, :] = self.model.G(x_est).full().squeeze()
            if save_Pest:
                Ps_est[k, :, :] = P_est

            if k < N:
                x_pred = self.model.Fdiscr(x_est, us[k], alpha).full().squeeze()
                A = misc.select_jac( dF(x_est, us[k], alpha, 0), self.model.nx).full()
                P_pred = A @ P_est @ A.T + Q
                # print(k, (M @ innov).T @ innov - np.log(LA.det(M)))  # printing
                if save_pred:
                    Ps_pred[k + 1, :, :] = P_pred[:, :]
                    xs_pred[k + 1, :] = x_pred
                    Ks[k, :, :] = K[:, :]

        to_return: dict[str, np.ndarray] = {
            "xs_est": xs_est,
            "ys_est": ys_est
        }
        if save_pred:
            to_return["xs_pred"] = xs_pred
            to_return["Ps_pred"] = Ps_pred
            to_return["Ks"] = Ks
            to_return["innovs"] = innovs
        elif save_Pest:
            to_return["Ps_est"] = Ps_est
        return to_return
        
    def predictions(self, alpha, beta, npred, data_ind=0, Npred=None, Spred=False):
        dict_kalman = self.kalman(alpha, beta, data_ind=data_ind, save_Pest=True)
        xs_est, ys_est, Ps_est = dict_kalman["xs_est"], dict_kalman["ys_est"], dict_kalman["Ps_est"]

        t_pred, y_pred = self.model.sim_predictions(self.us[data_ind], xs_est, alpha, npred, Npred=Npred)

        dict_to_return: dict[str, np.ndarray] = {
            "t_pred": t_pred,
            "y_pred": y_pred,
            "ys_est": ys_est
        }
        if Spred:
            S_pred = self.model.sim_predictions_S(self.us[data_ind], Ps_est, alpha, beta, npred, Npred=Npred)
            dict_to_return["S_pred"] = S_pred
        return dict_to_return

    def error_prediction(self, alpha, beta, npred, Npred):
        dict_pred = self.predictions(alpha, beta, npred, Npred=Npred)
        idx_preds, y_preds = dict_pred["t_pred"], dict_pred["y_pred"]
        error = 0.
        for idx_pred, y_pred in zip(idx_preds, y_preds):
            ys_true = self.ys[idx_pred]

            r = ys_true - y_pred
            L2_seq = np.sqrt(np.sum( r**2, axis=1 ))
            assert np.all(L2_seq <= 1e10 ), "something is really too big"
            error = error + L2_seq.mean()
        # print("error prediction = {:.2e}".format(error / len(y_preds)))
        return error / len(y_preds)

    def value(self, alpha, beta, data_ind=0, formulation="MLE"):
        dict_kalman = self.kalman(alpha, beta, data_ind=data_ind, save_pred=True)
        xs, Ps, innovs  = dict_kalman["xs_pred"], dict_kalman["Ps_pred"], dict_kalman["innovs"]
        val = 0.
        N = self.list_N[data_ind]
        Q, R = self.model.get_QR(beta)
        for i in range(N+1):
            C = self.model.G.jacobian()(xs[i], 0).full()
            S = C @ Ps[i] @ C.T + R
            if formulation == "MLE":
                st_cost = (np.linalg.inv(S) @ innovs[i]) @ innovs[i] + np.log( np.linalg.det(S)  )
            elif formulation == "PredErr":
                st_cost = innovs[i] @ innovs[i]
            val = val + st_cost
        return val

    def solve(self, alpha0, beta0, formulation, algorithm, opts={}, verbose=False, rescale=None):
        if rescale is None:
            rescale = (formulation == "PredErr")
        if algorithm == "SQP":
                from .kalmanSQP import OPTKF
                eqconstr = rescale & (formulation == "PredErr")
                optkalman = OPTKF(self, formulation, opts=opts, rescale=rescale, verbose=verbose, eqconstr=eqconstr)
                alpha, beta, stats = \
                    optkalman.SQP_kalman(alpha0, beta0)
        elif algorithm == "IPOPT":
            clean_opts_for_ipopt(opts)
            from .kalmanIPOPT import nlp_kalman_solve
            if verbose:
                alpha, beta, stats = nlp_kalman_solve(self, alpha0, beta0, formulation, opts=opts, rescale=rescale)
            else:
                stdout = open('nul', 'w')
                with contextlib.redirect_stdout(stdout):
                    alpha, beta, stats = nlp_kalman_solve(self, alpha0, beta0, formulation, opts=opts, rescale=rescale)
        else:
            raise ValueError("Algorithm {} is unknown. Choose between 'SQP' or 'IPOPT'".format(algorithm))
            

        return alpha, beta, stats

    def info(self, alpha, beta, stats, problemTest, true_param=None, Npred=30, npred=3):
        dicti = {}
        if true_param is not None:
            alpha_true, beta_true = true_param
            dalpha, dbeta = norm(alpha_true - alpha), norm(beta_true - beta)
            dicti["L2distance"] = dalpha + dbeta
        dicti["alpha"] = alpha.copy()
        dicti["beta"] = beta.copy()
        dicti["rtime"] = stats["rtime"]
        dicti["vTrain"] = self.value(alpha, beta)
        dicti["vTest"] = problemTest.value(alpha, beta)
        # dicti["eTrain"] = self.error_prediction(alpha, beta, npred, Npred=Npred)
        dicti["eTest"] = problemTest.error_prediction(alpha, beta, npred, Npred=Npred)
        return dicti

def clean_opts_for_ipopt(opts):
    keys = [
        "maxiter",
        "pen_step",
        "tol.kkt",
        "tol.direction",
        "rtol.cost_decrease",
        "globalization.maxiter",
        "globalization.beta",
        "globalization.gamma",
        "einsum"
    ]
    for key in keys:
        if key in opts.keys():
            del opts[key]