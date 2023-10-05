import casadi as ca #type: ignore
import numpy as np # type: ignore
import numpy.linalg as LA # type: ignore
import contextlib
from math import ceil
from numpy.linalg import norm # type: ignore
from scipy.linalg import sqrtm # type: ignore
from .misc import symmetrize, select_jac

class ModelParser:
    def __init__(self, Fdiscr, G, Q_fn, R_fn):
        self.Fdiscr = Fdiscr
        self.G =  G
        self.Q_fn = Q_fn
        self.R_fn = R_fn
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
        A = select_jac(A_, self.nx).full()
        P_next = A @ P @ A.T + Q
        return symmetrize(P_next)

    def feasible(self, alpha, beta):
        return self.Ineq(alpha, beta).full().min() > 0.

    def trajectory(self, x0, us, alpha):
        x = x0.copy()
        xs = [x]
        for i in range(len(us)):
            x = self.Fdiscr(x, us[i], alpha)
            xs.append(x)
        return xs
    
    def trajectory_P(self, P0, us, alpha, Q):
        P = P0.copy()
        Ps = [P]
        for i in range(len(us)):
            P = self.propP(P, us[i], alpha, Q)
            Ps.append(P)
        return Ps

    def predictions(self, us, xs, alpha, npred, Npred=None):
        N = len(us)
        ts = np.arange(N+1, dtype=int)
        ys_pred, ts_pred = [], []
        if Npred is None:
            Npred = int(float(N) / npred)  # this is the number of time point in one interval of prediction
        interval = int(  float(N - Npred) / (npred - 1) )
        for i in range(npred):
            idx = i * interval
            xstart = xs[idx]
            upred = us[idx:idx+Npred]
            tpred = ts[idx:idx+Npred+1]
            xpred = self.trajectory(xstart, upred, alpha)
            ypred = np.array([ self.G(x).full().squeeze() for x in xpred])
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
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        alpha = ca.SX.sym("a", self.nalpha)

        xplus = self.Fdiscr(x, u, alpha)
        y = self.G(x)

        d = ca.SX.sym("d", y.shape[0])
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
        beta = ca.SX.sym("betatemp", self.nbeta)
        Q, R = self.get_QR(beta)
        dQ_fn = ca.Function("dQ", [beta], [ca.jacobian(Q, beta)])
        dR_fn = ca.Function("dR", [beta], [ca.jacobian(R, beta)])
        return dQ_fn, dR_fn

class ProblemParser:
    def __init__(self, model, ys, us, x0, P0, L2pen=0., lti=False, no_u=False):
        self.model = model
        self.x0 = x0
        self.P0 = P0
        self.ys_tot = ys 
        self.us_tot = us
        self.L2pen = L2pen
        self.lti = lti
        self.no_u = no_u

        self.N_tot = self.ys_tot.shape[0] - 1 
        self.nx = self.model.nx
        self.ny = self.model.ny
        self.nu = self.model.nu
        self.nalpha = self.model.nalpha
        self.nbeta = self.model.nbeta

        self.cut(self.N_tot)
    
    def cut(self, N):
        self.N = N
        self.ys = self.ys_tot[:N+1]
        self.us = self.us_tot[:N]

    def kalman(self, alpha, beta, save_pred=False, save_Pest=False):
        x_pred = self.x0.copy()
        P_pred = self.P0.copy()
        xs_est, ys_est = np.empty((self.N+1, self.nx)), np.empty((self.N+1, self.ny))
        innovs = np.empty((self.N+1, self.ny))
        if save_pred:
            xs_pred, Ps_pred, Ks = (
                np.empty((self.N + 1, self.nx)),
                np.empty((self.N + 1, self.nx, self.nx)),
                np.empty((self.N, self.nx, self.ny))
            )
            Ps_pred[0, :, :] = P_pred[:, :]
            xs_pred[0, :] = x_pred
        if save_Pest:
            Ps_est = np.empty((self.N + 1, self.nx, self.nx))
            Ps_est[0, :, :] = P_pred[:, :]
        Q, R = self.model.get_QR(beta)
        Q = Q.full()
        R = R.full()
        dF = self.model.Fdiscr.jacobian()
        for k in range(self.N+1):
            C = self.model.G.jacobian()(x_pred, 0).full()
            M = LA.inv(C @ P_pred @ C.T + R)
            K = P_pred @ C.T @ M
            innovs[k, :] = self.ys[k] - self.model.G(x_pred).full().squeeze()
            x_est = x_pred + K @ innovs[k, :]
            P_est = P_pred - K @ C @ P_pred
            P_est = symmetrize(P_est)
            xs_est[k, :] = x_est
            ys_est[k, :] = self.model.G(x_est).full().squeeze()
            if save_Pest:
                Ps_est[k, :, :] = P_est

            if k < self.N:
                x_pred = self.model.Fdiscr(x_est, self.us[k], alpha).full().squeeze()
                A = select_jac( dF(x_est, self.us[k], alpha, 0), self.model.nx).full()
                P_pred = A @ P_est @ A.T + Q
                # print(k, (M @ innov).T @ innov - np.log(LA.det(M)))  # printing
                if save_pred:
                    Ps_pred[k + 1, :, :] = P_pred[:, :]
                    xs_pred[k + 1, :] = x_pred
                    Ks[k, :, :] = K[:, :]

        if save_pred:
            return xs_est, ys_est, xs_pred, Ps_pred, Ks, innovs
        elif save_Pest:
            return xs_est, ys_est, Ps_est
        else:
            return xs_est, ys_est

    def scale(self, alpha, beta):
        _, _, xs, Ps, _, innovs = self.kalman(alpha, beta, save_pred=True)
        error = 0.
        Q, R = self.model.get_QR(beta)
        for i in range(self.N+1):
            C = self.model.G.jacobian()(xs[i], 0).full()
            M = np.linalg.inv(C @ Ps[i] @ C.T + R)
            error = error + (M @ innovs[i]) @ innovs[i]
        scale = error / ((self.N+1 ) * self.ny )
        return scale
    
    def predictions(self, alpha, beta, npred, Npred=None, Spred=False):
        xs_est, ys_est, Ps_est = self.kalman(alpha, beta, save_Pest=True)

        t_pred, y_pred = self.model.predictions(self.us, xs_est, alpha, npred, Npred=Npred)
        if Spred:
            S_pred = self.model.predictions_S(self.us, Ps_est, alpha, beta, npred, Npred=Npred)
            return t_pred, y_pred, ys_est, S_pred
        else:
            return t_pred, y_pred, ys_est

    def error_prediction(self, alpha, beta, npred, Npred):
        idx_preds, y_preds, ys_est = self.predictions(alpha, beta, npred, Npred=Npred)
        error = 0.
        for idx_pred, y_pred in zip(idx_preds, y_preds):
            ys_true = self.ys[idx_pred]

            r = ys_true - y_pred
            L2_seq = np.sqrt(np.sum( r**2, axis=1 ))
            assert np.all(L2_seq <= 1e10 ), "something is really too big"
            error = error + L2_seq.mean()
        # print("error prediction = {:.2e}".format(error / len(y_preds)))
        return error / len(y_preds)

    def value(self, alpha, beta, formulation="MLE"):
        _, _, xs, Ps, _, innovs = self.kalman(alpha, beta, save_pred=True)
        val = 0.
        Q, R = self.model.get_QR(beta)
        for i in range(self.N+1):
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
                optkalman = OPTKF(self, eqconstr=eqconstr)
                optkalman.prepare()
                optkalman.rinit()
                alpha, beta, stats = \
                    optkalman.SQP_kalman(alpha0, beta0, formulation, opts=opts, verbose=verbose, rescale=rescale)
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

    def gradient_dyna_fn(self):
        xzero = np.zeros(self.model.nx)
        alpha = ca.SX.sym("alpha temp", self.model.nalpha)
        A_syms, b_syms = [], []
        dA_syms, db_syms = [], []
        dF = self.model.Fdiscr.jacobian()
        for k in range(self.N):
            A = select_jac( dF(xzero, self.us[k], alpha, 0), self.model.nx)
            dA = ca.jacobian(A, alpha)
            A_syms.append(A)
            dA_syms.append(dA)
            if self.lti:
                break # no need to add more than the first one if lti
        for k in range(self.N):
            b = self.model.Fdiscr(xzero, self.us[k], alpha)
            db = ca.jacobian(b, alpha)
            b_syms.append(b)
            db_syms.append(db)
            if self.no_u:
                break
        C = self.model.G.jacobian()(xzero, 0).full()
        A_fns = [
            ca.Function("Afn{}".format(k), [alpha], [A]) for k, A in enumerate(A_syms)]
        b_fns = [
            ca.Function("bfn{}".format(k), [alpha], [b]) for k, b in enumerate(b_syms)]
        dA_fns = [
            ca.Function("dAfn{}".format(k), [alpha], [dA]) for k, dA in enumerate(dA_syms)]
        db_fns = [
            ca.Function("dbfn{}".format(k), [alpha], [db]) for k, db in enumerate(db_syms)]
        return A_fns, b_fns, C, dA_fns, db_fns

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