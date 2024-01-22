import casadi as ca #type: ignore
import numpy as np # type: ignore
from qpsolvers import Problem as Problem_QP
from qpsolvers import solve_problem as solve_qp
from time import time
from .misc import symmetrize, symmetrize, psd_inverse, select_jac

default_opts = {"maxiter":50,
          "pen_step":1e-5,
          "tol.kkt":1e-5,
          "tol.direction":1e-5,
          "rtol.cost_decrease":1e-5,
          "globalization.maxiter":20,
          "globalization.beta":0.8,
          "globalization.gamma":0.1,
          }

def myprod1(dA, B):
    # computes the derivative of A(t) B
    # return np.einsum("uvw,vx->uxw", dA, B)
    return B.T @ dA

def myprodvec(dA, x):
    # return np.einsum("uvw,v->uw", dA, x)
    return np.swapaxes(dA, 1, 2) @ x

def myprod2(A, dB):
    # computes the derivative of A B(t)
    # return np.einsum("uv,vxw->uxw", A, dB)
    return np.tensordot(A, dB, axes=(1,0))

def myprod3(A, dB):
    # computes the derivative of A B(t) A.T
    return np.einsum("uv,vxw,yx->uyw", A, dB, A)
    # return myprod2(myprod1(A, dB), A.T  )

class OPTKF:

    def __init__(self, problem, formulation, eqconstr=True, verbose=True, rescale=True, opts={}):
        self.model = problem.model
        self.L2pen = problem.L2pen
        self.idx_start = problem.idx_start
        self.formulation = formulation
        self.verbose = verbose
        self.do_rescale = rescale
        self.opts = self.complete_opts(opts)
        self.nM = int(self.model.ny * (self.model.ny + 1) / 2)
        self.nP = int(self.model.nx * (self.model.nx + 1) / 2)
        self.nalpha = self.model.nalpha
        self.nbeta = self.model.nbeta
        self.nab = self.nalpha + self.nbeta 
        self.make_constr(eqconstr=eqconstr)
        
        self.datas = []
        self.linearizing_fns = []
        for (x0, P0, ys, us) in zip(problem.x0, problem.P0, problem.ys, problem.us):
            data = {"x0":x0, "P0":P0, "ys":ys}
            A_fns, b_fns, C, dA_fns, db_fns = gradient_dyna_fn(self.model, us, lti=problem.lti, no_u=problem.no_u) # prepare linearizing functions
            lin_fns = {"A":A_fns, "b":b_fns, "dA":dA_fns, "db":db_fns, "C":C}
            self.linearizing_fns.append(lin_fns)
            self.datas.append(data)
        self.N_data = len(self.datas)

        self.dQ_fn, self.dR_fn = self.model.gradient_covariances_fn()
        self.inds_tri_M = np.tri(self.model.ny, k=-1, dtype=bool) # used to be np.bool
        
        self.rinit()
        if formulation not in ["MLE", "PredErr"]:
            raise ValueError("Formulation {} is unknown. Choose between 'MLE' or 'PredError'".format(formulation))

    def rinit(self):
        fn_names = [
            "linearized_fn","kalman_simulate",
            "prepare","cost_eval","derivatives","QP",
            "get_AbC","get_dAb","get_dQR","total"
            ]
        self.rtimes = {key:0 for key in fn_names}
        self.ncall = {key:0 for key in fn_names}

    def complete_opts(self, opts):
        options = default_opts.copy()
        for key, item in opts.items():
            options[key] = item
        return options
            
    def make_constr(self, eqconstr=True):
        alpha = ca.SX.sym("alpha_sym", self.model.nalpha)
        beta = ca.SX.sym("beta_sym", self.model.nbeta)
        ab = ca.vertcat(alpha, beta)
        # inequality constraints are user defined
        self.ineqconstraints = ca.Function("ineqcstr", [ab], [self.model.Ineq(alpha, beta)])
        self.derineqconstraints = self.ineqconstraints.jacobian()
        # equality constraint, only used in PredErr to remove one degree of freedom on beta
        Q, R = self.model.get_QR(beta)
        eq = ca.trace(Q) + ca.trace(R) - 1.
        if eqconstr:
            self.eqconstr = ca.Function("eqcstr", [ab], [eq])
            self.dereqconstraints = self.eqconstr.jacobian()
        else:
            self.eqconstr = None

    def make_lin_constr(self, ab):
        # inequality constraints
        Gconstr = - self.derineqconstraints(ab, 0).full()
        hconstr = self.ineqconstraints(ab).full().squeeze() + Gconstr @ ab
        ineq = (Gconstr, hconstr)
        # equatlity constraints
        if self.eqconstr is None:
            eq = None
        else:
            Aconstr = -self.dereqconstraints(ab, 0).full()
            bconstr = self.eqconstr(ab).full().squeeze() + Aconstr @ ab
            eq = (Aconstr, bconstr)
        return ineq, eq

    def get_AbC(self, alpha, data_ind):
        t0 = time()
        # note that in the case of lti these lists are single value
        # furthermore, in the case of no_u, bs is empty
        lin_fns = self.linearizing_fns[data_ind]
        As = [A_fn(alpha).full() for A_fn in lin_fns["A"]]
        bs = [ b_fn(alpha).full().squeeze() for b_fn in lin_fns["b"] ]
        self.rtimes["get_AbC"] += time() - t0
        self.ncall["get_AbC"] += 1
        return As, bs, lin_fns["C"]

    def get_dAbC(self, alpha, data_ind):
        t0 = time()
        lin_fns = self.linearizing_fns[data_ind]
        dAs = [
            np.swapaxes(
                dA_fn(alpha).full().reshape(self.model.nx, self.model.nx, self.model.nalpha),
                0, 1)
            for dA_fn in lin_fns["dA"]]
        dbs = [db_fn(alpha).full() for db_fn in lin_fns["db"]]
        self.rtimes["get_dAb"] += time() - t0
        self.ncall["get_dAb"] += 1
        return dAs, dbs, lin_fns["C"]
    
    def get_dQR(self, beta):
        t0 = time()
        dQ = self.dQ_fn(beta).full().reshape(self.model.nx, self.model.nx, self.model.nbeta)
        dR = self.dR_fn(beta).full().reshape(self.model.ny, self.model.ny, self.model.nbeta)
        dQ = symmetrize(dQ)
        dR = symmetrize(dR)
        self.rtimes["get_dQR"] += time() - t0
        self.ncall["get_dQR"] += 1
        return dQ, dR

    # functions that look at only one data serie
    def kalman_simulate(self, alpha, beta, data_ind):
        t0 = time()
        ys = self.datas[data_ind]["ys"]
        N = len(ys)
        Ls = np.empty( (N, self.model.nx, self.model.ny) )
        Ms = np.empty( (N, self.model.ny, self.model.ny) )
        Ss = np.empty( (N, self.model.ny, self.model.ny) )
        Ps = np.empty( (N+1, self.model.nx, self.model.nx) )
        logdetS = 0.
        es = np.empty( (N, self.model.ny) )
        xs = np.empty( (N+1, self.model.nx) )
        As, bs, C = self.get_AbC(alpha, data_ind)
        if len(As) == 1:
            As = As * N
            bs = bs * N
        Q, R = self.model.get_QR(beta)
        Q = Q.full()
        R = R.full()
        xs[0] = self.datas[data_ind]["x0"]
        Ps[0] = self.datas[data_ind]["P0"]
        for k in range(N):
            Ss[k], logdetSk, Ms[k], Ls[k], es[k], xs[k+1], Ps[k+1] = \
                kalman_step(
                    As[k], bs[k], C, Q, R,
                    Ps[k], xs[k], ys[k],
                    self.inds_tri_M)
            if k >= self.idx_start:
                logdetS = logdetS + logdetSk
        aux = {
               "P":Ps, "L":Ls, "M":Ms, "S":Ss, "logdetS":logdetS,
               "A" : As, "b": bs,
               "e" : es, "x":xs
        }
        self.rtimes["kalman_simulate"] += time() - t0
        self.ncall["kalman_simulate"] += 1
        return aux

    def derivatives(self, alpha, beta, aux, data_ind):
        t0 = time()
        gradient, hessian = 0., 0.
        dx_dalpha, dP_dalpha = np.zeros((self.model.nx, self.model.nalpha)), np.zeros((self.model.nx, self.model.nx, self.model.nalpha))
        dx_dbeta, dP_dbeta = np.zeros((self.model.nx, self.model.nbeta)), np.zeros((self.model.nx, self.model.nx, self.model.nbeta))
        dAs, dbs, C = self.get_dAbC(alpha, data_ind)
        dQ, dR = self.get_dQR(beta)
        N = len(aux["e"])
        if len(dAs) == 1:
            dAs = dAs * N
            dbs = dbs * N
        for k in range(N):
            e, M, S  = aux["e"][k], aux["M"][k], aux["S"][k]
            dM_dalpha, de_dalpha, dP_dalpha, dx_dalpha = \
                kalman_step_dalpha(dx_dalpha, dP_dalpha, dAs[k], dbs[k],
                                   e, M, S, C,
                                   aux["x"][k], aux["L"][k], aux["P"][k], aux["A"][k])
            dM_dbeta, de_dbeta, dP_dbeta, dx_dbeta = \
                kalman_step_dbeta(dx_dbeta, dP_dbeta, dQ, dR, 
                                    e, M, S, C,
                                    aux["x"][k], aux["L"][k], aux["P"][k], aux["A"][k])
            
            de_dab = np.concatenate([de_dalpha, de_dbeta], axis=-1)
            dM_dab = np.concatenate([dM_dalpha, dM_dbeta], axis=-1)
            if self.formulation == "PredErr":
                gradient_k = 2 * de_dab.T @ e
                hessian_k = 2 * de_dab.T @ de_dab
            else:
                trSdM_k = np.sum( dM_dab * S[...,np.newaxis] , axis=(0,1) )
                dMe = (e[np.newaxis, :] @ dM_dab)[:, 0, :]
                Mde = M @ de_dab
                gradient_k = (dMe + 2 * Mde).T @ e - trSdM_k
                hessian_k = 2 * np.linalg.multi_dot([ (dMe+Mde).T, S, dMe+Mde  ])
            if k >= self.idx_start:
                gradient = gradient + gradient_k
                hessian = hessian + hessian_k
        delete_unecessary(aux)
        self.rtimes["derivatives"] += time() - t0
        self.ncall["derivatives"] += 1
        return gradient, hessian
    
    def rescale(self, auxs, cost):
        s = 0.
        dim = 0.
        for aux in auxs:
            es = aux["e"][self.idx_start:]
            Ms = aux["M"][self.idx_start:]
            dim = dim + len(es) * self.model.ny
            s = s + np.sum(es[:, np.newaxis] @ Ms @ es[..., np.newaxis]) 
        lamb = s / dim
        if self.formulation == "MLE":
            cost = dim *(1 + np.log(lamb)) + (cost - s) # (cost - s) is the logdetS, s / lamb is dim
        for aux in auxs:
            dimension = len(aux["e"][self.idx_start:]) * self.model.ny
            aux["M"] = aux["M"] / lamb
            aux["P"] = aux["P"] * lamb
            aux["S"] = aux["S"] * lamb
            aux["logdetS"] = aux["logdetS"] + dimension * np.log(lamb)
        return lamb, auxs, cost
    
    # function that looks at all data series
    def derivatives_all(self, alpha, beta, auxs):
        gradient, hessian = 0., 0.
        for data_ind in range(self.N_data):
            gradient_i, hessian_i = self.derivatives(alpha, beta, auxs[data_ind], data_ind)
            gradient = gradient + gradient_i
            hessian = hessian + hessian_i
        return gradient, hessian
    
    def evaluate(self, alpha, beta):
        # everytime one computes the cost, one also computes the simulation
        t0 = time()
        auxs, value = [], 0.
        for data_ind in range(self.N_data):
            aux_i = self.kalman_simulate(alpha, beta, data_ind)
            value_i = cost_eval(aux_i["M"][self.idx_start:], aux_i["e"][self.idx_start:], aux_i["logdetS"], alpha, beta, self.formulation, L2pen=self.L2pen)
            auxs.append(aux_i)
            value += value_i
        self.rtimes["cost_eval"] += time() - t0
        self.ncall["cost_eval"] += 1
        return auxs, value
    
    def eval_kkt_error(self, ab, gradient, multiplier):
        kkt_error =  gradient + self.derineqconstraints(ab, 0).full().T @ multiplier
        return np.sum( abs(kkt_error))

    def solveQP(self, Q, p, constr, eqconstr, x_start, solver="proxqp"):
        """
            solve the QP:
            minimize  0.5 *  x^T s*Q x  + s*p @ x
            s.t. G x <= h,  Ax = b
        """
        s = 0.5
        t0 = time()
        G, h = constr
        if eqconstr is None:
            A, b = None, None
        else: 
            A, b = eqconstr
        problem_QP = Problem_QP(s*Q, s*p, G, h, A, b)
        sol = solve_qp(problem_QP, initvals=x_start, solver=solver, verbose=False)
        x = sol.x
        lam = sol.z
        der = -(p + Q @ x_start) @ (x - x_start)
        self.rtimes["QP"] += time() - t0
        self.ncall["QP"] += 1
        # print(f"sol = {sol.x}, primal residual {sol.primal_residual()}")
        return x, lam, der
    
    def line_search(self, alpha0, beta0, alpha1, beta1, der, cost, maxiter, gamma, b):
        tau = 1.
        if der < 0.:
            return None, 0., cost
        for i_glob in range(maxiter):
            alpha_middle = (1 - tau) * alpha0 + tau * alpha1
            beta_middle = (1 - tau) * beta0 + tau * beta1
            if np.all(beta_middle >= 0.):
                auxs_middle, cost_middle = self.evaluate(alpha_middle, beta_middle)
                condition = (cost - cost_middle) / tau > gamma * der
                if condition:
                        if self.verbose: print(f" tau = {tau}")
                        return auxs_middle, tau, cost_middle
            tau = tau * b
        if self.verbose: print("Line-search did not finish with tau = {}".format(tau))
        return None, 0., cost
    
    def get_termination(self, prev_cost, auxs_is_None, der, kkt_error, new_cost):
        if auxs_is_None:
            if der < 0.:
                return f"non-descending direction : derivative={der:.2e}"
            return "globalization.maxiter"     
        elif kkt_error < self.opts["tol.kkt"]:
            return f"tol.kkt : kkt error= {kkt_error:.2e}"
        elif der < self.opts["tol.direction"]:
            return f"tol.direction, derivative={der:.2e}"
        elif prev_cost - new_cost < max(abs(prev_cost),abs(new_cost)) * self.opts["rtol.cost_decrease"]:
            return f"rtol.cost_decrease, cost decrease = {prev_cost - new_cost:.2e}"
        else:
            return None

    def stepSQP(self, alpha, beta, prev_cost, auxs):
        gradient, hessian = self.derivatives_all(alpha, beta, auxs)
        ab = np.concatenate([alpha, beta])
        linconstr, lin_eqconstr = self.make_lin_constr(ab)
        Q, p = prepareQP(ab, gradient, hessian, pen_step=self.opts["pen_step"], L2reg=self.L2pen)
        ab_next, multiplier, der = self.solveQP(Q, p, linconstr, lin_eqconstr, ab)
        alpha_next, beta_next = ab_next[:self.nalpha], ab_next[self.nalpha:]
        kkt_error = self.eval_kkt_error(ab, gradient, multiplier)
        auxs, tau, new_cost = self.line_search(alpha, beta, alpha_next, beta_next, der, prev_cost,
                                            self.opts["globalization.maxiter"], self.opts["globalization.gamma"], self.opts["globalization.beta"])
        alpha = (1 - tau) * alpha + tau * alpha_next
        beta = (1 - tau) * beta + tau * beta_next
        if self.do_rescale and (auxs is not None):
            scale, auxs, new_cost = self.rescale(auxs, new_cost)
            beta = beta * scale
        termination = self.get_termination(prev_cost, (auxs is None), der, kkt_error, new_cost)
        return alpha, beta, new_cost, auxs, termination

    def SQP_kalman(self, alpha0, beta0):
        t0 = time()
        alpha = alpha0.copy()
        beta = beta0.copy()
        auxs, cost = self.evaluate(alpha, beta)
        if self.do_rescale:
            scale, auxs, cost = self.rescale(auxs, cost)
            beta = beta * scale
        for j in range(self.opts["maxiter"]):
            alpha, beta, cost, auxs, termination  = self.stepSQP(alpha, beta, cost, auxs)
            if self.verbose:
                print(f"Iteration {j+1}, Current cost {cost:2e}")
            if termination is not None:
                break
        if termination is None:
            termination = "maxiter"
        if self.verbose:
            print("termination :", termination)
        self.rtimes["total"] += time() - t0
        self.ncall["total"] += 1
        stats =  {"termination":termination, "niter":j+1, "rtimes":self.rtimes, "ncall":self.ncall}
        stats["return_status"] = stats["termination"]
        return alpha, beta, stats

# utils

def gradient_dyna_fn(model, us, lti=False, no_u=False):
    xzero = np.zeros(model.nx)
    alpha = ca.SX.sym("alpha temp", model.nalpha)
    A_syms, b_syms = [], []
    dA_syms, db_syms = [], []
    dF = model.Fdiscr.jacobian()
    N = len(us)
    for k in range(N):
        A = select_jac( dF(xzero, us[k], alpha, 0), model.nx)
        dA = ca.jacobian(A, alpha)
        A_syms.append(A)
        dA_syms.append(dA)
        if lti:
            break # no need to add more than the first one if lti
    for k in range(N):
        b = model.Fdiscr(xzero, us[k], alpha)
        db = ca.jacobian(b, alpha)
        b_syms.append(b)
        db_syms.append(db)
        if no_u:
            break
    C = model.G.jacobian()(xzero, 0).full()
    A_fns = [
        ca.Function("Afn{}".format(k), [alpha], [A]) for k, A in enumerate(A_syms)]
    b_fns = [
        ca.Function("bfn{}".format(k), [alpha], [b]) for k, b in enumerate(b_syms)]
    dA_fns = [
        ca.Function("dAfn{}".format(k), [alpha], [dA]) for k, dA in enumerate(dA_syms)]
    db_fns = [
        ca.Function("dbfn{}".format(k), [alpha], [db]) for k, db in enumerate(db_syms)]
    return A_fns, b_fns, C, dA_fns, db_fns

def delete_unecessary(aux):
    del aux["x"]
    del aux["S"]
    del aux["P"]
    del aux["A"]
    del aux["b"]
    del aux["L"]

def cost_eval(Ms, es, logdet, alpha, beta, formulation, L2pen=0.):
    if formulation=="MLE":
        value1 = np.sum(es[:, np.newaxis] @ Ms @ es[..., np.newaxis])
        value = value1 + logdet
    elif formulation=="PredErr":
        value = np.sum( es**2 )
    value = value + L2pen * (np.sum( alpha**2 ) + np.sum(beta**2 ))
    return value

def prepareQP(xbar, gradient, hessian, pen_step=0, L2reg=0.):
    I = np.eye(len(xbar))
    Q = hessian + 2* pen_step * I
    p = gradient - Q @ xbar
    Q = Q + 2 * L2reg * I
    return Q, p
    
def kalman_step(A, b, C, Q, R, P, x, y, inds_tri):
    PC = P @ C.T
    S = C @ PC + R
    M, logdetS = psd_inverse(S, inds_tri)
    L = A @ (PC @ M)
    e = y - C @ x
    Pplus = A @ P @ A.T - L @ S @ L.T + Q
    xplus = A @ x + L @ e + b
    return S, logdetS, M, L, e, xplus, Pplus

def kalman_step_dalpha(dx_dalpha, dP_dalpha, dA, db, e, M, S, C, x, L, P, A):
    dS_dalpha = myprod3(C, dP_dalpha)
    dM_dalpha = - myprod3(M, dS_dalpha)
    de_dalpha = - C @ dx_dalpha
    dL_dalpha = myprod2(A @ (P @ C.T), dM_dalpha) + myprod1(myprod2(A, dP_dalpha), C.T @ M) + myprod1(dA, P @ (C.T @ M))
    dPplus_dalpha = myprod3(A, dP_dalpha) - myprod3(L, dS_dalpha) -  2 * myprod1(dL_dalpha, S @ L.T) + 2 * myprod1(dA, P @ A.T)
    dPplus_dalpha = symmetrize(dPplus_dalpha)
    dxplus_dalpha = A @ dx_dalpha + myprodvec(dL_dalpha, e) + L @ de_dalpha  + myprodvec(dA, x) + db
    return dM_dalpha, de_dalpha, dPplus_dalpha, dxplus_dalpha

def kalman_step_dbeta(dx_dbeta, dP_dbeta, dQ, dR, e, M, S, C, x, L, P, A):
    dS_dbeta = myprod3(C, dP_dbeta) + dR
    dM_dbeta = - myprod3(M, dS_dbeta)
    de_dbeta = - C @ dx_dbeta
    dL_dbeta = myprod2(A @ (P @ C.T), dM_dbeta) + myprod1(myprod2(A, dP_dbeta), C.T @ M)
    dPplus_dbeta = myprod3(A, dP_dbeta) - myprod3(L, dS_dbeta) -  2 * myprod1(dL_dbeta, S @ L.T) + dQ
    dPplus_dbeta = symmetrize(dPplus_dbeta)
    dxplus_dbeta = A @ dx_dbeta + myprodvec(dL_dbeta, e)+ L @ de_dbeta
    return dM_dbeta, de_dbeta, dPplus_dbeta, dxplus_dbeta