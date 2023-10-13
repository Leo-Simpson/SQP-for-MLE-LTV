import casadi as ca #type: ignore
import numpy as np # type: ignore
from time import time
from .misc import symmetrize, symmetrize, psd_inverse

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
        self.problem = problem
        self.N = problem.N
        self.formulation = formulation
        self.verbose = verbose
        self.rescale = rescale
        self.opts = self.complete_opts(opts)
        self.nM = int(self.model.ny * (self.model.ny + 1) / 2)
        self.nP = int(self.model.nx * (self.model.nx + 1) / 2)
        self.nalpha = self.model.nalpha
        self.nbeta = self.model.nbeta
        self.nab = self.nalpha + self.nbeta 
        
        self.make_constr(eqconstr=eqconstr)
        # prepare linearizing functions
        self.A_fns, self.b_fns, self.C, self.dA_fns, self.db_fns = self.problem.gradient_dyna_fn()
        self.dQ_fn, self.dR_fn = self.model.gradient_covariances_fn()
        self.inds_tri_M = np.tri(self.model.ny, k=-1, dtype=bool) # used to be np.bool
        
        self.rinit()
        if formulation not in ["MLE", "PredErr"]:
            raise ValueError("Formulation {} is unknown. Choose between 'MLE' or 'PredError'".format(formulation))

    def rinit(self):
        fn_names = [
            "linearized_fn","kalman_simulate",
            "prepare","cost_eval","cost_derivatives","QP",
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

    def get_AbC(self, alpha):
        t0 = time()
        # note that in the case of lti these lists are single value
        # furthermore, in the case of no_u, bs is empty
        As = [A_fn(alpha).full() for A_fn in self.A_fns]
        bs = [ b_fn(alpha).full().squeeze() for b_fn in self.b_fns ]
        if len(As) == 1:
            As = As * self.N
        if len(bs) == 1:
            bs = bs * self.N
        self.rtimes["get_AbC"] += time() - t0
        self.ncall["get_AbC"] += 1
        return As, bs

    def get_dAb(self, alpha):
        t0 = time()
        dAs = [
            np.swapaxes(
                dA_fn(alpha).full().reshape(self.model.nx, self.model.nx, self.model.nalpha),
                0, 1)
            for dA_fn in self.dA_fns]
        dbs = [db_fn(alpha).full() for db_fn in self.db_fns]

        if len(dAs) == 1:
            dAs = dAs * self.N
        if len(dbs) == 1:
            dbs = dbs * self.N
        self.rtimes["get_dAb"] += time() - t0
        self.ncall["get_dAb"] += 1
        return dAs, dbs
    
    def get_dQR(self, beta):
        t0 = time()
        dQ = self.dQ_fn(beta).full().reshape(self.model.nx, self.model.nx, self.model.nbeta)
        dR = self.dR_fn(beta).full().reshape(self.model.ny, self.model.ny, self.model.nbeta)
        dQ = symmetrize(dQ)
        dR = symmetrize(dR)
        self.rtimes["get_dQR"] += time() - t0
        self.ncall["get_dQR"] += 1
        return dQ, dR

    def kalman_simulate(self, alpha, beta):
        t0 = time()
        Ls = np.empty( (self.N, self.model.nx, self.model.ny) )
        Ms = np.empty( (self.N, self.model.ny, self.model.ny) )
        Ss = np.empty( (self.N, self.model.ny, self.model.ny) )
        Ps = np.empty( (self.N+1, self.model.nx, self.model.nx) )
        logdetS = 0.
        es = np.empty( (self.N, self.model.ny) )
        xs = np.empty( (self.N+1, self.model.nx) )
        As, bs = self.get_AbC(alpha)
        Q, R = self.model.get_QR(beta)

        xs[0] = self.problem.x0
        Ps[0] = self.problem.P0
        for k in range(self.N):
            Ss[k], logdetSk, Ms[k], Ls[k], es[k], xs[k+1], Ps[k+1] = \
                kalman_step(
                    As[k], bs[k], self.C, Q, R,
                    Ps[k], xs[k], self.problem.ys[k],
                    self.inds_tri_M)

            logdetS = logdetS + logdetSk
        self.aux = {
               "P":Ps, "L":Ls, "M":Ms, "S":Ss, "logdetS":logdetS,
               "A" : As, "b": bs,
               "e" : es, "x":xs
        }
        self.rtimes["kalman_simulate"] += time() - t0
        self.ncall["kalman_simulate"] += 1

    def cost_derivatives(self, alpha, beta):
        t0 = time()
        gradient, hessian = 0., 0.
        dx_dalpha, dP_dalpha = np.zeros((self.model.nx, self.model.nalpha)), np.zeros((self.model.nx, self.model.nx, self.model.nalpha))
        dx_dbeta, dP_dbeta = np.zeros((self.model.nx, self.model.nbeta)), np.zeros((self.model.nx, self.model.nx, self.model.nbeta))
        dAs, dbs = self.get_dAb(alpha)
        dQ, dR = self.get_dQR(beta)
        for k in range(self.N):
            e, M, S  = self.aux["e"][k], self.aux["M"][k], self.aux["S"][k]
            dM_dalpha, de_dalpha, dP_dalpha, dx_dalpha = \
                kalman_step_dalpha(dx_dalpha, dP_dalpha, dAs[k], dbs[k],
                                   e, M, S, self.C,
                                   self.aux["x"][k], self.aux["L"][k], self.aux["P"][k], self.aux["A"][k])
            dM_dbeta, de_dbeta, dP_dbeta, dx_dbeta = \
                kalman_step_dbeta(dx_dbeta, dP_dbeta, dQ, dR, 
                                    e, M, S, self.C,
                                    self.aux["x"][k], self.aux["L"][k], self.aux["P"][k], self.aux["A"][k])
            
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
            gradient = gradient + gradient_k
            hessian = hessian + hessian_k
        self.delete_unecessary()
        self.rtimes["cost_derivatives"] += time() - t0
        self.ncall["cost_derivatives"] += 1
        return gradient, hessian
      
    def delete_unecessary(self):
        del self.aux["x"]
        del self.aux["S"]
        del self.aux["P"]
        del self.aux["A"]
        del self.aux["b"]
        del self.aux["L"]

    def cost(self, alpha, beta):
        t0 = time()
        value = cost_eval(self.aux["M"], self.aux["e"], self.aux["logdetS"], alpha, beta, self.formulation, L2pen=self.problem.L2pen)
        self.rtimes["cost_eval"] += time() - t0
        self.ncall["cost_eval"] += 1
        return value
    
    def eval_kkt_error(self, ab, gradient, multiplier):
        kkt_error =  gradient + self.derineqconstraints(ab, 0).full().T @ multiplier
        return np.sum( abs(kkt_error))

    def solveQP(self, Q, p, constr, eqconstr, x_start):
        """
            solve the QP:
            minimize  0.5 *  x^T Q x  + p @ x
            s.t. G x <= h,  Ax = b
        """
        t0 = time()
        from cvxopt import matrix, solvers #type: ignore
        solvers.options['show_progress'] = False
        G, h = constr
        if eqconstr is None:
            sol = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h), initvals=matrix(x_start))
        else: 
            A, b = eqconstr
            sol = solvers.qp(matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b), initvals=matrix(x_start))
        x = np.squeeze(np.array(sol["x"]))  
        lam = np.squeeze(np.array(sol["z"]))
        der = -(p + Q @ x_start) @ (x - x_start)
        self.rtimes["QP"] += time() - t0
        self.ncall["QP"] += 1
        return x, lam, der
    
    def scale(self):
        dimension = self.N * self.model.ny
        lamb = np.sum(self.aux["e"][:, np.newaxis] @ self.aux["M"] @ self.aux["e"][..., np.newaxis]) / dimension
        self.aux["M"] = self.aux["M"] / lamb # TODO
        self.aux["P"] = self.aux["P"] * lamb
        self.aux["S"] = self.aux["S"] * lamb
        self.aux["logdetS"] = self.aux["logdetS"] + dimension * np.log(lamb)
        return lamb

    def stepSQP(self, alpha, beta, prev_cost):
        gradient, hessian = self.cost_derivatives(alpha, beta)
        ab = np.concatenate([alpha, beta])
        linconstr, lin_eqconstr = self.make_lin_constr(ab)
        Q, p = prepareQP(ab, gradient, hessian, pen_step=self.opts["pen_step"], L2reg=self.problem.L2pen)
        ab_next, multiplier, der = self.solveQP(Q, p, linconstr, lin_eqconstr, ab)
        alpha_next, beta_next = ab_next[:self.nalpha], ab_next[self.nalpha:]
        kkt_error = self.eval_kkt_error(ab, gradient, multiplier)
        termination = None
        if der < 0.:
            termination = f"non-descending direction : derivative={der:.2e}"
        else:
            tau, new_cost = self.line_search(alpha, beta, alpha_next, beta_next, der, prev_cost,
                                            self.opts["globalization.maxiter"], self.opts["globalization.gamma"], self.opts["globalization.beta"])
            if kkt_error < self.opts["tol.kkt"]:
                termination = f"tol.kkt : kkt error= {kkt_error:.2e}"
            elif der < self.opts["tol.direction"]:
                termination = f"tol.direction, derivative={der:.2e}"
            elif prev_cost - new_cost < max(abs(prev_cost),abs(new_cost)) * self.opts["rtol.cost_decrease"]:
                termination = f"rtol.cost_decrease, cost decrease = {prev_cost - new_cost:.2e}"
            elif tau is None:
                termination = "globalization.maxiter"
                tau = 0.
        alpha = (1 - tau) * alpha + tau * alpha_next
        beta = (1 - tau) * beta + tau * beta_next
        if self.rescale:
            beta = beta * self.scale()
        return alpha, beta, new_cost, termination

    def SQP_kalman(self, alpha0, beta0):
        t0 = time()
        alphaj = alpha0.copy()
        betaj = beta0.copy()
        self.kalman_simulate(alphaj, betaj)
        cost = self.cost(alphaj, betaj)
        if self.rescale:
            betaj = betaj * self.scale()
        termination = "maxiter"
        for j in range(self.opts["maxiter"]):
            alphaj, betaj, cost, termination  = self.stepSQP(alphaj, betaj, cost)
            if self.verbose:
                print(f"Iteration {j+1}, Current cost {cost:2e}")
            if termination is not None:
                break
        if self.verbose:
            print("termination :", termination)
        self.rtimes["total"] += time() - t0
        self.ncall["total"] += 1
        stats =  {"termination":termination, "niter":j+1, "rtimes":self.rtimes, "ncall":self.ncall}
        stats["return_status"] = stats["termination"]
        return alphaj, betaj, stats

    def line_search(self, alpha0, beta0, alpha1, beta1, der, cost, maxiter, gamma, b):
        tau = 1.
        for i_glob in range(maxiter):
            alpha_middle = (1 - tau) * alpha0 + tau * alpha1
            beta_middle = (1 - tau) * beta0 + tau * beta1
            self.kalman_simulate(alpha_middle, beta_middle)
            cost_middle = self.cost(alpha_middle, beta_middle)
            condition = (cost - cost_middle) / tau > gamma * der and np.all(beta_middle >= 0.)
            if condition:
                    if self.verbose: print(f" tau = {tau}")
                    return tau, cost_middle
            tau = tau * b
        if self.verbose: print("Globalization did not finish with tau = {}".format(tau))
        return None, cost

    # def cost_derivatives_fd(self, alpha, beta, formulation, dalpha, dbeta):
    #     eps = 1e-5
    #     _, cost0 = self.cost(alpha, beta, formulation)
    #     _, cost_plus = self.cost(alpha+eps*dalpha, beta+eps*dbeta, formulation)
    #     der = (cost_plus - cost0)/eps
    #     return der
    # def verif(self, gradient, alpha, beta, formulation):
    #     dalpha = np.random.random(self.model.nalpha)
    #     dbeta = np.random.random(self.model.nbeta)

    #     ad_alpha = gradient[:self.model.nalpha] @ dalpha
    #     fd_alpha = self.cost_derivatives_fd(alpha, beta, formulation, dalpha, 0.)
        
    #     ad_beta = gradient[self.model.nalpha:] @ dbeta
    #     fd_beta = self.cost_derivatives_fd(alpha, beta, formulation, 0., dbeta)
        
    #     print(f"verif dalpha : fd = {fd_alpha:.2e}, ad = {ad_alpha:.2e}, dif = {(fd_alpha-ad_alpha):.2e}")
    #     print(f"verif dbeta : fd = {fd_beta:.2e}, ad = {ad_beta:.2e}, dif = {(fd_beta-ad_beta):.2e}")

# utils
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