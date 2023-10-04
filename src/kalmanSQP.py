import casadi as ca #type: ignore
import numpy as np # type: ignore
from time import time
from .misc import symmetrize, symmetrize, psd_inverse

default_opts = {"maxiter":50,
          "pen_step":1e-4,
          "tol.kkt":1e-6,
          "tol.direction":1e-6,
          "rtol.cost_decrease":1e-6,
          "globalization.maxiter":20,
          "globalization.beta":0.8,
          "globalization.gamma":0.1,
          }

class OPTKF:

    def __init__(self, problem, eqconstr=True):
        self.model = problem.model
        self.problem = problem
        self.N = problem.N
        self.make_constr(eqconstr=eqconstr)
        self.nM = int(self.model.ny * (self.model.ny + 1) / 2)
        self.nP = int(self.model.nx * (self.model.nx + 1) / 2)
        self.nab = self.model.nalpha + self.model.nbeta
        self.rinit()

    def rinit(self):
        fn_names = [
            "linearized_fn","kalman_simulate_matrices","kalman_simulate_states",
            "prepare","cost_eval","cost_derivatives","QP",
            "get_AbC","get_dAb","get_dQR","total"
            ]
        self.rtimes = {key:0 for key in fn_names}
        self.ncall = {key:0 for key in fn_names}

    def linearized_fn(self):
        t0 = time()
        self.A_fns, self.b_fns, self.C, self.dA_fns, self.db_fns = self.problem.gradient_dyna_fn()
        self.dQ_fn, self.dR_fn = self.model.gradient_covariances_fn()
        self.rtimes["linearized_fn"] += time() - t0
        self.ncall["linearized_fn"] += 1
            
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
        Gconstr = - self.derineqconstraints(ab, 0).full()
        hconstr = self.ineqconstraints(ab).full().squeeze() + Gconstr @ ab
        return Gconstr, hconstr

    def make_lin_eqconstr(self, ab):
        if self.eqconstr is None:
            return None
        Aconstr = -self.dereqconstraints(ab, 0).full()
        bconstr = self.eqconstr(ab).full().squeeze() + Aconstr @ ab
        return Aconstr, bconstr

    def prepare(self):
        t0 = time()
        self.linearized_fn()
        self.inds_tri_M = np.tri(self.model.ny, k=-1, dtype=np.bool)
        self.rtimes["prepare"] += time() - t0
        self.ncall["prepare"] += 1

    def complete_opts(self, opts):
        options = default_opts.copy()
        for key, item in opts.items():
            options[key] = item
        return options

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

    def kalman_simulate_matrices(self, alpha, beta):
        t0 = time()
        # symbol = type(alpha) is ca.SX
        K = np.empty( (self.N+1, self.model.nx, self.model.ny) )
        M = np.empty( (self.N+1, self.model.ny, self.model.ny) )
        S = np.empty( (self.N+1, self.model.ny, self.model.ny) )
        P = np.empty( (self.N+1, self.model.nx, self.model.nx) )
        logdetS = np.empty( (self.N+1) )
        P[0] = self.problem.P0
        As, bs = self.get_AbC(alpha)
        Q, R = self.model.get_QR(beta)
        for k in range(self.N+1):
            PC = P[k] @ self.C.T
            S[k] = self.C @ PC + R
            M[k], logdetS[k] = psd_inverse(S[k], self.inds_tri_M)
            K[k] = PC @ M[k]
            if k == self.N:
                break
            Pest = P[k] - K[k] @ PC.T
            P[k+1] = symmetrize( As[k] @ Pest @ As[k].T + Q )
                
        matrices = {
               "P":P, "K":K, "M":M, "S":S, "logdetS":logdetS,
               "A" : As, "b": bs
        }
        self.rtimes["kalman_simulate_matrices"] += time() - t0
        self.ncall["kalman_simulate_matrices"] += 1
        return matrices

    def kalman_simulate_states(self, matrices):
        t0 = time()
        # symbol = type(alpha) is ca.SX
        e = np.empty( (self.N+1, self.model.ny) )
        x = np.empty( (self.N+1, self.model.nx) )
        x[0] = self.problem.x0
        for k in range(self.N+1):
            e[k] = self.problem.ys[k] - self.C @ x[k]
            if k == self.N:
                break
            x[k+1] = matrices["A"][k] @ (x[k] + matrices["K"][k] @ e[k]) + matrices["b"][k]
        states = {"e" : e, "x":x}
        self.rtimes["kalman_simulate_states"] += time() - t0
        self.ncall["kalman_simulate_states"] += 1
        return states

    def kalman_simulate(self, alpha, beta):
        matrices = self.kalman_simulate_matrices(alpha, beta)
        states = self.kalman_simulate_states(matrices)
        return states, matrices
  
    def cost_derivatives(self, states, matrices, alpha, beta, formulation):
        t0 = time()
        gradient, hessian = 0., 0.
        dx_dalpha, dP_dalpha = np.zeros((self.model.nx, self.model.nalpha)), np.zeros((self.model.nx, self.model.nx, self.model.nalpha))
        dx_dbeta, dP_dbeta = np.zeros((self.model.nx, self.model.nbeta)), np.zeros((self.model.nx, self.model.nx, self.model.nbeta))
        dAs, dbs = self.get_dAb(alpha)
        dQ, dR = self.get_dQR(beta)
        for k in range(self.N+1):
            ek, Mk, Sk, C  = states["e"][k], matrices["M"][k], matrices["S"][k], self.C
            dS_dalpha = np.tensordot(C, C @ dP_dalpha, axes=(1,0))
            dS_dbeta = np.tensordot(C, C @ dP_dbeta, axes=(1,0)) + dR
            dM_dalpha = -np.tensordot(Mk, Mk @ dS_dalpha, axes=(1, 0))
            dM_dbeta = -np.tensordot(Mk, Mk @ dS_dbeta, axes=(1, 0))
            dM_dalpha = symmetrize(dM_dalpha)
            dM_dbeta = symmetrize(dM_dbeta)
            de_dalpha_k = - C @ dx_dalpha
            de_dbeta_k = - C @ dx_dbeta
            if k == self.N:
                break
            xk, Kk, Pk, A = states["x"][k], matrices["K"][k], matrices["P"][k], matrices["A"][k]
            
            PC = Pk @ C.T
            KC = Kk @ C

            xestk = xk + Kk @ ek
            Pestk = Pk - Kk @ PC.T

            dK_dalpha = np.tensordot(PC, dM_dalpha, axes=(1,0)) + (Mk @ C) @ dP_dalpha
            dK_dbeta = np.tensordot(PC, dM_dbeta, axes=(1,0)) + (Mk @ C) @ dP_dbeta
            
            dPest_dalpha = dP_dalpha - np.tensordot( KC, dP_dalpha, axes=(1,0)) -  PC @ dK_dalpha
            dPest_dbeta = dP_dbeta - np.tensordot( KC, dP_dbeta, axes=(1,0)) -  PC @ dK_dbeta
            dP_dalpha = np.tensordot(A, A @ dPest_dalpha, axes=(1, 0)) \
                + 2 * np.tensordot(Pestk @ A.T, dAs[k], axes=(0, 1))
            dP_dbeta = np.tensordot(A, A @ dPest_dbeta, axes=(1, 0)) + dQ
            
            dxest_dalpha = dx_dalpha + Kk @ de_dalpha_k + np.swapaxes(dK_dalpha, 1, 2) @ ek
            dxest_dbeta = dx_dbeta + Kk @ de_dbeta_k +  np.swapaxes(dK_dbeta, 1, 2) @ ek
            dx_dalpha = A @ dxest_dalpha + dbs[k] + np.swapaxes(dAs[k], 1, 2) @ xestk
            dx_dbeta = A @ dxest_dbeta

            dP_dalpha = symmetrize(dP_dalpha)
            dP_dbeta = symmetrize(dP_dbeta)

            de_dab_k = np.concatenate([de_dalpha_k, de_dbeta_k], axis=-1)
            dM_dab_k = np.concatenate([dM_dalpha, dM_dbeta], axis=-1)
            if formulation == "PredErr":
                gradient_k = 2 * de_dab_k.T @ ek
                hessian_k = 2 * de_dab_k.T @ de_dab_k
            else:
                trSdM_k = np.sum( dM_dab_k * Sk[...,np.newaxis] , axis=(0,1) )
                dMe = (ek[np.newaxis, :] @ dM_dab_k)[:, 0, :]
                Mde = Mk @ de_dab_k
                gradient_k = (dMe + 2 * Mde).T @ ek - trSdM_k
                hessian_k = 2 * np.linalg.multi_dot([ (dMe+Mde).T, Sk, dMe+Mde  ])
            gradient = gradient + gradient_k
            hessian = hessian + hessian_k
        self.delete_unecessary(states, matrices)
        self.rtimes["cost_derivatives"] += time() - t0
        self.ncall["cost_derivatives"] += 1
        return gradient, hessian
            
    def delete_unecessary(self, states, matrices):
        del states["x"]
        del matrices["S"]
        del matrices["P"]
        del matrices["A"]
        del matrices["b"]
        del matrices["K"]

    def cost(self, alpha, beta, formulation, states_and_matrices=None):
        t0 = time()
        if states_and_matrices is None:
            states, matrices = self.kalman_simulate(alpha, beta)
        else:
            states, matrices = states_and_matrices
        value = cost_eval(states, matrices, alpha, beta, formulation, L2pen=self.problem.L2pen)
        self.rtimes["cost_eval"] += time() - t0
        self.ncall["cost_eval"] += 1
        return states, matrices, value
    
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

    def SQP_kalman(self, alpha0, beta0, formulation, opts={}, verbose=True, path=False, rescale=True):
        if formulation not in ["MLE", "PredErr"]:
            raise ValueError("Formulation {} is unknown. Choose between 'MLE' or 'PredError'".format(formulation))
        t0 = time()
        options = self.complete_opts(opts)
        nalpha = self.model.nalpha
        alphaj = alpha0.copy()
        betaj = beta0.copy()
        states, matrices, cost = self.cost(alphaj, betaj, formulation)
        objective_scale = 1.
        tol_direction = objective_scale * options["tol.direction"]
        if path:
            alphas, betas = [], []
        for j in range(options["maxiter"]):
            if path:
                alphas.append(alphaj)
                betas.append(betaj)
            if rescale:
                betaj = betaj * self.scale(alphaj, betaj, states_and_matrices=(states, matrices))
            gradient, hessian = self.cost_derivatives(states, matrices, alphaj, betaj, formulation)
            ab = np.concatenate([alphaj, betaj])
            hessian_ = hessian + 2*options["pen_step"]* np.eye(len(ab))
            grad0 = gradient - hessian_ @ ab
            hessian_ = hessian_ + 2*self.problem.L2pen * np.eye(len(ab))
            linconstr = self.make_lin_constr(ab)
            lin_eqconstr = self.make_lin_eqconstr(ab)
            ab_next, multiplier, der = self.solveQP(hessian_, grad0, linconstr, lin_eqconstr, ab)
            alpha_next, beta_next = ab_next[:nalpha], ab_next[nalpha:]
            if np.any(beta_next<0.):
                print(f"beta {betaj}, betanext {beta_next}")
            kkt_error = self.eval_kkt_error(ab, gradient, multiplier)
            
            if kkt_error < options["tol.kkt"]:
                termination = "tol.kkt"
                niter = j
                break
            if der < tol_direction:
                termination = "tol.direction"
                niter = j
                if der < - tol_direction:
                    termination = "non-descending direction"
                    niter = j
                    if verbose:
                        print(
                        f"non-descendent direction, der={der:.2e}, tol={tol_direction:.2e}")
                break
            if verbose:
                print(f"Iteration {j}, Current cost {cost:2e}, kkt error {kkt_error:2e}, direction {der:2e}")
            tau, new_cost, states, matrices = self.globalization(alphaj, betaj, alpha_next, beta_next, der, cost, formulation,
                                            options["globalization.maxiter"], options["globalization.gamma"], options["globalization.beta"],
                                            verbose=verbose)
            if states is None:
                termination = "globalization.maxiter"
                niter = j+1
                break
            if cost - new_cost < max(abs(cost),abs(new_cost)) * options["rtol.cost_decrease"]:
                termination = "rtol.cost_decrease"
                niter = j+1
                break
            if verbose:
                print("tau", tau)
            alphaj = (1 - tau) * alphaj + tau * alpha_next
            betaj = (1 - tau) * betaj + tau * beta_next
            cost = new_cost
        if j == options["maxiter"]-1:
            niter = options["maxiter"]
            termination = "maxiter"

        if path:
            alphas.append(alphaj)
            betas.append(betaj)
            return alphas, betas

        if rescale:
            betaj = betaj *self.scale(alphaj, betaj)
        self.rtimes["total"] += time() - t0
        self.ncall["total"] += 1
        stats =  {"termination":termination, "niter":niter, "rtimes":self.rtimes, "ncall":self.ncall}
        stats["return_status"] = stats["termination"]
        return alphaj, betaj, stats

    def globalization(self, alpha0, beta0, alpha1, beta1, der, cost, formulation, maxiter, gamma, b, verbose=False):
        tau = 1.
        for i_glob in range(maxiter):
            alpha_middle = (1 - tau) * alpha0 + tau * alpha1
            beta_middle = (1 - tau) * beta0 + tau * beta1
            states, matrices, cost_middle = self.cost(alpha_middle, beta_middle, formulation)
            condition = (cost - cost_middle) / tau > gamma * der and np.all(beta_middle >= 0.)
            if condition:
                    return tau, cost_middle, states, matrices
            tau = tau * b
        if verbose: print("Globalization did not finish with tau = {}".format(tau))
        return tau, cost, None, None  

    def scale(self, alpha, beta, states_and_matrices=None):
        if states_and_matrices is None:
            states, matrices = self.kalman_simulate(alpha, beta)
        else:
            states, matrices = states_and_matrices
        dimension = (self.N+1) * self.model.ny
        lamb = np.sum(states["e"][:, np.newaxis] @ matrices["M"] @ states["e"][..., np.newaxis]) / dimension
        return lamb

    def cost_derivatives_fd(self, states, matrices, alpha, beta, formulation):
        der = 0.
        return der


# utils
def cost_eval(states, matrices, alpha, beta, formulation, L2pen=0.):
    if formulation=="MLE":
        value1 = np.sum(states["e"][:, np.newaxis] @ matrices["M"] @ states["e"][..., np.newaxis])
        value = value1 + np.sum( matrices["logdetS"] )
    elif formulation=="PredErr":
        value = np.sum( states["e"]**2 )
    value = value + L2pen * (np.sum( alpha**2 ) + np.sum(beta**2 ))
    return value