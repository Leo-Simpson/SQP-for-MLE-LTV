import casadi as ca #type: ignore
import numpy as np # type: ignore
from time import time
from .misc import symmetrize, symmetrize, psd_inverse

default_params = {"maxiter":50,
          "pen_step":0.,
          "tol.kkt":0.1,
          "tol.direction":1e-3,
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
        self.rtimes = {
            "linearized_fn":0.,
            "kalman_simulate_matrices":0.,
            "kalman_simulate_states":0.,
            "prepare":0.,
            "kalman_derivatives":0.,
            "cost_eval":0.,
            "cost_derivatives":0.,
            "QP":0.,
            "get_AbC":0.,
            "get_dAb":0.,
            "get_dQR":0.,
            "prepare_ein_opt":0.,
            "total":0.
            }

    def linearized_fn(self):
        t0 = time()
        self.A_fns, self.b_fns, self.C, self.dA_fns, self.db_fns = self.problem.gradient_dyna_fn()
        self.dQ_fn, self.dR_fn = self.model.gradient_covariances_fn()
        self.rtimes["linearized_fn"] += time() - t0
            
    def make_constr(self, eqconstr=True):
        self.derconstraints = self.model.Ineq.jacobian()
        # equality constraint, only used in approx
        alpha = ca.SX.sym("alpha_sym", self.model.nalpha)
        beta = ca.SX.sym("beta_sym", self.model.nbeta)
        Q, R = self.model.get_QR(beta)
        eq = ca.trace(Q) + ca.trace(R) - 1.
        if eqconstr:
            self.eqconstr = ca.Function("eqcstr", [alpha, beta], [eq])
            self.dereqconstraints = self.eqconstr.jacobian()
        else:
            self.eqconstr = None

    def make_lin_constr(self, alpha, beta):
        ab = np.concatenate([alpha, beta])
        Gconstr = - self.derconstraints(alpha, beta, 0).full()
        hconstr = self.model.Ineq(alpha, beta).full().squeeze() + Gconstr @ ab
        return Gconstr, hconstr

    def make_lin_eqconstr(self, alpha, beta):
        if self.eqconstr is None:
            return None
        ab = np.concatenate([alpha, beta])
        Aconstr = -self.dereqconstraints(alpha, beta, 0).full()
        bconstr = self.eqconstr(alpha, beta).full().squeeze() + Aconstr @ ab
        return Aconstr, bconstr

    def prepare(self):
        t0 = time()
        self.linearized_fn()
        self.inds_tri_M = np.tri(self.model.ny, k=-1, dtype=np.bool)
        self.rtimes["prepare"] += time() - t0

    def complete_param(self, param):
        params = default_params.copy()
        for key, item in param.items():
            params[key] = item
        return params

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
        return dAs, dbs
    
    def get_dQR(self, beta):
        t0 = time()
        dQ = self.dQ_fn(beta).full().reshape(self.model.nx, self.model.nx, self.model.nbeta)
        dR = self.dR_fn(beta).full().reshape(self.model.ny, self.model.ny, self.model.nbeta)
        dQ = symmetrize(dQ)
        dR = symmetrize(dR)
        self.rtimes["get_dQR"] += time() - t0
        return dQ, dR
   
    def kalman_simulate_matrices(self, alpha, beta):
        t0 = time()
        # symbol = type(alpha) is ca.SX
        S = np.empty( (self.N+1, self.model.ny, self.model.ny) )
        M = np.empty( (self.N+1, self.model.ny, self.model.ny) )
        P = np.empty( (self.N+1, self.model.nx, self.model.nx) )
        L = np.empty( (self.N+1, self.model.nx, self.model.ny) )
        logdetS = np.empty( (self.N+1) )
        P[0] = self.problem.P0
        As, bs = self.get_AbC(alpha)
        Q, R = self.model.get_QR(beta)
        for k in range(self.N+1):
            P[k] = symmetrize(P[k])
            PC = P[k] @ self.C.T
            S[k] = self.C @ PC + R
            M[k], logdetS[k] = psd_inverse(S[k], self.inds_tri_M)
            if k == self.N:
                break
            L[k] = As[k] @ PC
            P[k+1] = np.einsum("fx,xw,gw->fg",  As[k], P[k], As[k], optimize=False) \
                - np.einsum("fy,yz,gz->fg",  L[k], M[k], L[k], optimize=False) + Q
        matrices = {
               "S": S, "M": M, "L":L,
               "P":P, "logdetS": logdetS,
               "A" : As, "b": bs
        }
        self.rtimes["kalman_simulate_matrices"] += time() - t0
        return matrices
   
    def kalman_simulate_matrices_approx(self, alpha, beta):
        t0 = time()
        # symbol = type(alpha) is ca.SX
        K = np.empty( (self.N+1, self.model.nx, self.model.ny) )
        M = np.empty( (self.N+1, self.model.ny, self.model.ny) )
        P = np.empty( (self.N+1, self.model.nx, self.model.nx) )
        Pest = np.empty( (self.N, self.model.nx, self.model.nx) )
        P[0] = self.problem.P0
        As, bs = self.get_AbC(alpha)
        Q, R = self.model.get_QR(beta)
        for k in range(self.N+1):
            P[k] = symmetrize(P[k])
            PC = P[k] @ self.C.T
            S= self.C @ PC + R
            M[k] = psd_inverse(S, self.inds_tri_M, det=False)
            K[k] = PC @ M[k]
            if k == self.N:
                break
            Pest[k] = P[k] - K[k] @ PC.T
            P[k+1] = np.einsum("fx,xw,gw->fg",  As[k], Pest[k], As[k], optimize=False) + Q
        matrices = {
               "P":P, "Pest":Pest, "K":K, "M":M,
               "A" : As, "b": bs
        }
        self.rtimes["kalman_simulate_matrices"] += time() - t0
        return matrices

    def kalman_simulate_states(self, matrices):
        t0 = time()
        # symbol = type(alpha) is ca.SX
        z = np.empty( (self.N+1, self.model.ny) )
        x = np.empty( (self.N+1, self.model.nx) )
        x[0] = self.problem.x0
        for k in range(self.N+1):
            z[k] = matrices["M"][k] @ (self.C @ x[k] - self.problem.ys[k] )
            if k == self.N:
                break
            x[k+1] = matrices["A"][k] @ x[k] - matrices["L"][k] @ z[k] + matrices["b"][k]
        states = {"z" : z, "x":x}
        self.rtimes["kalman_simulate_states"] += time() - t0
        return states

    def kalman_simulate_states_approx(self, matrices):
        t0 = time()
        # symbol = type(alpha) is ca.SX
        e = np.empty( (self.N+1, self.model.ny) )
        x = np.empty( (self.N+1, self.model.nx) )
        xest = np.empty( (self.N, self.model.nx) )
        x[0] = self.problem.x0
        for k in range(self.N+1):
            e[k] = self.C @ x[k] - self.problem.ys[k]
            if k == self.N:
                break
            xest[k] = x[k] - matrices["K"][k] @ e[k]
            x[k+1] = matrices["A"][k] @ xest[k] + matrices["b"][k]
        states = {"e" : e, "x":x, "xest":xest}
        self.rtimes["kalman_simulate_states"] += time() - t0
        return states

    def kalman_simulate(self, alpha, beta, approx=False):
        if approx:
            matrices = self.kalman_simulate_matrices_approx(alpha, beta)
            states = self.kalman_simulate_states_approx(matrices)
        else:
            matrices = self.kalman_simulate_matrices(alpha, beta)
            states = self.kalman_simulate_states(matrices)
        return states, matrices
  
    def kalman_derivatives_(self, alpha, beta, states, matrices):
        t0 = time()
        # ab is the concatenation (alpha, beta)
        dz_dalpha = np.empty((self.N+1, self.model.ny, self.model.nalpha))
        dz_dbeta = np.empty((self.N+1, self.model.ny, self.model.nbeta))
        dM_dalpha = np.empty((self.N+1, self.model.ny, self.model.ny, self.model.nalpha))
        dM_dbeta = np.empty((self.N+1, self.model.ny, self.model.ny, self.model.nbeta))

        dx_dalpha, dP_dalpha = np.zeros((self.model.nx, self.model.nalpha)), np.zeros((self.model.nx, self.model.nx, self.model.nalpha))
        dx_dbeta, dP_dbeta = np.zeros((self.model.nx, self.model.nbeta)), np.zeros((self.model.nx, self.model.nx, self.model.nbeta))
        dAs, dbs = self.get_dAb(alpha)
        dQ, dR = self.get_dQR(beta)
        for k in range(self.N+1):
            xk, zk, Mk, C = states["x"][k], states["z"][k], matrices["M"][k], self.C
            dS_dbeta = np.einsum("xwb,yx,zw->yzb", dP_dbeta, C, C, optimize=False) + dR
            dM_dalpha[k] = symmetrize( -np.einsum("xwa,yx,zw,yv,uz->vua", dP_dalpha, C, C, Mk, Mk, optimize=False) )
            dM_dbeta[k] = symmetrize( -np.einsum("yzb,yv,uz->vub", dS_dbeta, Mk, Mk, optimize=False) )

            dz_dalpha[k] = np.einsum("wya,y->wa", dM_dalpha[k], C @xk - self.problem.ys[k], optimize=False) \
                        + Mk @ (C @ dx_dalpha)
            dz_dbeta[k] = np.einsum("wyb,y->wb", dM_dbeta[k], C @xk - self.problem.ys[k], optimize=False) \
                        + Mk @ (C @ dx_dbeta)

            if k == self.N:
                break
            Pk, Lk, A = matrices["P"][k], matrices["L"][k], matrices["A"][k]
            dL_dalpha = np.einsum("xwa,wv,yv->xya", dAs[k], Pk, C, optimize=False) \
                + np.einsum("xw,wva,yv->xya", A, dP_dalpha, C, optimize=False)
            dL_dbeta = np.einsum("xw,wvb,yv->xyb", A, dP_dbeta, C, optimize=False)
            dx_dalpha = A @ dx_dalpha - Lk @ dz_dalpha[k] + dbs[k] \
                + np.einsum("xwa,w->xa", dAs[k], xk, optimize=False) - np.einsum("xwa,w->xa", dL_dalpha, zk, optimize=False)
            dx_dbeta = A @ dx_dbeta - Lk @ dz_dbeta[k] \
                - np.einsum("xwa,w->xa", dL_dbeta, zk, optimize=False)

            dP_dalpha = 2 * np.einsum("xwa,wv,uv->xua", dAs[k], Pk, A, optimize=False) \
                + np.einsum("xw,wva,uv->xua", A, dP_dalpha, A, optimize=False) \
                - 2 * np.einsum("xwa,wv,uv->xua", dL_dalpha, Mk, Lk, optimize=False) \
                - np.einsum("xw,wva,uv->xua", Lk, dM_dalpha[k], Lk, optimize=False)
            

            dP_dbeta = np.einsum("xw,wvb,uv->xub", A, dP_dbeta, A, optimize=False) \
                - 2 * np.einsum("xwb,wv,uv->xub", dL_dbeta, Mk, Lk, optimize=False) \
                - np.einsum("xw,wvb,uv->xub", Lk, dM_dbeta[k], Lk, optimize=False) \
                + dQ
            dP_dalpha = symmetrize(dP_dalpha)
            dP_dbeta = symmetrize(dP_dbeta)

        dz_dab = np.concatenate([dz_dalpha, dz_dbeta], axis=-1)
        dM_dab = np.concatenate([dM_dalpha, dM_dbeta], axis=-1)
        derivatives = {"dz" : dz_dab, "dM" : dM_dab}
        self.delete_unecessary(states, matrices, approx=False)
        self.rtimes["kalman_derivatives"] += time() - t0
        return derivatives

    def kalman_derivatives_approx(self, alpha, beta, states, matrices):
        t0 = time()
        # ab is the concatenation (alpha, beta)
        de_dalpha = np.empty((self.N+1, self.model.ny, self.model.nalpha))
        de_dbeta = np.empty((self.N+1, self.model.ny, self.model.nbeta))

        dx_dalpha, dP_dalpha = np.zeros((self.model.nx, self.model.nalpha)), np.zeros((self.model.nx, self.model.nx, self.model.nalpha))
        dx_dbeta, dP_dbeta = np.zeros((self.model.nx, self.model.nbeta)), np.zeros((self.model.nx, self.model.nx, self.model.nbeta))
        dAs, dbs = self.get_dAb(alpha)
        dQ, dR = self.get_dQR(beta)
        for k in range(self.N+1):
            ek, Mk, C  = states["e"][k], matrices["M"][k], self.C
            dS_dbeta = np.einsum("xwb,yx,zw->yzb", dP_dbeta, C, C, optimize=False) + dR
            dM_dalpha = symmetrize( -np.einsum("xwa,yx,zw,yv,uz->vua", dP_dalpha, C, C, Mk, Mk, optimize=False) )
            dM_dbeta = symmetrize( -np.einsum("yzb,yv,uz->vub", dS_dbeta, Mk, Mk, optimize=False) )
            de_dalpha[k] = C @ dx_dalpha
            de_dbeta[k] = C @ dx_dbeta
            if k == self.N:
                break
            xestk, Kk, Pk, Pestk, A = states["xest"][k], matrices["K"][k], matrices["P"][k], matrices["Pest"][k], matrices["A"][k]

            dK_dalpha = np.einsum("xv,yv,yza->xza", Pk, C, dM_dalpha, optimize=False) \
                + np.einsum("xva,yv,yz->xza", dP_dalpha, C, Mk, optimize=False) 
            dK_dbeta = np.einsum("xv,yv,yzb->xzb", Pk, C, dM_dbeta, optimize=False) \
                + np.einsum("xvb,yv,yz->xzb", dP_dbeta, C, Mk, optimize=False)

            dxest_dalpha = dx_dalpha - Kk @ de_dalpha[k] - np.einsum("xwa,w->xa", dK_dalpha, ek, optimize=False)
            dxest_dbeta = dx_dbeta - Kk @ de_dbeta[k] - np.einsum("xwb,w->xb", dK_dbeta, ek, optimize=False)

            dx_dalpha = A @ dxest_dalpha + dbs[k] + np.einsum("xwa,w->xa", dAs[k], xestk, optimize=False) 
            dx_dbeta = A @ dxest_dbeta

            dPest_dalpha = dP_dalpha - np.einsum("xy,yu,uva->xva", Kk, C, dP_dalpha) - np.einsum("xya,yu,uv->xva", dK_dalpha, C, Pk)
            dPest_dbeta = dP_dbeta - np.einsum("xy,yu,uvb->xvb", Kk, C, dP_dbeta) - np.einsum("xyb,yu,uv->xvb", dK_dbeta, C, Pk)

            dP_dalpha = np.einsum("xw,wva,uv->xua", A, dPest_dalpha, A, optimize=False) \
                        + 2 * np.einsum("xwa,wv,uv->xua", dAs[k], Pestk, A, optimize=False)
            dP_dbeta = np.einsum("xw,wvb,uv->xub", A, dPest_dbeta, A, optimize=False) + dQ

            dP_dalpha = symmetrize(dP_dalpha)
            dP_dbeta = symmetrize(dP_dbeta)

        de_dab = np.concatenate([de_dalpha, de_dbeta], axis=-1)
        derivatives = {"de" : de_dab}
        self.delete_unecessary(states, matrices, approx=True)
        self.rtimes["kalman_derivatives"] += time() - t0
        return derivatives

    def kalman_derivatives(self, alpha, beta, states, matrices, approx=False):
        if approx:
            return self.kalman_derivatives_approx(alpha, beta, states, matrices)
        else:
            return self.kalman_derivatives_(alpha, beta, states, matrices)

    def delete_unecessary(self, states, matrices, approx=False):
        del states["x"]
        del matrices["M"]
        del matrices["P"]
        del matrices["A"]
        del matrices["b"]
        if approx:
            del matrices["Pest"]
            del matrices["K"]
        else:
            del matrices["L"]

    def cost_eval(self, states, matrices, alpha, beta, approx=False):
        t0 = time()
        if approx:
            value = np.sum( states["e"]**2 )
        else:
            value = np.einsum("kyz,ky,kz->", matrices["S"], states["z"], states["z"], optimize=False) \
                + np.sum( matrices["logdetS"] )
        value = value + self.problem.L2pen * (np.sum( alpha**2 ) + np.sum(beta**2 ))
        self.rtimes["cost_eval"] += time() - t0
        return value

    def cost_derivatives(self, states, matrices, derivatives, approx=False, new_hessian=True, neglect_logdet_hessian=True):
        t0 = time()
        if approx:
            gradient =  2*np.einsum("kzp,kz->p",derivatives["de"], states["e"], optimize=False)
            hessian = 2*np.einsum("kzp,kzq->pq", derivatives["de"], derivatives["de"], optimize=False)
        else:
            
            e =  np.einsum("kyz,kz->ky", matrices["S"], states["z"], optimize=False)
            SdM = np.einsum("kyw,kwzp->kyzp", matrices["S"], derivatives["dM"], optimize=False)
            dMe = np.einsum("kywp,kw->kyp", derivatives["dM"], e, optimize=False)
            gradient = np.einsum("kzp,kz->p", 2*derivatives["dz"] - dMe, e, optimize=False) - np.trace(np.sum(SdM, axis=0))
            h1 =  np.einsum( "kzp,kzy,kyq->pq", derivatives["dz"], matrices["S"], derivatives["dz"], optimize=False  )
            if new_hessian:
                h_quad = 2 * h1
            else:
                h2 = np.einsum("kyp,kyw,kwq->pq", derivatives["dz"] - dMe, matrices["S"], derivatives["dz"] - dMe, optimize=False)
                h3 = np.einsum( "kzp,kzy,kyq->pq", dMe, matrices["S"], dMe , optimize=False )
                h_quad = h1 + h2 + h3
            h_logdet = np.einsum("kyzp,kyzq->pq", SdM, SdM, optimize=False)
            if neglect_logdet_hessian:
                hessian = h_quad
            else:
                hessian = h_quad + h_logdet

        self.rtimes["cost_derivatives"] += time() - t0
        return gradient, hessian

    def eval_kkt_error(self, alpha, beta, gradient, multiplier):
        kkt_error =  gradient + self.derconstraints(alpha, beta, 0).full().T @ multiplier
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
        return x, lam, der

    def SQP_kalman(self, alpha0, beta0, param={}, approx=False, verbose=True, path=False, rescale=True):
        t0 = time()
        params = self.complete_param(param)
        nalpha = self.model.nalpha
        alphaj = alpha0.copy()
        betaj = beta0.copy()
        states, matrices = self.kalman_simulate(alphaj, betaj, approx=approx)
        cost = self.cost_eval(states, matrices, alphaj, betaj, approx=approx)
        objective_scale = 1.
        tol_direction = objective_scale * params["tol.direction"]
        if path:
            alphas, betas = [], []
        for j in range(params["maxiter"]):
            if path:
                alphas.append(alphaj)
                betas.append(betaj)
            if (not approx) and rescale:
                betaj = betaj * self.scale(alphaj, betaj, states_and_matrices=(states, matrices), approx=False)
            derivatives = self.kalman_derivatives(alphaj, betaj, states, matrices, approx=approx)
            gradient, hessian = self.cost_derivatives(states, matrices, derivatives, approx=approx)
            ab = np.concatenate([alphaj, betaj])
            hessian_ = hessian + 2*params["pen_step"]* np.eye(len(ab))
            grad0 = gradient - hessian_ @ ab
            hessian_ = hessian_ + 2*self.problem.L2pen * np.eye(len(ab))
            linconstr = self.make_lin_constr(alphaj, betaj)
            if approx:
                lin_eqconstr = self.make_lin_eqconstr(alphaj, betaj)
            else:
                lin_eqconstr = None
            ab_next, multiplier, der = self.solveQP(hessian_, grad0, linconstr, lin_eqconstr, ab)
            alpha_next, beta_next = ab_next[:nalpha], ab_next[nalpha:]
            if np.any(beta_next<0.):
                print(f"beta {betaj}, betanext {beta_next}")
            kkt_error = self.eval_kkt_error(alphaj, betaj, gradient, multiplier)
            
            if kkt_error < params["tol.kkt"]:
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
            tau, new_cost, states, matrices = self.globalization(alphaj, betaj, alpha_next, beta_next, der, cost,
                                            params["globalization.maxiter"], params["globalization.gamma"], params["globalization.beta"],
                                            approx=approx, verbose=verbose)
            if states is None:
                termination = "globalization.maxiter"
                niter = j+1
                break
            if cost - new_cost < max(abs(cost),abs(new_cost)) * params["rtol.cost_decrease"]:
                termination = "rtol.cost_decrease"
                niter = j+1
                break
            if verbose:
                print("tau", tau)
            alphaj = (1 - tau) * alphaj + tau * alpha_next
            betaj = (1 - tau) * betaj + tau * beta_next
            cost = new_cost
        if j == params["maxiter"]-1:
            niter = params["maxiter"]
            termination = "maxiter"

        if path:
            alphas.append(alphaj)
            betas.append(betaj)
            return alphas, betas

        if rescale:
            betaj = betaj *self.scale(alphaj, betaj, approx=approx)
        self.rtimes["total"] += time() - t0
        stats =  {"termination":termination, "niter":niter, "rtimes":self.rtimes}
        stats["return_status"] = stats["termination"]
        return alphaj, betaj, stats

    def globalization(self, alpha0, beta0, alpha1, beta1, der, cost, maxiter, gamma, b, approx=False, verbose=False):
        tau = 1.
        for i_glob in range(maxiter):
            alpha_middle = (1 - tau) * alpha0 + tau * alpha1
            beta_middle = (1 - tau) * beta0 + tau * beta1
            states, matrices = self.kalman_simulate(alpha_middle, beta_middle, approx=approx)
            cost_middle = self.cost_eval(states, matrices, alpha_middle, beta_middle, approx=approx)
            condition = (cost - cost_middle) / tau > gamma * der and np.all(beta_middle >= 0.)
            if condition:
                    return tau, cost_middle, states, matrices
            tau = tau * b
        if verbose: print("Globalization did not finish with tau = {}".format(tau))
        return tau, cost, None, None  

    def scale(self, alpha, beta, states_and_matrices=None, approx=False):
        if states_and_matrices is None:
            states, matrices = self.kalman_simulate(alpha, beta, approx=approx)
        else:
            states, matrices = states_and_matrices
        dimension = (self.N+1) * self.model.ny
        if approx:
            lamb =  np.einsum("kyz,ky,kz->", matrices["M"], states["e"], states["e"]) / dimension
        else:
            lamb = np.einsum("kyz,ky,kz->", matrices["S"], states["z"], states["z"])  / dimension
        return lamb
