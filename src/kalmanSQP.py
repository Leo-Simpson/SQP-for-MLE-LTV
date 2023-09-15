import casadi as ca #type: ignore
import numpy as np # type: ignore
from time import time
from .misc import symmetrize, symmetrize, psd_inverse

default_opts = {"maxiter":50,
          "pen_step":0.,
          "tol.kkt":0.1,
          "tol.direction":1e-3,
          "rtol.cost_decrease":1e-6,
          "globalization.maxiter":20,
          "globalization.beta":0.8,
          "globalization.gamma":0.1,
          "einsum": True,
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
            "cost_eval":0.,
            "cost_derivatives":0.,
            "QP":0.,
            "get_AbC":0.,
            "get_dAb":0.,
            "get_dQR":0.,
            "total":0.
            }

    def linearized_fn(self):
        t0 = time()
        self.A_fns, self.b_fns, self.C, self.dA_fns, self.db_fns = self.problem.gradient_dyna_fn()
        self.dQ_fn, self.dR_fn = self.model.gradient_covariances_fn()
        self.rtimes["linearized_fn"] += time() - t0
            
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
   
    def kalman_simulate_matrices_MLE(self, alpha, beta):
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
            if self.einsum:
                P[k+1] = np.einsum("fx,xw,gw->fg",  As[k], P[k], As[k], optimize=False) \
                    - np.einsum("fy,yz,gz->fg",  L[k], M[k], L[k], optimize=False) + Q
            else:
                P[k+1] = As[k] @ P[k] @ As[k].T - L[k] @ M[k] @ L[k].T + Q
                
        matrices = {
               "S": S, "M": M, "L":L,
               "P":P, "logdetS": logdetS,
               "A" : As, "b": bs
        }
        self.rtimes["kalman_simulate_matrices"] += time() - t0
        return matrices
   
    def kalman_simulate_matrices_PredErr(self, alpha, beta):
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
            if self.einsum:
                P[k+1] = np.einsum("fx,xw,gw->fg",  As[k], Pest[k], As[k], optimize=False) + Q
            else:
                P[k+1] = As[k] @ Pest[k] @ As[k].T + Q
                
        matrices = {
               "P":P, "Pest":Pest, "K":K, "M":M,
               "A" : As, "b": bs
        }
        self.rtimes["kalman_simulate_matrices"] += time() - t0
        return matrices

    def kalman_simulate_states_MLE(self, matrices):
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

    def kalman_simulate_states_PredErr(self, matrices):
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

    def kalman_simulate(self, alpha, beta, formulation):
        if formulation=="MLE":
            matrices = self.kalman_simulate_matrices_MLE(alpha, beta)
            states = self.kalman_simulate_states_MLE(matrices)
        elif formulation=="PredErr":
            matrices = self.kalman_simulate_matrices_PredErr(alpha, beta)
            states = self.kalman_simulate_states_PredErr(matrices)
        return states, matrices
  
    def cost_derivatives_MLE(self,  states, matrices, alpha, beta, neglect_logdet_hessian=True):
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
            if self.einsum:
                dS_dbeta = np.einsum("xwb,yx,zw->yzb", dP_dbeta, C, C, optimize=False) + dR
                dM_dalpha[k] = -np.einsum("xwa,yx,zw,yv,uz->vua", dP_dalpha, C, C, Mk, Mk, optimize=False)
                dM_dbeta[k] = -np.einsum("yzb,yv,uz->vub", dS_dbeta, Mk, Mk, optimize=False)
                dz_dalpha[k] = np.einsum("wya,y->wa", dM_dalpha[k], C @xk - self.problem.ys[k], optimize=False) + Mk @ (C @ dx_dalpha)
                dz_dbeta[k] = np.einsum("wya,y->wa", dM_dbeta[k], C @xk - self.problem.ys[k], optimize=False) + Mk @ (C @ dx_dbeta)
            else:
                dS_dalpha = np.tensordot(C, C @ dP_dalpha, axes=(1,0))
                dS_dbeta = np.tensordot(C, C @ dP_dbeta, axes=(1,0)) + dR
                dM_dalpha[k] = -np.tensordot(Mk, Mk @ dS_dalpha, axes=(1, 0))
                dM_dbeta[k] = -np.tensordot(Mk, Mk @ dS_dbeta, axes=(1, 0))
                dz_dalpha[k] = np.swapaxes(dM_dalpha[k], 1, 2) @ (C @ xk - self.problem.ys[k]) + Mk @ (C @ dx_dalpha)
                dz_dbeta[k] = np.swapaxes(dM_dbeta[k], 1, 2) @ (C @ xk - self.problem.ys[k]) + Mk @ (C @ dx_dbeta)

            dM_dalpha[k] = symmetrize(dM_dalpha[k])
            dM_dbeta[k] = symmetrize(dM_dbeta[k])
            if k == self.N:
                break
            Pk, Lk, A = matrices["P"][k], matrices["L"][k], matrices["A"][k]

            if self.einsum:
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
            else:
                dL_dalpha = C @ (Pk.T @ dAs[k] + np.tensordot(A, dP_dalpha, axes=(1, 0)))
                dL_dbeta = C @ np.tensordot(A, dP_dbeta, axes=(1, 0))
                dx_dalpha = A @ dx_dalpha - Lk @ dz_dalpha[k] + dbs[k] \
                    + np.swapaxes(dAs[k], 1, 2) @ xk - np.swapaxes(dL_dalpha, 1, 2) @ zk
                dx_dbeta = A @ dx_dbeta - Lk @ dz_dbeta[k] \
                    - np.swapaxes(dL_dbeta, 1, 2) @ zk
                dP_dalpha = np.tensordot(A, A @ dP_dalpha, axes=(1, 0)) \
                    + 2 * np.tensordot(Pk @ A.T, dAs[k], axes=(0, 1)) \
                    - 2 * np.tensordot(Mk @ Lk.T, dL_dalpha, axes=(0, 1)) \
                    - np.tensordot(Lk, Lk @ dM_dalpha[k], axes=(1, 0))
                dP_dbeta = np.tensordot(A, A @ dP_dbeta, axes=(1, 0)) \
                    - 2 * np.tensordot(Mk @ Lk.T, dL_dbeta, axes=(0, 1)) \
                    - np.tensordot(Lk, Lk @ dM_dbeta[k], axes=(1, 0)) \
                    + dQ
            dP_dalpha = symmetrize(dP_dalpha)
            dP_dbeta = symmetrize(dP_dbeta)

        dz_dab = np.concatenate([dz_dalpha, dz_dbeta], axis=-1)
        dM_dab = np.concatenate([dM_dalpha, dM_dbeta], axis=-1)
        self.delete_unecessary(states, matrices, "MLE")
        if self.einsum:
            e =  np.einsum("kyz,kz->ky", matrices["S"], states["z"], optimize=False)
            SdM = np.einsum("kyw,kwzp->kyzp", matrices["S"], dM_dab, optimize=False)
            dMe = np.einsum("kywp,kw->kyp", dM_dab, e, optimize=False)
            gradient = np.einsum("kzp,kz->p", 2*dz_dab - dMe, e, optimize=False) - np.trace(np.sum(SdM, axis=0))
            hessian =  np.einsum( "kzp,kzy,kyq->pq", dz_dab, matrices["S"], dz_dab, optimize=False  )
            if not neglect_logdet_hessian:
                hessian = hessian + np.einsum("kyzp,kyzq->pq", SdM, SdM, optimize=False)
        else:
            e =  (matrices["S"] @ states["z"][..., np.newaxis])[..., 0]
            SdM = np.transpose(matrices["S"][:, np.newaxis] @ np.transpose(dM_dab, (0, 3, 1, 2)), (0, 2, 3, 1))
            dMe = (e[:, np.newaxis, np.newaxis] @ dM_dab)[:, :, 0]
            gradient = np.sum((e[:, np.newaxis] @ (2*dz_dab - dMe))[:,0], axis=0) - np.trace(np.sum(SdM, axis=0))
            hessian = np.sum(np.swapaxes(dz_dab, 1, 2) @ matrices["S"] @ dz_dab, axis=0)
            if not neglect_logdet_hessian:
                hessian = hessian + np.sum(np.swapaxes(SdM, -1, -2) @ SdM, axis=(0, 1))
        return gradient, hessian

    def cost_derivatives_PredErr(self, states, matrices, alpha, beta):
        de_dalpha = np.empty((self.N+1, self.model.ny, self.model.nalpha))
        de_dbeta = np.empty((self.N+1, self.model.ny, self.model.nbeta))

        dx_dalpha, dP_dalpha = np.zeros((self.model.nx, self.model.nalpha)), np.zeros((self.model.nx, self.model.nx, self.model.nalpha))
        dx_dbeta, dP_dbeta = np.zeros((self.model.nx, self.model.nbeta)), np.zeros((self.model.nx, self.model.nx, self.model.nbeta))
        dAs, dbs = self.get_dAb(alpha)
        dQ, dR = self.get_dQR(beta)
        for k in range(self.N+1):
            ek, Mk, C  = states["e"][k], matrices["M"][k], self.C
            if self.einsum:
                dS_dbeta = np.einsum("xwb,yx,zw->yzb", dP_dbeta, C, C, optimize=False) + dR
                dM_dalpha = -np.einsum("xwa,yx,zw,yv,uz->vua", dP_dalpha, C, C, Mk, Mk, optimize=False)
                dM_dbeta = -np.einsum("yzb,yv,uz->vub", dS_dbeta, Mk, Mk, optimize=False)
            else:
                dS_dalpha = np.tensordot(C, C @ dP_dalpha, axes=(1,0))
                dS_dbeta = np.tensordot(C, C @ dP_dbeta, axes=(1,0)) + dR
                dM_dalpha =-np.tensordot(Mk, Mk @ dS_dalpha, axes=(1, 0))
                dM_dbeta = -np.tensordot(Mk, Mk @ dS_dbeta, axes=(1, 0))
            dM_dalpha = symmetrize(dM_dalpha)
            dM_dalpha = symmetrize(dM_dbeta)
            de_dalpha[k] = C @ dx_dalpha
            de_dbeta[k] = C @ dx_dbeta
            if k == self.N:
                break
            xestk, Kk, Pk, Pestk, A = states["xest"][k], matrices["K"][k], matrices["P"][k], matrices["Pest"][k], matrices["A"][k]
            PC = Pk @ C.T
            if self.einsum:
                dK_dalpha = np.einsum("xv,yv,yza->xza", Pk, C, dM_dalpha, optimize=False) \
                    + np.einsum("xva,yv,yz->xza", dP_dalpha, C, Mk, optimize=False)
                dK_dbeta = np.einsum("xv,yv,yzb->xzb", Pk, C, dM_dbeta, optimize=False) \
                    + np.einsum("xvb,yv,yz->xzb", dP_dbeta, C, Mk, optimize=False)
                dxest_dalpha = dx_dalpha - Kk @ de_dalpha[k] - np.einsum("xwa,w->xa", dK_dalpha, ek, optimize=False)
                dxest_dbeta = dx_dbeta - Kk @ de_dbeta[k] - np.einsum("xwb,w->xb", dK_dbeta, ek, optimize=False)
                dx_dalpha = A @ dxest_dalpha + dbs[k] + np.einsum("xwa,w->xa", dAs[k], xestk, optimize=False)
                dPest_dalpha = dP_dalpha - np.einsum("xy,yu,uva->xva", Kk, C, dP_dalpha) - np.einsum("xya,yu,uv->xva", dK_dalpha, C, Pk)
                dPest_dbeta = dP_dbeta - np.einsum("xy,yu,uvb->xvb", Kk, C, dP_dbeta) - np.einsum("xyb,yu,uv->xvb", dK_dbeta, C, Pk)
                dP_dalpha = np.einsum("xw,wva,uv->xua", A, dPest_dalpha, A, optimize=False) \
                            + 2 * np.einsum("xwa,wv,uv->xua", dAs[k], Pestk, A, optimize=False)
                dP_dbeta = np.einsum("xw,wvb,uv->xub", A, dPest_dbeta, A, optimize=False) + dQ
            else:
                dK_dalpha = np.tensordot(PC, dM_dalpha, axes=(1,0)) + (Mk @ C) @ dP_dalpha
                dK_dbeta = np.tensordot(PC, dM_dbeta, axes=(1,0)) + (Mk @ C) @ dP_dbeta
                dxest_dalpha = dx_dalpha - Kk @ de_dalpha[k] - np.swapaxes(dK_dalpha, 1, 2) @ ek
                dxest_dbeta = dx_dbeta - Kk @ de_dbeta[k] -  np.swapaxes(dK_dbeta, 1, 2) @ ek
                dx_dalpha = A @ dxest_dalpha + dbs[k] + np.swapaxes(dAs[k], 1, 2) @ xestk
                dPest_dalpha = dP_dalpha - np.tensordot( Kk @ C, dP_dalpha, axes=(1,0)) -  PC @ dK_dalpha
                dPest_dbeta = dP_dbeta - np.tensordot( Kk @ C, dP_dbeta, axes=(1,0)) -  PC @ dK_dbeta
                dP_dalpha = np.tensordot(A, A @ dPest_dalpha, axes=(1, 0)) \
                    + 2 * np.tensordot(Pestk @ A.T, dAs[k], axes=(0, 1))
                dP_dbeta = np.tensordot(A, A @ dPest_dbeta, axes=(1, 0)) + dQ
            dx_dbeta = A @ dxest_dbeta

            dP_dalpha = symmetrize(dP_dalpha)
            dP_dbeta = symmetrize(dP_dbeta)

        de_dab = np.concatenate([de_dalpha, de_dbeta], axis=-1)
        if self.einsum:
            gradient =  2*np.einsum("kzp,kz->p",de_dab, states["e"], optimize=False)
            hessian = 2*np.einsum("kzp,kzq->pq", de_dab, de_dab, optimize=False)
        else:
            gradient =  2 * np.tensordot(de_dab, states["e"], axes=([0,1], [0,1])  )
            hessian = 2 * np.tensordot(de_dab, de_dab, axes=([0,1], [0,1])  )
        self.delete_unecessary(states, matrices, "PredErr")
        return gradient, hessian
            
    def delete_unecessary(self, states, matrices, formulation):
        del states["x"]
        del matrices["M"]
        del matrices["P"]
        del matrices["A"]
        del matrices["b"]
        if formulation=="MLE":
            del matrices["L"]
        elif formulation=="PredErr":
            del matrices["Pest"]
            del matrices["K"]

    def cost_eval(self, states, matrices, alpha, beta, formulation):
        t0 = time()
        if formulation=="MLE":
            if self.einsum:
                value1 = np.einsum("kyz,ky,kz->", matrices["S"], states["z"], states["z"], optimize=False)
            else:
                value1 = np.sum(states["z"][:, np.newaxis] @ matrices["S"] @ states["z"][..., np.newaxis])
            value = value1 + np.sum( matrices["logdetS"] )
        elif formulation=="PredErr":
            value = np.sum( states["e"]**2 )
        value = value + self.problem.L2pen * (np.sum( alpha**2 ) + np.sum(beta**2 ))
        self.rtimes["cost_eval"] += time() - t0
        return value

    def cost_derivatives(self, states, matrices, alpha, beta, formulation, neglect_logdet_hessian=True):
        t0 = time()
        if formulation=="MLE":
            gradient, hessian = self.cost_derivatives_MLE(states, matrices, alpha, beta, neglect_logdet_hessian=neglect_logdet_hessian)
        elif formulation=="PredErr":
            gradient, hessian = self.cost_derivatives_PredErr(states, matrices, alpha, beta)
        self.rtimes["cost_derivatives"] += time() - t0
        return gradient, hessian

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
        return x, lam, der

    def SQP_kalman(self, alpha0, beta0, formulation, opts={}, verbose=True, path=False, rescale=True):
        if formulation not in ["MLE", "PredErr"]:
            raise ValueError("Formulation {} is unknown. Choose between 'MLE' or 'PredError'".format(formulation))
        t0 = time()
        options = self.complete_opts(opts)
        self.einsum = options["einsum"]
        nalpha = self.model.nalpha
        alphaj = alpha0.copy()
        betaj = beta0.copy()
        states, matrices = self.kalman_simulate(alphaj, betaj, formulation)
        cost = self.cost_eval(states, matrices, alphaj, betaj, formulation)
        objective_scale = 1.
        tol_direction = objective_scale * options["tol.direction"]
        if path:
            alphas, betas = [], []
        for j in range(options["maxiter"]):
            if path:
                alphas.append(alphaj)
                betas.append(betaj)
            if (formulation=="MLE") and rescale:
                betaj = betaj * self.scale(alphaj, betaj, formulation, states_and_matrices=(states, matrices))
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
            betaj = betaj *self.scale(alphaj, betaj, formulation)
        self.rtimes["total"] += time() - t0
        stats =  {"termination":termination, "niter":niter, "rtimes":self.rtimes}
        stats["return_status"] = stats["termination"]
        return alphaj, betaj, stats

    def globalization(self, alpha0, beta0, alpha1, beta1, der, cost, formulation, maxiter, gamma, b, verbose=False):
        tau = 1.
        for i_glob in range(maxiter):
            alpha_middle = (1 - tau) * alpha0 + tau * alpha1
            beta_middle = (1 - tau) * beta0 + tau * beta1
            states, matrices = self.kalman_simulate(alpha_middle, beta_middle, formulation)
            cost_middle = self.cost_eval(states, matrices, alpha_middle, beta_middle, formulation)
            condition = (cost - cost_middle) / tau > gamma * der and np.all(beta_middle >= 0.)
            if condition:
                    return tau, cost_middle, states, matrices
            tau = tau * b
        if verbose: print("Globalization did not finish with tau = {}".format(tau))
        return tau, cost, None, None  

    def scale(self, alpha, beta, formulation, states_and_matrices=None):
        if states_and_matrices is None:
            states, matrices = self.kalman_simulate(alpha, beta, formulation)
        else:
            states, matrices = states_and_matrices
        dimension = (self.N+1) * self.model.ny
        if formulation=="MLE":
            if self.einsum:
                lamb = np.einsum("kyz,ky,kz->", matrices["S"], states["z"], states["z"])  / dimension
            else:
                lamb = np.sum(states["z"][:, np.newaxis] @ matrices["S"] @ states["z"][..., np.newaxis]) / dimension
        elif formulation=="PredErr":
            if self.einsum:
                lamb = np.einsum("kyz,ky,kz->", matrices["M"], states["e"], states["e"]) / dimension
            else:
                lamb = np.sum(states["e"][:, np.newaxis] @ matrices["M"] @ states["e"][..., np.newaxis]) / dimension
        return lamb
