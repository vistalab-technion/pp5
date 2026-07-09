#!/usr/bin/env python
"""
MMD breakdown arm: unbiased MMD^2 (U-statistic) and biased RBF-MMD (V-statistic),
Gaussian kernel on the flat torus (sigma=10 deg). For the 6 candidate pairs:
baseline MMD^2 + permutation p, and breakdown-k for both estimators.

Significance is assessed at the per-SS BH thresholds established by the published
KDE tests (HELIX 0.00115, TURN 0.00057) so the 5-statistic side-by-side is on a
common footing; a statistic-specific MMD-BH (all 87 pairs) is a later refinement.

Fast permutation via the row-sum identity (sum only the smaller group's block);
clean O(1)-update leave-one-out for the influence ranking.
"""
import numpy as np
import pandas as pd

DS = "out/pnas-2026-repro/pointwise_cdist-SMOKE-kde_g_10-cr_none/_intermediate_/dataset.csv"
SIG = np.deg2rad(10.0)
KP, SEED = 5000, 12345
THR = {"HELIX": 0.0011494, "TURN": 0.0005747}
PAIRS = [("HELIX","L-CTC","L-TTG"),("HELIX","L-CTC","L-CTG"),("HELIX","L-CTC","L-CTT"),
         ("HELIX","R-AGG","R-CGA"),("TURN","A-GCG","A-GCT"),("TURN","P-CCC","P-CCG")]


def kmat(Z):
    d = np.abs(Z[:, None, :] - Z[None, :, :]); d = np.minimum(d, 2*np.pi - d)
    return np.exp(-(d**2).sum(2) / (2*SIG**2))


def mmd2(Sxx, Syy, Sxy, nx, ny, est):
    if est == "b":
        return Sxx/nx**2 + Syy/ny**2 - 2*Sxy/(nx*ny)
    return (Sxx-nx)/(nx*(nx-1)) + (Syy-ny)/(ny*(ny-1)) - 2*Sxy/(nx*ny)


def perm_p(K, nx, ny, est, k=KP, seed=SEED):
    N = nx+ny; R = K.sum(1); T = R.sum()
    Sxx0 = K[:nx, :nx].sum(); Sxy0 = K[:nx, nx:].sum(); Syy0 = T - 2*(Sxx0+Sxy0) + Sxx0
    Syy0 = K[nx:, nx:].sum()
    obs = mmd2(Sxx0, Syy0, Sxy0, nx, ny, est)
    rng = np.random.default_rng(seed); c = 0
    for _ in range(k):
        S = rng.permutation(N)[:nx]
        sxx = K[np.ix_(S, S)].sum(); rs = R[S].sum()
        sxy = rs - sxx; syy = T - 2*rs + sxx
        if mmd2(sxx, syy, sxy, nx, ny, est) >= obs - 1e-15:
            c += 1
    return obs, (c+1)/(k+1)


def breakdown(X, Y, thresh, k_grid, est):
    xk, yk = list(range(len(X))), list(range(len(Y)))
    rows, bk = [], None
    for k in range(0, max(k_grid)+1):
        Z = np.vstack([X[xk], Y[yk]]); K = kmat(Z); nx, ny = len(xk), len(yk)
        if k in k_grid:
            _, p = perm_p(K, nx, ny, est)
            rows.append((k, p, p <= thresh))
            if p > thresh and bk is None:
                bk = k
        if k == max(k_grid) or bk is not None:
            break
        Sxx = K[:nx, :nx].sum(); Syy = K[nx:, nx:].sum(); Sxy = K[:nx, nx:].sum()
        base = mmd2(Sxx, Syy, Sxy, nx, ny, est)
        RxIn = K[:nx, :nx].sum(1); RxCr = K[:nx, nx:].sum(1)
        RyIn = K[nx:, nx:].sum(1); RyCr = K[nx:, :nx].sum(1)
        best, bd, side = None, -1e9, None
        for i in range(nx):
            v = mmd2(Sxx-2*RxIn[i]+K[i, i], Syy, Sxy-RxCr[i], nx-1, ny, est)
            if base - v > bd: bd, best, side = base - v, i, "X"
        for j in range(ny):
            jj = nx+j
            v = mmd2(Sxx, Syy-2*RyIn[j]+K[jj, jj], Sxy-RyCr[j], nx, ny-1, est)
            if base - v > bd: bd, best, side = base - v, j, "Y"
        (xk if side == "X" else yk).pop(best)
    return rows, bk


def main():
    df = pd.read_csv(DS)
    print(f"{'pair':<14}{'SS':<6}{'MMD2u p':>9}{'sig':>5}{'bk_u':>6}"
          f"{'RBF-MMD p':>11}{'sig':>5}{'bk_b':>6}", flush=True)
    out = []
    for ss, c1, c2 in PAIRS:
        A = np.deg2rad(df[(df.condition_group==ss)&(df.codon==c1)][["phi","psi"]].values)
        B = np.deg2rad(df[(df.condition_group==ss)&(df.codon==c2)][["phi","psi"]].values)
        K = kmat(np.vstack([A, B]))
        _, pu = perm_p(K, len(A), len(B), "u")
        _, pb = perm_p(K, len(A), len(B), "b")
        nmin = min(len(A), len(B)); kg = [k for k in [0,1,2,3,5,8,12,20] if k <= nmin-2]
        bku = breakdown(A, B, THR[ss], kg, "u")[1] if pu <= THR[ss] else None
        bkb = breakdown(A, B, THR[ss], kg, "b")[1] if pb <= THR[ss] else None
        su, sb = pu <= THR[ss], pb <= THR[ss]
        print(f"{c1+':'+c2.split('-')[1]:<14}{ss:<6}{pu:>9.4f}{('yes' if su else 'no'):>5}"
              f"{str(bku):>6}{pb:>11.4f}{('yes' if sb else 'no'):>5}{str(bkb):>6}", flush=True)
        out.append(dict(SS=ss, pair=f"{c1}:{c2}", mmd2u_p=round(pu,5), mmd2u_sig=su, mmd2u_bk=bku,
                        rbfmmd_p=round(pb,5), rbfmmd_sig=sb, rbfmmd_bk=bkb))
    pd.DataFrame(out).to_csv("out/pnas-2026-repro/mmd_breakdown.csv", index=False)
    print("\nsaved: out/pnas-2026-repro/mmd_breakdown.csv")


if __name__ == "__main__":
    main()
