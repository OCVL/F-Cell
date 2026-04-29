import numpy as np
from scipy.sparse import diags, eye, spdiags
from scipy.sparse.linalg import spsolve


def differ(x, dt):
    """Numerical derivative using central differences with forward/backward at edges."""
    dx = np.zeros_like(x)
    dx[0]    = (x[1] - x[0]) / dt          # forward difference
    dx[-1]   = (x[-1] - x[-2]) / dt        # backward difference
    dx[1:-1] = (x[2:] - x[:-2]) / (2*dt)  # central difference
    return dx


def acmd(s, fs, eIF, alpha0=1e-3, beta=1e-4, tol=1e-9):
    """
    https://www.sciencedirect.com/science/article/pii/S0022460X18306771?via%3Dihub
    Adaptive Chirp Mode Decomposition (ACMD)
    Python translation of MATLAB code by Shiqian Chen and Zhike Peng.

    Parameters
    ----------
    s      : array_like       Signal to decompose, 1D array of length N
    fs     : float            Sampling frequency (Hz)
    eIF    : array_like       Initial instantaneous frequency estimate, 1D array of length N
    alpha0 : float            Bandwidth penalty — smaller = narrower bandwidth
    beta   : float            IF smoothness penalty — smaller = smoother IF
    tol    : float            Convergence tolerance (default 1e-8)

    Returns
    -------
    IF_est : np.ndarray       Estimated instantaneous frequency
    IA_est : np.ndarray       Estimated instantaneous amplitude
    s_est  : np.ndarray       Estimated signal mode
    """
    s   = np.asarray(s, dtype=float)
    eIF = np.asarray(eIF, dtype=float).copy()

    N  = len(eIF)
    t  = np.arange(N) / fs
    dt = 1.0 / fs

    # --- Second-order difference matrix (N-2 x N) ---
    e   = np.ones(N)
    D2  = diags([e[:-2], -2*e[:-1], e], offsets=[0, 1, 2],
                shape=(N-2, N), format='csc')
    D2tD2 = D2.T @ D2   # (N x N)

    # --- Block matrix for joint [cos, sin] demodulated signal penalty ---
    from scipy.sparse import bmat
    zero_block = diags([np.zeros(N-2)], offsets=[0], shape=(N-2, N), format='csc')
    phi    = bmat([[D2, zero_block], [zero_block, D2]], format='csc')     # (2(N-2) x 2N)
    phitphi = phi.T @ phi                                                  # (2N x 2N)

    iternum    = 300
    IF_history = np.zeros((iternum, N))
    s_history  = np.zeros((iternum, N))
    y_history  = np.zeros((iternum, 2*N))

    alpha = alpha0
    sDif  = tol + 1.0
    iter_ = 0

    while sDif > tol and iter_ < iternum:

        # --- Build kernel matrix from current IF estimate ---
        phase = 2 * np.pi * np.cumsum(eIF) * dt   # cumulative trapezoid approximation
        cosm  = np.cos(phase)
        sinm  = np.sin(phase)

        # Sparse diagonal matrices
        Cm = diags(cosm, 0, shape=(N, N), format='csc')
        Sm = diags(sinm, 0, shape=(N, N), format='csc')

        # Kernel: (N x 2N)
        from scipy.sparse import hstack
        Ker    = hstack([Cm, Sm], format='csc')
        KtK    = Ker.T @ Ker                        # (2N x 2N)

        # --- Update demodulated signal ---
        lhs = (1.0 / alpha) * phitphi + KtK         # (2N x 2N)
        rhs = Ker.T @ s                              # (2N,)
        ym  = spsolve(lhs, rhs)                      # (2N,)

        # Reconstruct signal mode
        si = Ker @ ym                                # (N,)
        s_history[iter_, :]  = si
        y_history[iter_, :]  = ym

        # --- Update instantaneous frequency ---
        ycm = ym[:N]
        ysm = ym[N:]

        ycm_dot = differ(ycm, dt)
        ysm_dot = differ(ysm, dt)

        # IF increment via arctangent demodulation
        denom   = ycm**2 + ysm**2
        denom   = np.where(denom < 1e-12, 1e-12, denom)   # avoid division by zero
        deltaIF = (ycm * ysm_dot - ysm * ycm_dot) / denom / (2 * np.pi)

        # Smooth IF increment
        smoother = (1.0 / beta) * D2tD2 + eye(N, format='csc')
        deltaIF  = spsolve(smoother, deltaIF)

        # Update IF
        eIF = eIF - deltaIF
        IF_history[iter_, :] = eIF

        # --- Convergence check ---
        if iter_ > 0:
            prev = s_history[iter_-1, :]
            norm_prev = np.linalg.norm(prev)
            if norm_prev > 0:
                sDif = (np.linalg.norm(si - prev) / norm_prev) ** 2
            else:
                sDif = 0.0

        iter_ += 1

    # --- Extract final estimates ---
    final = iter_ - 1
    IF_est = IF_history[final, :]
    s_est  = s_history[final, :]

    ycm_f  = y_history[final, :N]
    ysm_f  = y_history[final, N:]
    IA_est = np.sqrt(ycm_f**2 + ysm_f**2)

    return IF_est, IA_est, s_est