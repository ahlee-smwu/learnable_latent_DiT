import numpy as np

# -------------------------------------------------
# Posterior-aware Diag GMM
# -------------------------------------------------

class PosteriorDiagGMM:
    def __init__(self, n_components, dim, reg=1e-6, seed=0):
        self.K = n_components
        self.dim = dim
        self.reg = reg
        self.rng = np.random.RandomState(seed)

    def init_params(self, mu):
        idx = self.rng.choice(mu.shape[0], self.K, replace=False)
        self.m = mu[idx].copy()
        self.s2 = np.ones((self.K, self.dim), dtype=np.float32)
        self.pi = np.ones(self.K, dtype=np.float32) / self.K

    def e_step(self, mu, sigma2):
        N = mu.shape[0]
        log_r = np.zeros((N, self.K), dtype=np.float32)

        for k in range(self.K):
            diff2 = (mu - self.m[k]) ** 2
            log_r[:, k] = (
                np.log(self.pi[k] + 1e-12)
                - 0.5 * np.sum(
                    (diff2 + sigma2) / self.s2[k]
                    + np.log(self.s2[k]),
                    axis=1
                )
            )

        log_r -= log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r)
        r /= r.sum(axis=1, keepdims=True)
        return r

    def m_step(self, mu, sigma2, r):
        Nk = r.sum(axis=0) + 1e-8
        self.pi = Nk / Nk.sum()

        self.m = (r.T @ mu) / Nk[:, None]

        for k in range(self.K):
            diff2 = (mu - self.m[k]) ** 2
            self.s2[k] = (
                (r[:, k][:, None] * (diff2 + sigma2)).sum(axis=0)
                / Nk[k]
            )

        self.s2 = np.maximum(self.s2, self.reg)

    def fit(self, mu, sigma2, iters=50):
        self.init_params(mu)
        for it in range(iters):
            r = self.e_step(mu, sigma2)
            self.m_step(mu, sigma2, r)
            print(f"[EM] iter {it+1}/{iters} done")