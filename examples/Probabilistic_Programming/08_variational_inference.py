"""
Variational Inference Examples
- ELBO, mean-field VI, reparameterization trick
"""
import numpy as np
from scipy import stats


class MeanFieldVI:
    """Mean-field Gaussian VI via stochastic gradient ascent on ELBO."""
    def __init__(self, log_joint_fn, n_params, lr=0.01):
        self.log_joint = log_joint_fn
        self.d = n_params
        self.lr = lr
        self.mu = np.zeros(n_params)
        self.log_sigma = np.zeros(n_params)

    def fit(self, n_steps=1000, n_samples=50):
        for step in range(n_steps):
            sigma = np.exp(self.log_sigma)
            eps = np.random.normal(size=(n_samples, self.d))
            samples = self.mu + sigma * eps
            log_joints = np.array([self.log_joint(s) for s in samples])
            log_qs = np.sum(-0.5*eps**2 - self.log_sigma - 0.5*np.log(2*np.pi), axis=1)
            advantages = log_joints - log_qs
            self.mu += self.lr * np.mean(advantages[:, None] * eps / sigma, axis=0)
            self.log_sigma += self.lr * 0.1 * np.mean(advantages[:, None] * (eps**2 - 1), axis=0)
            if step % 200 == 0:
                print(f"Step {step}: ELBO≈{advantages.mean():.2f}, mu={self.mu.round(3)}")
        return self


if __name__ == "__main__":
    np.random.seed(42)
    n = 50
    x = np.random.randn(n)
    y = 2.5 * x - 1.0 + np.random.normal(0, 0.5, n)

    def log_joint(params):
        w, b, log_s = params
        s = np.exp(log_s)
        lp = stats.norm.logpdf(w, 0, 5) + stats.norm.logpdf(b, 0, 5)
        lp += stats.halfnorm.logpdf(s, scale=5) + log_s
        lp += np.sum(stats.norm.logpdf(y, w*x + b, s))
        return lp

    vi = MeanFieldVI(log_joint, 3, lr=0.005)
    vi.fit(1000, 100)
    print(f"\nVI: w={vi.mu[0]:.3f}, b={vi.mu[1]:.3f}, sigma={np.exp(vi.mu[2]):.3f}")
    print(f"True: w=2.5, b=-1.0, sigma=0.5")
