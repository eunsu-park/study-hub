"""
MCMC Fundamentals Examples
- Metropolis-Hastings, Gibbs sampling, convergence diagnostics, HMC
"""
import numpy as np
from scipy import stats


class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler."""
    def __init__(self, log_target, proposal_std=1.0):
        self.log_target = log_target
        self.proposal_std = proposal_std

    def sample(self, initial, n_samples, burn_in=1000):
        ndim = len(initial) if hasattr(initial, '__len__') else 1
        scalar = ndim == 1
        if scalar:
            initial = np.array([initial])
        samples = np.zeros((n_samples + burn_in, ndim))
        samples[0] = initial
        current_log_p = self.log_target(initial)
        n_accepted = 0
        for t in range(1, n_samples + burn_in):
            proposal = samples[t-1] + np.random.normal(0, self.proposal_std, size=ndim)
            proposed_log_p = self.log_target(proposal)
            if np.log(np.random.uniform()) < proposed_log_p - current_log_p:
                samples[t] = proposal
                current_log_p = proposed_log_p
                n_accepted += 1
            else:
                samples[t] = samples[t-1]
        acc_rate = n_accepted / (n_samples + burn_in)
        result = samples[burn_in:]
        return result.flatten() if scalar else result, acc_rate


def gibbs_normal(data, n_samples=10000, burn_in=2000):
    """Gibbs sampler for Normal model."""
    mu_0, tau_0_sq, a, b = 0.0, 100.0, 2.0, 1.0
    n = len(data)
    mu, sigma_sq = data.mean(), data.var()
    samples_mu = np.zeros(n_samples)
    samples_sigma = np.zeros(n_samples)
    for t in range(n_samples + burn_in):
        prec = 1/tau_0_sq + n/sigma_sq
        mu_post = (mu_0/tau_0_sq + n*data.mean()/sigma_sq) / prec
        mu = np.random.normal(mu_post, 1/np.sqrt(prec))
        a_post = a + n/2
        b_post = b + np.sum((data - mu)**2)/2
        sigma_sq = 1/np.random.gamma(a_post, 1/b_post)
        if t >= burn_in:
            samples_mu[t - burn_in] = mu
            samples_sigma[t - burn_in] = np.sqrt(sigma_sq)
    return samples_mu, samples_sigma


def compute_rhat(chains):
    """Compute R-hat (potential scale reduction factor)."""
    n_chains, n_samples = chains.shape
    W = np.mean(np.var(chains, axis=1, ddof=1))
    B = n_samples * np.var(np.mean(chains, axis=1), ddof=1)
    var_hat = (1 - 1/n_samples) * W + (1/n_samples) * B
    return np.sqrt(var_hat / W)


if __name__ == "__main__":
    np.random.seed(42)
    # MH on mixture
    log_target = lambda t: np.log(0.3*stats.norm.pdf(t[0],-2,0.5) + 0.7*stats.norm.pdf(t[0],3,1) + 1e-300)
    mh = MetropolisHastings(log_target, proposal_std=1.5)
    samples, acc = mh.sample(initial=0.0, n_samples=20000)
    print(f"MH: acceptance={acc:.3f}, mean={samples.mean():.3f}, std={samples.std():.3f}")

    # Gibbs
    data = np.random.normal(5.0, 2.0, 50)
    mu_s, sig_s = gibbs_normal(data)
    print(f"Gibbs: mu={mu_s.mean():.3f} (true 5.0), sigma={sig_s.mean():.3f} (true 2.0)")

    # R-hat
    chains = np.array([mh.sample(s, 5000)[0] for s in [-5, 0, 5, 10]])
    print(f"R-hat: {compute_rhat(chains):.4f}")
