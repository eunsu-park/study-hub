"""
Stan and CmdStanPy Examples
- Stan model writing, CmdStanPy interface, diagnostics
"""
import numpy as np
import os
import tempfile


def write_stan_model():
    """Write and display a Stan model for normal inference."""
    stan_code = """
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  sigma ~ normal(0, 5);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu, sigma);
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
  }
}
"""
    model_path = os.path.join(tempfile.gettempdir(), "normal_model.stan")
    with open(model_path, "w") as f:
        f.write(stan_code)
    print(f"Stan model written to: {model_path}")
    print("Stan code:")
    print(stan_code)
    return model_path


def run_stan_model():
    """Run Stan model via CmdStanPy (requires cmdstanpy installed)."""
    try:
        import cmdstanpy
        model_path = write_stan_model()
        model = cmdstanpy.CmdStanModel(stan_file=model_path)
        np.random.seed(42)
        data = np.random.normal(5.0, 2.0, 100)
        fit = model.sample(data={"N": len(data), "y": data.tolist()},
                          chains=4, iter_sampling=2000, seed=42)
        print(fit.summary())
    except ImportError:
        print("CmdStanPy not installed. Run: pip install cmdstanpy && install_cmdstan")


if __name__ == "__main__":
    write_stan_model()
    print("\nTo run the model, install cmdstanpy and call run_stan_model()")
