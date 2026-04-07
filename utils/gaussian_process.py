import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
from gpjax.parameters import Parameter
from jax import config

config.update("jax_enable_x64", True)


def fit_gp(arc: list[np.ndarray], token_offsets: list[list[tuple]]) -> tuple:
    """Fits Gaussian Process to a sentiment arc using GPJax and sparse variational GPs"""
    X = []
    Y = []
    for doc_arc, offs in zip(arc, token_offsets):
        X.extend([start for start, _ in offs])
        Y.extend(doc_arc)
    X = np.array(X).astype(np.float64)[:, None]
    Y = np.array(Y).astype(np.float64)[:, None]
    grid = jnp.linspace(0, 15000, 500).reshape(-1, 1)
    meanf = gpx.mean_functions.Zero()
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=len(X))
    kernel = gpx.kernels.RBF()  # 1-dimensional inputs
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    p = prior * likelihood
    q = gpx.variational_families.VariationalGaussian(posterior=p, inducing_inputs=grid)
    D = gpx.Dataset(X=X, y=Y)
    schedule = ox.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=0.02,
        warmup_steps=75,
        decay_steps=4000,
        end_value=0.001,
    )
    opt_posterior, history = gpx.fit(
        model=q,
        # we are minimizing the elbo so we negate it
        objective=lambda p, d: -gpx.objectives.elbo(p, d),
        train_data=D,
        optim=ox.adam(learning_rate=schedule),
        num_iters=4000,
        key=jr.key(42),
        batch_size=64,
        trainable=Parameter,
    )
    latent_dist = opt_posterior(grid)
    predictive_dist = opt_posterior.posterior.likelihood(latent_dist)
    pred_mean = np.array(predictive_dist.mean)
    pred_sigma = np.array(jnp.sqrt(predictive_dist.variance))
    return np.array(grid), (pred_mean, pred_sigma)
