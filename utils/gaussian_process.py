import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax as ox
import plotly.graph_objects as go
from gpjax.parameters import Parameter
from jax import config
from plotly.colors import convert_colors_to_same_type, unlabel_rgb

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


def rgba(r, g, b, a):
    return f"rgba({r}, {g}, {b}, {a:.2f})"


def plot_gp(
    grid, pred_mean, pred_sigma, trace_name: str = None, trace_color: str = "#22577A"
) -> go.Figure:
    trace_rgb, _ = convert_colors_to_same_type(trace_color, colortype="rgb")
    (r, g, b) = unlabel_rgb(trace_rgb[0])
    grid = grid[:, 0]
    fig = go.Figure()
    if trace_name is None:
        trace_name = ""
    fig.add_scatter(
        name=f"Mean {trace_name}",
        showlegend=False,
        x=grid,
        y=pred_mean,
        mode="lines",
        line=dict(
            color=rgba(r, g, b, 1.0),
            width=3,
        ),
    )
    fig.add_scatter(
        name="Upper Bound",
        x=grid,
        y=pred_mean + pred_sigma,
        mode="lines",
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False,
    )
    fig.add_scatter(
        name="Lower Bound",
        x=grid,
        y=pred_mean - pred_sigma,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode="lines",
        fillcolor=rgba(r, g, b, 0.4),
        fill="tonexty",
        showlegend=False,
    )
    fig.add_hline(y=0, line=dict(color="black", dash="dash", width=2))
    fig = fig.update_layout(
        margin=dict(t=40, b=20, l=10, r=10),
        template="plotly_white",
        font=dict(size=14, color="black", family="Merriweather"),
        width=600,
        height=400,
    )
    fig = fig.update_annotations(
        font=dict(size=18, color="black", family="Merriweather"),
    )
    fig = fig.update_xaxes(matches="x", title="Character index")
    fig = fig.update_yaxes(matches="y", title="Sentiment")
    return fig
