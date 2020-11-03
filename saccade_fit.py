import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

# Copied from
# https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

matplotlib.rcParams["figure.figsize"] = (16.0, 12.0)
matplotlib.style.use("ggplot")

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha,
        st.anglit,
        st.arcsine,
        st.beta,
        st.betaprime,
        st.bradford,
        st.burr,
        st.cauchy,
        st.chi,
        st.chi2,
        st.cosine,
        st.cosine,
        st.dgamma,
        st.dweibull,
        st.erlang,
        st.expon,
        st.exponnorm,
        st.exponweib,
        st.exponpow,
        st.f,
        st.fatiguelife,
        st.fisk,
        st.foldcauchy,
        st.foldnorm,
        st.frechet_r,
        st.frechet_l,
        st.genlogistic,
        st.genpareto,
        st.gennorm,
        st.genexpon,
        st.genextreme,
        st.gausshyper,
        st.gamma,
        st.gengamma,
        st.genhalflogistic,
        st.gilbrat,
        st.gompertz,
        st.gumbel_r,
        st.gumbel_l,
        st.halfcauchy,
        st.halflogistic,
        st.halfnorm,
        st.halfgennorm,
        st.hypsecant,
        st.invgamma,
        st.invgauss,
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = (
        dist.ppf(0.01, *arg, loc=loc, scale=scale)
        if arg
        else dist.ppf(0.01, loc=loc, scale=scale)
    )
    end = (
        dist.ppf(0.99, *arg, loc=loc, scale=scale)
        if arg
        else dist.ppf(0.99, loc=loc, scale=scale)
    )

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


DISTRIBUTIONS = [
    st.alpha,
    st.anglit,
    st.arcsine,
    st.beta,
    st.betaprime,
    st.bradford,
    st.burr,
    st.cauchy,
    st.chi,
    st.chi2,
    st.cosine,
    st.cosine,
    st.dgamma,
    st.dweibull,
    st.erlang,
    st.expon,
    st.exponnorm,
    st.exponweib,
    st.exponpow,
    st.f,
    st.fatiguelife,
    st.fisk,
    st.foldcauchy,
    st.foldnorm,
    st.frechet_r,
    st.frechet_l,
    st.genlogistic,
    st.genpareto,
    st.gennorm,
    st.genexpon,
    st.genextreme,
    st.gausshyper,
    st.gamma,
    st.gengamma,
    st.genhalflogistic,
    st.gilbrat,
    st.gompertz,
    st.gumbel_r,
    st.gumbel_l,
    st.halfcauchy,
    st.halflogistic,
    st.halfnorm,
    st.halfgennorm,
    st.hypsecant,
    st.invgamma,
    st.invgauss,
]

# Load data from statsmodels datasets
df1 = pd.read_csv("einar_siri_fixation.tsv", sep="\t")
df2 = pd.read_csv("fixation_data_final.tsv", sep="\t")
df = df1.append(df2)
cols_to_keep = [
    "Recording timestamp",
    "Participant name",
    "Recording name",
    "Gaze point X (MCSnorm)",
    "Gaze point Y (MCSnorm)",
    "Eye movement type",
    "Eye movement type index",
    "Gaze event duration",
    "Presented Media name",
]
df = df[cols_to_keep]
df = df[df["Presented Media name"] == "test1.png"]
df = df[df["Eye movement type"] == "Saccade"]
unique_saccades = df[
    df["Eye movement type index"] != df["Eye movement type index"].shift(1)
]
data = unique_saccades["Gaze event duration"]

# Plot for comparison
plt.figure(figsize=(12, 8))
ax = data.plot(
    kind="hist",
    bins=50,
    density=True,
    alpha=0.5,
    color=list(matplotlib.rcParams["axes.prop_cycle"])[1]["color"],
)
# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(
    f"Saccade durations, all participants\nAttempted fitting of {len(DISTRIBUTIONS)} distributions"
)
ax.set_xlabel("Duration (ms)")
ax.set_ylabel("Frequency")

# Make PDF with best params
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(12, 8))
ax = pdf.plot(lw=2, label="PDF", legend=True)
data.plot(
    kind="hist", bins=50, density=True, alpha=0.5, label="Data", legend=True, ax=ax
)
param_names = (
    (best_dist.shapes + ", loc, scale").split(", ")
    if best_dist.shapes
    else ["loc", "scale"]
)
param_str = ", ".join(
    ["{}={:0.2f}".format(k, v) for k, v in zip(param_names, best_fit_params)]
)
dist_str = "{}({})".format(best_fit_name, param_str)


ax.set_title(
    f"Saccade durations, all participants\nAttempted fitting of {len(DISTRIBUTIONS)} distributions"
)
ax.set_xlabel("Duration (ms)")
ax.set_ylabel("Frequency")
plt.show()
