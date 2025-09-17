import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    from scipy.constants import c, k as kB, h
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 18})
    rc('text', usetex=True)
    from matplotlib.colors import Normalize

    c *= 100
    c2 = h * c / kB

    B = 2
    Jmax = 40
    J = np.arange(Jmax + 1)
    E = B * J * (J+1)

    def get_pops(J, T):
        terms = (2*J + 1) * np.exp(-c2 * E / T) 
        return terms / sum(terms)
    return J, Jmax, Normalize, get_pops, np, plt


@app.cell
def _(mo):
    Tmin, Tmax, dT = 30, 500, 10
    Tinit = 300
    temperature_slider = mo.ui.slider(
        start=Tmin,
        stop=Tmax,
        step=dT,
        value=Tinit,
        label="Temperature (K)"
    )
    return Tmax, Tmin, temperature_slider


@app.cell
def _(
    J,
    Jmax,
    Normalize,
    Tmax,
    Tmin,
    get_pops,
    mo,
    np,
    plt,
    temperature_slider,
):
    # Calculate the rotational population distribution based on the slider's current value
    T = temperature_slider.value
    population_distribution = get_pops(J, T)

    # Create the matplotlib plot
    cmap = plt.get_cmap("viridis")
    normalizer = Normalize(vmin=Tmin, vmax=Tmax)
    color = cmap(normalizer(T))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(J, population_distribution, color=color, width=0.8)
    ax.set_xlabel("Rotational Quantum Number, $J$")
    ax.set_ylabel("Relative Population")
    ax.set_title(f"{temperature_slider.value} K")
    ax.set_xticks(np.arange(0, Jmax + 1, 5))
    ax.set_ylim(0, 0.3)
    plt.tight_layout()

    # Return the slider and the plot horizontally stacked
    mo.vstack([temperature_slider, fig])
    return


app._unparsable_cell(
    r"""
    |
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
