import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from scipy.constants import h, c, k as kB, u
    from scipy.interpolate import interp1d
    from scipy.optimize import newton
    c2 = h * c * 100 / kB   # Second radiation constant, cm.K
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 14})
    rc('text', usetex=True)
    return c, c2, interp1d, kB, np, pd, plt, u


@app.cell(hide_code=True)
def _(c, c2, interp1d, kB, np, pd):
    class ParMeta:
        widths = (2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 15, 6, 12, 1, 7, 7)
        names = ('molec_id', 'iso_id', 'nu', 'S', 'A', 'gamma_air', 'gamma_self',
                 'Epp', 'n_air', 'delta_air', 'Vp', 'Vpp', 'Qp', 'Qpp', 'Ierr',
                 'Iref', 'flag', 'gp', 'gpp')
        dtypes = {0: int, 1: int, 2: np.float64, 3: np.float64, 4:np.float64,
                  5: np.float64, 6: np.float64, 7: np.float64, 8: np.float64,
                  9: np.float64, 10: str, 11: str, 12: str, 13: str,
                  14: str, 15: str, 16: str, 17: np.float64, 18: np.float64}
        Tref = 296

        def __init__(self):
            pass

        def read_par(self, par_name):
            df = pd.read_fwf(par_name, widths=self.widths, header=None, names=self.names,
                     delim_whitespace=True, dtype=self.dtypes)
            return df

    def read_q(q_filename):
        Tgrid, qgrid = np.loadtxt(q_filename, unpack=True)
        q = interp1d(Tgrid, qgrid)
        return q(ParMeta.Tref), q

    par_meta = ParMeta()
    par_df = par_meta.read_par('CO-all.par')
    par_df['Vp'] = par_df['Vp'].astype(int)
    par_df['Vpp'] = par_df['Vpp'].astype(int)
    qref, q = read_q('CO-q.txt')

    def get_S(T):
        S = (
            par_df['S'].values
             * qref / q(T)
             * np.exp(-c2 * par_df.Epp.values / T)
             / np.exp(-c2 * par_df.Epp.values / ParMeta.Tref)
             * (1 - np.exp(-c2 * par_df.nu.values / T))
             / (1 - np.exp(-c2 * par_df.nu.values / ParMeta.Tref))
            )
        return S

    FAC = np.sqrt(np.log(2))
    FAC2 = FAC/np.sqrt(np.pi)

    def gaussian(x0, alpha, x):
        """Normalized gaussian function centred at x0 with HWHM alpha.""" 

        return FAC2 / alpha * np.exp(-(FAC * (x - x0) / alpha)**2)

    def get_alpha(T, m, nu):
        return np.sqrt(2 * kB * T * FAC / m) / c * nu

    def lorentzian(x0, gamma, x):
        """Normalized Lorentzian function centred on x0 with HWHM gamma."""
        return gamma / np.pi * gamma**2 / ((x - x0)**2 + gamma**2)
    return get_S, lorentzian, par_df


@app.cell
def _(get_S, lorentzian, np, par_df, plt, u):
    def plot_spec(T, p, numin, numax):
        """Plot spec from numin to numax at temperature T (K) and pressure p (atm)."""
    
        m = (12+16) * u
        par_df['S_T'] = get_S(T)
        par_df['gamma_p'] = p * par_df['gamma_air']
        Npts = int((numax - numin) / min(par_df['gamma_p']) * 10)
        print(f"Npts = {Npts}")
        nugrid = np.linspace(numin, numax, Npts)
        spec = np.zeros(nugrid.shape)
        for line in par_df.iterrows():
            nu0 = line[1]['nu']
            S = line[1]['S_T']
            gamma = line[1]['gamma_p']
            #spec += S * gaussian(nu0, alpha, nugrid)
            spec += S * lorentzian(nu0, gamma, nugrid)

        # Uncomment for a stick spectrum only (no broadening).
        #plt.stem(par_df['nu'], par_df['S_T'])
        plt.plot(nugrid, spec, c="#32746d")
        plt.xlabel(r"$\tilde{\nu}\,/\mathrm{cm^{-1}}$")
        plt.ylabel(r"$\sigma\,/\mathrm{cm\,(molec\,cm^{-2})}$")
        plt.show()
    return (plot_spec,)


@app.cell
def _(plot_spec):
    # e.g. Plot the vibrational spectrum of CO from 4050-4400 cm-1
    # at 300 K, 10 atm.
    plot_spec(300, 10, 4050, 4400)
    return


@app.cell
def _(par_df, plot_spec):
    def plot_band(vpp, vp, T, p):
        """Plot the spectrum at T, p for a given vibrational band vpp -> vp."""
    
        df = par_df[(par_df['Vp']==vp) & (par_df['Vpp']==vpp)]
        numin, numax = df['nu'].agg(['min', 'max'])
        print(f"nu = {numin:.1f} - {numax:.1f} cm-1")
        plot_spec(T, p, numin, numax)
    return (plot_band,)


@app.cell
def _(plot_band):
    plot_band(0, 2, 1000, 20)
    return


if __name__ == "__main__":
    app.run()
