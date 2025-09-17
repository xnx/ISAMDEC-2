import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.constants import h, c, k as kB
    # The second radiation constant, cm.K.
    c2 = h * c * 100 / kB
    return c2, find_peaks, np, plt


@app.cell
def _(np):
    # Load in the spectrum and normalize the intensities.
    nugrid, spec = np.loadtxt("HI-rotational-spectrum.txt", unpack=True)
    spec /= max(spec)
    return nugrid, spec


@app.cell
def _(nugrid, plt, spec):
    plt.plot(nugrid, spec)
    plt.xlabel(r"$\tilde{\nu}\;/\mathrm{cm^{-1}}$")
    plt.ylabel(r"$I_\mathrm{abs}$ (arb. units)")
    return


@app.cell
def _(find_peaks, np, nugrid, spec):
    # Find the peak positions (transition wavenumbers, in cm-1).
    peak_indices, _ = find_peaks(spec)
    peak_positions = nugrid[peak_indices]
    # How many transitions are there?
    Jmax = len(peak_indices) - 1
    Jgrid = np.arange(Jmax+1)
    peak_positions
    return Jgrid, peak_indices, peak_positions


@app.cell
def _(c2, np, peak_indices, peak_positions, spec):
    # Find the most intense transition J -> J+1.
    most_intense_J = np.argmax(spec[peak_indices])
    most_intense_J
    # The transitions are approximately separated by 2B.
    Bapprox = (peak_positions[1] - peak_positions[0]) / 2
    # Retreive the approximate temperature of the spectrum.
    Tapprox = (most_intense_J / np.sqrt(3) + 0.5)**2 * 2 * Bapprox * c2
    print(f"Approximate temperature, Tapprox = {Tapprox:.1f} K")
    return (Tapprox,)


@app.cell
def _(c2, np):
    def calc_E(J, prms):
        """Calculate the energy of level J from its rotational parameters."""
        B, D = prms
        E = B * J * (J + 1) - D * (J * (J+1))**2
        return E
    
    def calc_nu(J, prms):
        """Calculate the transition wavenumber for J -> J+1."""
        return calc_E(J+1, prms) - calc_E(J, prms)

    def calc_intens(J, B, T):
        """Calculate the approximate relative intensity for transition J -> J+1."""
        intensities = (J+1)**2 * np.exp(-B * J * (J+1) * c2 / T) * (1 - np.exp(-2*B*(J+1) * c2 / T))
        return intensities / max(intensities)
    return calc_intens, calc_nu


@app.cell
def _(np):
    def fit_spec(X, peak_positions):
        b, resid, rank, s = np.linalg.lstsq(X, peak_positions)
        print("Fit rms =", resid)
        return b
    return (fit_spec,)


@app.cell
def _(
    Jgrid,
    Tapprox,
    calc_intens,
    calc_nu,
    fit_spec,
    np,
    nugrid,
    peak_positions,
    plt,
    spec,
):
    # First fit the Rigid Rotor model.
    def X_rigid_rotor():
        return np.atleast_2d(2 * (Jgrid + 1)).T
    
    b_rr = fit_spec(X_rigid_rotor(), peak_positions)
    B_rr = b_rr[0]
    print("Rigid Rotor model")
    print(f"B = {B_rr:.6f} cm-1")
    intensities = calc_intens(Jgrid, B_rr, Tapprox)
    plt.plot(nugrid, spec)
    plt.plot(calc_nu(Jgrid, (B_rr, 0)), intensities, 'x')
    return (intensities,)


@app.cell
def _(
    Jgrid,
    calc_nu,
    fit_spec,
    intensities,
    np,
    nugrid,
    peak_positions,
    plt,
    spec,
):
    # Now include centrifugal distortion.
    def X_centifugal_distortion():
        return np.array([2 * (Jgrid + 1), -4*(Jgrid+1)**3]).T

    b_cd = fit_spec(X_centifugal_distortion(), peak_positions)
    B_cd, D_cd = b_cd
    print("Rigid Rotor model")
    print(f"B = {B_cd:.6f} cm-1")
    print(f"D = {D_cd:.8f} cm-1")
    plt.plot(nugrid, spec)
    plt.plot(calc_nu(Jgrid, (B_cd, D_cd)), intensities, 'x')
    return (B_cd,)


@app.cell
def _(B_cd, Jgrid, Tapprox, calc_intens, np, peak_indices, spec):
    from scipy.optimize import leastsq

    def f(prms):
        return np.sum((calc_intens(Jgrid, B_cd, prms[0]) - spec[peak_indices])**2)

    ret = leastsq(f, [Tapprox,], args=())
    T = ret[0][0]
    print(f"Fitted temperature, T = {T:.1f} K")
    return


if __name__ == "__main__":
    app.run()
