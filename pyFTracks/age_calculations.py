import numpy as np
from scipy.stats import chi2

def chi_square(Ns, Ni):
    """ Return $\chi^2_{\text stat}$ value and the associate p-value"""
    
    NsNi = np.ndarray((len(Ns), 2))
    NsNi[:, 0] = Ns
    NsNi[:, 1] = Ni
    
    length = len(Ns)
    Ns = sum(Ns)
    Ni = sum(Ni)

    X2 = 0.
    for Nsj, Nij in NsNi:
        X2 += (Nsj*Ni - Nij*Ns)**2 / (Nsj + Nij)

    X2 *= 1.0/(Ns*Ni)
    rv = chi2(length - 1)
    return 1.0 - rv.cdf(X2)

def calculate_central_age(Ns, Ni, zeta, seZeta, rhod, Nd, sigma=0.15):
    """Function to calculate central age."""

    Ns = np.array(Ns)
    Ni = np.array(Ni)

    # Calculate mj
    LAMBDA = 1.55125e-4
    G = 0.5
    m = Ns + Ni
    p = Ns / m

    theta = np.sum(Ns) / np.sum(m)

    for i in range(0, 30):
        w = m / (theta * (1 - theta) + (m - 1) * theta**2 * (1 - theta)**2 * sigma**2)
        sigma = sigma * np.sqrt(np.sum(w**2 * (p - theta)**2) / np.sum(w))
        theta = np.sum(w * p) / np.sum(w)

    t = (1.0 / LAMBDA) * np.log( 1.0 + G * LAMBDA * zeta * rhod * (theta) / (1.0 - theta))
    se = np.sqrt(1 / (theta**2 * (1.0 - theta)**2 * np.sum(w)) + 1.0 / Nd + (seZeta / zeta)**2) * t

    return {"Central": t, "se": se, "sigma": sigma}

def calculate_pooled_age(Ns, Ni, zeta, seZeta, rhod, Nd):

    Ns = np.sum(Ns)
    Ni = np.sum(Ni)

    LAMBDA = 1.55125e-4
    G = 0.5
    t = 1.0 / LAMBDA * np.log(1.0 + G * LAMBDA * zeta * rhod * Ns / Ni)
    se = t * (1.0 / Ns + 1.0 / Ni + 1 / Nd + seZeta**2)**0.5

    return {"Pooled Age": t, "se": se}

def calculate_ages(Ns, Ni, zeta, seZeta, rhod, Nd):

    Ns = np.array(Ns)
    Ni = np.array(Ni)
    Nd = np.array(Nd)

    def true_divide(val):
        with np.errstate(divide='ignore', invalid='ignore'):
            val = np.true_divide(1.0, val)
            val[val == np.inf] = 0
            return np.nan_to_num(val)

    # Calculate mj
    LAMBDA = 1.55125e-4
    G = 0.5
    t = 1.0 / LAMBDA * np.log(1.0 + G * LAMBDA * zeta * rhod * Ns / Ni)

    se = (true_divide(Ns) + true_divide(Ni) + 1.0 / Nd + seZeta**2)**0.5
    se *= t

    return {"Age(s)": t, "se(s)": se}

