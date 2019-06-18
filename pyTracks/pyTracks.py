import ketcham
import matplotlib.pyplot as plt
import numpy as np
import pint
from .annealing import KetchamEtAl
from .utilities import draw_from_distrib

u = pint.UnitRegistry()


class ForwardModel(object):

    def __init__(self, history, annealing_model):
        self.annealing_model = annealing_model
        self.history = history

    def solve(self):
        pass


class KetchamModel(ForwardModel):

    def __init__(self, history, initial_track_length=None,
                 kinetic_parameter_type=None, kinetic_parameter_value=0.0,
                 projected_track=False):

        super(KetchamModel, self).__init__(history, KetchamEtAl)

        self.input_initial_track_length = initial_track_length
        self.kinetic_parameter_type = kinetic_parameter_type
        self.kinetic_parameter_value = kinetic_parameter_value
        self.projected_track = projected_track

    def get_initial_track_length(self):
        """
            Returns the initial track length for the population based on
            Carlson and Donelick experiments depending on the kinetic
            parameter type. If no kinetic is entered return track length
            enterer by user
        """
        if self.input_initial_track_length:
            return self.input_initial_track_length

        return initial_track_length(self.kinetic_parameter_type,
                                    self.kinetic_parameter_value,
                                    self.projected_track)

    def calculate_track_lengths(self, nbins=200, use_confined=False,
                                min_length=2.15):
        time = (self.history.time * u.megayear).to(u.seconds).magnitude
        temperature = self.history.temperature
        kinpar = {"ETCH_PIT_LENGTH": 0, "CL_PFU": 1,
                  "OH_PFU": 2, "CL_WT_PCT": 3}

        track_l0 = self.get_initial_track_length()

        kinetic_parameter_type = kinpar[self.kinetic_parameter_type]
        reduced_length, pdf_axis, pdf, cdf = ketcham.track_lengths(
            time, temperature, kinetic_parameter_type,
            self.kinetic_parameter_value, track_l0, min_length, nbins,
            self.projected_track, use_confined)

        self.pdf_axis = pdf_axis
        self.pdf = pdf * 0.1

        return self.pdf_axis, self.pdf

    def plot_track_length_density(self):
        plt.plot(self.pdf_axis, self.pdf)
        plt.xlabel("Length (microns)")
        plt.ylabel("Density")

    def get_synthetic_lengths(self, ntl=100):
        self.tls = draw_from_distrib(self.pdf_axis, self.pdf, ntl)
        self.mtl = (float(sum(self.tls))/len(self.tls) if len(self.tls) > 0 else float('nan'))
        self.mtl_sd = np.std(self.tls)

    def plot_track_histogram(self):
        plt.hist(self.tls)
        plt.xlim(0, 20)
        plt.xlabel("Length (microns)")
        plt.ylabel("counts")

    def solve(self):
        pass


def initial_track_length(kinetic_parameter_type,
                         kinetic_parameter_value,
                         use_projected_track=False):
    """

        Returns the initial track length for the population based
        on the apatite kinetics, using data from experiment H0
        by W.D.Carlson and R.A.Donelick (UT Austin)

        kinetic_parameter_type: ETCH_PIT_LENGTH, CL_PFU, OH_PFU, CL_WT_PFU
        kinetic_parameter_value: value of the kinetic parameter
        use_projected_track: use projected track? default is False

    """

    unprojected = {"ETCH_PIT_LENGTH": {"m": 0.283, "b": 15.63},
                   "CL_PFU": {"m": 0.544, "b": 16.18},
                   "OH_PFU": {"m": 0.0, "b": 16.18},
                   "CL_WT_PCT": {"m": 0.13824, "b": 16.288}}

    projected = {"ETCH_PIT_LENGTH": {"m": 0.205, "b": 16.10},
                 "CL_PFU": {"m": 0.407, "b": 16.49},
                 "OH_PFU": {"m": 0.000, "b": 16.57},
                 "CL_WT_PCT": {"m": 0.17317, "b": 16.495}}

    if use_projected_track:
        if kinetic_parameter_type not in projected.keys():
            raise ValueError("""{0} is not a valid kinetic parameter
                             type""".format(kinetic_parameter_type))
        else:
            m = projected[kinetic_parameter_type]["m"]
            b = projected[kinetic_parameter_type]["b"]
    else:
        if kinetic_parameter_type not in unprojected.keys():
            raise ValueError("""{0} is not a valid kinetic parameter
                             type""".format(kinetic_parameter_type))
        else:
            m = unprojected[kinetic_parameter_type]["m"]
            b = unprojected[kinetic_parameter_type]["b"]

    return m * kinetic_parameter_value + b


class Sample(object):

    def __init__(self, name, counts=[], AFT=None, AFT_error=None,
                 tls=[], zeta=None, rhod=None):
        self.name = name
        self.counts = counts
        self.nc = len(counts)

        if counts:
            self.ns, self.ni = list(zip(counts))

        self.AFT = AFT
        self.AFT_error = AFT_error
        self.tls = tls
        self.zeta = zeta
        self.rhod = rhod

    def write_mtx_file(self, filename):
        write_mtx_file(filename, self.name, self.AFT, self.AFT_error, self.tls,
                       self.ns, self.ni, self.zeta, self.rhod)


class Synthetic(Sample):

    def __init__(self, name=None, nc=30, ntl=100, history=None):
        Sample.__init__(self, name=name)
        if not self.name:
            self.name = "Synthetic"
        if not self.zeta:
            self.zeta = 323.
        if not self.rhod:
            self.rhod = 1.e6
        self.nc = nc
        self.ntl = ntl
        self.history = history
        if history:
            self.ketcham_model()
            self.synthetic_counts()
            self.synthetic_lengths()

    def ketcham_model(self, alo=16.3):
        """Return Apatite Fission Track Age (AFTA) and Track Length
        distribution using Ketcham et al. 1999 annealing model.

        Parameter:
        ---------
        alo -- Initial track length
        """
        data = KetchamModel(self.history, alo=alo)
        # Process Fission Track Distribution
        # distribution range from 0 to 20 microns
        # We have 200 values.
        vals, fdist = data["Fission Track length distribution"]
        probs = [i for i in fdist]

        self.AFT = data["Final Age"]
        self.AFT_error = self.AFT*0.05
        self.Oldest_Age = data["Oldest Age"]
        self.MTL = data["Mean Track Length"]
        self.TLD = fdist
        self.reDensity = data["redDensity"]
        self.rho = self.reDensity
        self.bins = vals
        return

    def synthetic_counts(self):
        data = generate_synthetic_counts(self.rho, self.nc)
        self.ns = data["Spontaneous tracks (Ns)"]
        self.ni = data["Induced tracks (Ni)"]
        self.counts = list(zip(self.ns, self.ni))
        return

    def synthetic_lengths(self):
        self.tls = draw_from_distrib(self.bins, self.TLD, self.ntl)
        self.mtl = (float(sum(self.tls))/len(self.tls)
                    if len(self.tls) > 0 else float('nan'))
        self.mtl_sd = np.std(self.tls)

    def plot_predicted_TLD(self):
        plt.plot(self.bins, self.TLD)
        plt.xlabel("Length (microns)")
        plt.ylabel("Density")

    def plot_history(self):
        t = self.history.time
        T = self.history.Temperature
        plt.plot(t, T)
        plt.ylim((max(T)+10, min(T)-10))
        plt.xlim((max(t), min(t)))
        plt.xlabel("time (Ma)")
        plt.ylabel("Temperature (Celcius)")

    def plot_track_histogram(self):
        plt.hist(self.tls)
        plt.xlim(0, 20)
        plt.xlabel("Length (microns)")
        plt.ylabel("counts")


def write_mtx_file(filename, sample_name, FTage, FTage_error, TL, NS, NI, zeta, rhod):

    f = open(filename, "w")
    f.write("{name:s}\n".format(name=sample_name))
    f.write("{value:s}\n".format(value=str(-999)))
    f.write("{nconstraints:d} {ntl:d} {nc:d} {zeta:5.1f} {rhod:12.1f} {totco:d}\n".format(
             nconstraints=0, ntl=len(TL), nc=NS.size, zeta=zeta, rhod=rhod,
            totco=2000))
    f.write("{age:5.1f} {age_error:5.1f}\n".format(age=FTage,
                                                   age_error=FTage_error))
    TLmean = (float(sum(TL))/len(TL) if len(TL) > 0 else float('nan'))
    TLmean_sd = np.std(TL)

    f.write("{mtl:5.1f} {mtl_error:5.1f}\n".format(mtl=TLmean,
                                                   mtl_error=TLmean*0.05))
    f.write("{mtl_std:5.1f} {mtl_std_error:5.1f}\n".format(mtl_std=TLmean_sd,
                                                           mtl_std_error=TLmean_sd*0.05))
    for i in range(NS.size):
        f.write("{ns:d} {ni:d}\n".format(ns=NS[i], ni=NI[i]))

    for track in TL:
        f.write("{tl:4.1f}\n".format(tl=track))

    f.close()
    return 0


def generate_synthetic_counts(rho, Nc=30):
    """Generate Synthetic AFT data.

    Parameters:
    rho : track density
    Nc : Number of crystals

    """
    # Probability in binomial distribution
    prob = rho / (1. + rho)

    # For Nc crystals, generate synthetic Ns and Ni count data using binomial
    # distribution, conditional on total counts Ns + Ni, sampled randomly with
    # a maximum of 1000.
    # Nc is the number of
    NsNi = np.random.randint(5, MAXCOUNT, Nc)
    Ns = np.array([drawbinom(I, prob) for I in NsNi])
    Ni = NsNi - Ns
    return {"Spontaneous tracks (Ns)": Ns, "Induced tracks (Ni)": Ni}


def calculate_central_age(Ns, Ni, zeta, seZeta, rhod, seRhod, sigma=0.15):
    """Function to calculate central age."""

    # Calculate mj
    lbda = 1.55125e-10
    m = Ns+Ni
    p = Ns/m

    theta = sum(Ns)/sum(m)

    for i in range(0, 30):
        w = m/(theta*(1-theta)+(m-1)*theta**2*(1-theta)**2*sigma**2)
        # Calculate new value of sigma and theta
        sigma = sigma*sqrt(sum(w**2*(p-theta)**2)/sum(w))
        theta = sum(w*p)/sum(w)

    t = (1/lbda)*log(1+1/2*lbda*zeta*rhod*(theta)/(1-theta))
    se = sqrt(1/(theta**2*(1-theta)**2*sum(w))+(seRhod/rhod)**2+(seZeta/zeta)**2)*t

    return {"Central":t/1e6, "2 sigma Error": 2*se/1e6, "Dispersion":sigma*100}


def calculate_pooled_age(Ns, Ni, zeta, seZeta, rhod, seRhod):

    # Calculate mj
    lbda = 1.55125e-10
    sigma = 0
    m = Ns+Ni
    p = Ns/m

    theta = sum(Ns)/sum(m)

    for i in range(0, 30):
        w = m/(theta*(1-theta)+(m-1)*theta**2*(1-theta)**2*sigma**2)
        theta = sum(w*p)/sum(w)

    t = (1/lbda)*log(1+1/2*lbda*zeta*rhod*(theta)/(1-theta))
    #se<-sqrt(1/sum(Ns)+1/sum(Ni)+(seRhod/rhod)**2+(seZeta/zeta)**2)*t

    return {"Pooled Age": t/1e6, "2 sigma Error": 2*se/1e6}


def calculate_single_grain_ages(Ns, Ni, rhod, zeta, g=0.5, trf="Linear"):
    # Total Decay constant for 238U
    lbda = 1.55125e-10

    # Linear Transformation
    if (trf == "linear"):
        z = 1/lbda*log(1+g*zeta*lbda*rhod*(Ns/Ni))
        sez = z*np.sqrt(1/Ns + 1/Ni)
        z0 = sum(z/sez**2) / sum(1/sez**2)

    # Logarithmic transformation
    if trf == "Log":
        z = log(g*zeta*lbda*rhod*(Ns/Ni))
        sez = z*sqrt(1/Ns+1/Ni)
        z0 = log(g*zeta*lbda*rhod*(sum(Ns)/sum(Ni)))

    # Arcsine
    if trf == "arcsine":
        z = asin(sqrt((Ns+3/8)/(Ns+Ni+3/4)))
        sez = 1/(2*sqrt(Ns+Ni))
        z0 = asin(sqrt(sum(Ns)/sum(Ns+Ni)))

    Age = z/1e6
    Error = sez / 1e6

    return Age, Error


#def KetchamModel(history, alo=16.3):
#    """Return Apatite Fission Track Age (AFTA) and Track Length
#    distribution using Ketcham et al. 1999 annealing model.
#
#    Parameter:
#    ---------
#    history -- A Thermal history class instance
#    alo -- Initial track length
#    """
#    t = history.time
#    T = history.Temperature
#
#    A = cdll.LoadLibrary(_KETCHAM_)
#    ketcham = A.ketch_main_
#    n = c_int(len(t))
#    n = pointer(n)
#    alo = c_double(alo)
#    t = (c_float*len(t))(*t)
#    T = (c_float*len(T))(*T)
#    alo = pointer(alo)
#    final_age = pointer(c_double())
#    oldest_age = pointer(c_double())
#    fmean = pointer(c_double())
#    fdist = (c_double*200)()
#    dx = 20. / 200.
#    redDensity = pointer(c_double())
#    ketcham(n, t, T, alo, final_age, oldest_age, fmean, fdist, redDensity)
#    return {"Final Age": final_age.contents.value,
#            "Oldest Age": oldest_age.contents.value,
#            "Mean Track Length": fmean.contents.value,
#            "Fission Track length distribution": (np.array([dx/2.0 +
#             dx*float(a) for a in range(200)]),np.array([i*dx for i in fdist])),
#            "redDensity": redDensity.contents.value}
#
