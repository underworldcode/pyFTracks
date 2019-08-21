import numpy as np
import pint
from .annealing import KetchamEtAl
from .utilities import draw_from_distrib, drawbinom
from ketcham import ketcham99_annealing_model
from ketcham import ketcham07_annealing_model
from ketcham import sum_population
from ketcham import calculate_model_age
from .plot import Viewer

u = pint.UnitRegistry()

kinpar = {"ETCH_PIT_LENGTH": 0, "CL_PFU": 1,
          "OH_PFU": 2, "CL_WT_PCT": 3}

etchants = {"5.5": 0, "5.0": 1}

class ForwardModel():

    def __init__(self, history, use_projected_track=False,
                 use_confined_track=False, min_length=2.15,
                 length_reduction=0.893):

        self.history = history
        self.use_projected_track = use_projected_track
        self.use_confined_track = use_confined_track
        self.min_length = min_length
        self.length_reduction = length_reduction

    def _get_reduced_length(self, grain, nbins=200):
        return

    def _get_distribution(self, grain, nbins=200):
        time = (self.history.time * u.megayear).to(u.seconds).magnitude
        temperature = self.history.temperature
        
        if self.use_projected_track:
            track_l0 = grain.l0_projected
        else:
            track_l0 = grain.l0

        pdf_axis, pdf, cdf = sum_population(
                time, temperature,
                self.reduced_lengths, self.first_node,
                track_l0, self.min_length, nbins,
                self.use_projected_track, self.use_confined_track
                )
        self.pdf_axis = np.array(pdf_axis)
        self.pdf = np.array(pdf) * 0.1
        self.MTL = np.sum(self.pdf_axis * self.pdf) * 200.0 / self.pdf.shape[0] 

        return self.pdf_axis, self.pdf, self.MTL

    def calculate_age(self, grain, nbins=200):
        time = (self.history.time * u.megayear).to(u.seconds).magnitude
        temperature = self.history.temperature

        self._get_reduced_length(grain, nbins)
        self._get_distribution(grain, nbins)
        oldest_age, ft_model_age, reduced_density = calculate_model_age(
        time, temperature, self.reduced_lengths, 
        self.first_node, self.length_reduction
        )

        self.oldest_age = oldest_age
        self.ft_model_age = ft_model_age
        self.reduced_density = reduced_density

        return self.oldest_age, self.ft_model_age, self.reduced_density

    solve = calculate_age

    def generate_synthetic_counts(self, Nc=30):
        """Generate Synthetic AFT data.

        Parameters:
        Nc : Number of crystals

        """
        rho = self.reduced_density

        # Probability in binomial distribution
        prob = rho / (1. + rho)

        # For Nc crystals, generate synthetic Ns and Ni
        # count data using binomial distribution, conditional
        # on total counts Ns + Ni, sampled randomly with
        # a maximum of 100.

        NsNi = np.random.randint(5, 100, Nc)
        Ns = np.array([drawbinom(I, prob) for I in NsNi])
        Ni = NsNi - Ns
        return Ns, Ni

    def generate_synthetic_lengths(self, ntl):
        tls = draw_from_distrib(self.pdf_axis, self.pdf, ntl)
        return tls


class Ketcham1999(ForwardModel):

    def __init__(self, history, use_projected_track=False,
                 use_confined_track=False, min_length=2.15,
                 length_reduction=0.893, etchant="5.5"):

        super(Ketcham1999, self).__init__(
                history, use_projected_track,
                use_confined_track, min_length,
                length_reduction
                )
        self.etchant = etchants[etchant]

    def _get_reduced_length(self, grain, nbins=200):
        time = (self.history.time * u.megayear).to(u.seconds).magnitude
        temperature = self.history.temperature
        kinetic_parameter_type = kinpar[grain.kinetic_parameter_type]
        kinetic_parameter_value = grain.kinetic_parameter_value
        
        reduced_lengths, first_node = ketcham99_annealing_model(
                time, temperature, kinetic_parameter_type,
                kinetic_parameter_value, nbins
                )
        self.reduced_lengths = reduced_lengths
        self.first_node = first_node
        return reduced_lengths, first_node


class Ketcham2007(ForwardModel):

    def __init__(self, history, use_projected_track=False,
                 use_confined_track=False, min_length=2.15,
                 length_reduction=0.893, etchant="5.5"):

        super(Ketcham2007, self).__init__(
                history, use_projected_track,
                use_confined_track, min_length,
                length_reduction
                )
        self.etchant = etchants[etchant]

    def _get_reduced_length(self, grain, nbins=200):
        time = (self.history.time * u.megayear).to(u.seconds).magnitude
        temperature = self.history.temperature
        kinetic_parameter_type = kinpar[grain.kinetic_parameter_type]
        kinetic_parameter_value = grain.kinetic_parameter_value
        
        reduced_lengths, first_node = ketcham07_annealing_model(
                time, temperature, kinetic_parameter_type,
                kinetic_parameter_value, nbins, self.etchant
                )
        self.reduced_lengths = reduced_lengths
        self.first_node = first_node
        return reduced_lengths, first_node


class Grain(object):

    unprojected = {"ETCH_PIT_LENGTH": {"m": 0.283, "b": 15.63},
                   "CL_PFU": {"m": 0.544, "b": 16.18},
                   "OH_PFU": {"m": 0.0, "b": 16.18},
                   "CL_WT_PCT": {"m": 0.13824, "b": 16.288}}

    projected = {"ETCH_PIT_LENGTH": {"m": 0.205, "b": 16.10},
                 "CL_PFU": {"m": 0.407, "b": 16.49},
                 "OH_PFU": {"m": 0.000, "b": 16.57},
                 "CL_WT_PCT": {"m": 0.17317, "b": 16.495}}

    def __init__(self, Ns=None, Ni=None, track_lengths=None, Dpars=None,
                 Cl=None, name=None):
        """
          Grain

          Ns: number of spontaneous tracks
          Ni: number of induced tracks
          track_lengths: track length measurements
          Dpars: Dpar values
          Cl: Chlorine content
          name: optional name

        """

        self.name = name
        self.spontaneous = self.Ns = Ns
        self.induced = self.Ni = Ni
        self.track_lengths = track_lengths

        if Dpars:
            self.kinetic_parameter_type = "ETCH_PIT_LENGTH"
            self.kinetic_parameter_value = np.mean(Dpars)
        elif Cl:
            self.kinetic_parameter_type = "CL_PFU"
            self.kinetic_parameter_value = np.mean(Cl)

        self._get_initial_track_length()

        if self.track_lengths:
            self.min_track_lengths = self.MTL = np.mean(self.track_lengths)

    def _get_initial_track_length(self):
        """

            Returns the initial track length for the population based
            on the apatite kinetics, using data from experiment H0
            by W.D.Carlson and R.A.Donelick (UT Austin)

            kinetic_parameter_type: ETCH_PIT_LENGTH, CL_PFU, OH_PFU, CL_WT_PFU
            kinetic_parameter_value: value of the kinetic parameter
            use_projected_track: use projected track? default is False

        """

        m = Grain.projected[self.kinetic_parameter_type]["m"]
        b = Grain.projected[self.kinetic_parameter_type]["b"]
        self.l0_projected = m * self.kinetic_parameter_value + b

        m = Grain.unprojected[self.kinetic_parameter_type]["m"]
        b = Grain.unprojected[self.kinetic_parameter_type]["b"]
        self.l0 = m * self.kinetic_parameter_value + b

        return


class Sample(object):

    def __init__(self, grains, coordinates=None,
                 elevation=None, name=None):

        """
          Sample

          grains: list of grains
          coordinates: coordinates of the sample location
          elevation: elevation of the sample location
          name: sample name
        """

        self.name = name

        grains = list(grains)
        for grain in grains:
            if not isinstance(grain, Grain):
                raise ValueError("grains must be a list of Grain instance")
        self.grains = grains
        self.ngrains = len(grains)


Apatite = Grain


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
