import numpy as np
import pint
from .annealing import KetchamEtAl
from .utilities import draw_from_distrib, drawbinom
from ketcham import ForwardModel

u = pint.UnitRegistry()


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

    def solve(self, nbins=200, use_confined=False, min_length=2.15,
              length_reduction=0.893):
        time = (self.history.time * u.megayear).to(u.seconds).magnitude
        temperature = self.history.temperature
        kinpar = {"ETCH_PIT_LENGTH": 0, "CL_PFU": 1,
                  "OH_PFU": 2, "CL_WT_PCT": 3}

        track_l0 = self.get_initial_track_length()

        kinetic_parameter_type = kinpar[self.kinetic_parameter_type]

        pdf_axis, pdf, cdf, oldest_age, ft_model_age, reduced_density = (
            self.calculate_density_distribution(
                time, temperature, kinetic_parameter_type,
                track_l0, min_length, nbins, length_reduction,
                self.projected_track, use_confined
            )
        )

        self.oldest_age = oldest_age
        self.ft_model_age = ft_model_age
        self.reduced_density = reduced_density
        self.pdf_axis = np.array(pdf_axis)
        self.pdf = np.array(pdf) * 0.1

        return

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
        mtl = (float(sum(tls))/len(tls) if len(tls) > 0 else float('nan'))
        mtl_sd = np.std(tls)
        return tls, mtl, mtl_sd


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


class Grain(object):

    def __init__(self, spontaneous_tracks, induced_tracks,
                 track_lengths=None, Dpars=None,
                 Cl=None, name=None):
        """
          Grain

          spontaneous_tracks: number of spontaneous tracks
          induced_tracks: number of induced tracks
          track_lengths: track length measurements
          Dpars: Dpar values
          Cl: Chlorine content
          name: optional name

        """

        self.name = name
        self.spontaneous = self.Ns = spontaneous_tracks
        self.induced = self.Ni = induced_tracks
        self.track_lengths = track_lengths
        self.Dpars = Dpars

        self.min_Dpars = np.mean(Dpars)
        self.min_track_lengths = self.MTL = np.mean(self.track_lengths)


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
