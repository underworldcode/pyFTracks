import numpy as np
import pint
from .utilities import draw_from_distrib, drawbinom
from ketcham import ketcham99_annealing_model
from ketcham import ketcham07_annealing_model
from ketcham import sum_population
from ketcham import calculate_model_age
from .viewer import Viewer
from .grain import Grain
from scipy.stats import chi2

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

    def _get_reduced_length(self, sample, nbins=200):
        return

    def _get_distribution(self, sample, nbins=200):
        time = (self.history.time * u.megayear).to(u.seconds).magnitude
        temperature = self.history.temperature
        
        if self.use_projected_track:
            track_l0 = sample.l0_projected
        else:
            track_l0 = sample.l0

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

    def calculate_age(self, sample, nbins=200):
        time = (self.history.time * u.megayear).to(u.seconds).magnitude
        temperature = self.history.temperature

        self._get_reduced_length(sample, nbins)
        self._get_distribution(sample, nbins)
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

    def generate_synthetic_lengths(self, ntl=100):
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

    def _get_reduced_length(self, sample, nbins=200):
        time = (self.history.time * u.megayear).to(u.seconds).magnitude
        temperature = self.history.temperature
        kinetic_parameter_type = kinpar[sample.kinetic_parameter_type]
        kinetic_parameter_value = sample.kinetic_parameter_value
        
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

    def _get_reduced_length(self, sample, nbins=200):
        time = (self.history.time * u.megayear).to(u.seconds).magnitude
        temperature = self.history.temperature
        kinetic_parameter_type = kinpar[sample.kinetic_parameter_type]
        kinetic_parameter_value = sample.kinetic_parameter_value
        
        reduced_lengths, first_node = ketcham07_annealing_model(
                time, temperature, kinetic_parameter_type,
                kinetic_parameter_value, nbins, self.etchant
                )
        self.reduced_lengths = reduced_lengths
        self.first_node = first_node
        return reduced_lengths, first_node


