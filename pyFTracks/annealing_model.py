import numpy as np
from .utilities import draw_from_distrib, drawbinom
from .ketcham import ketcham99_annealing_model
from .ketcham import ketcham07_annealing_model
from .ketcham import sum_population
from .ketcham import calculate_model_age
from .viewer import Viewer

_seconds_in_megayears = 31556925974700

class AnnealingModel():

    def __init__(self, use_projected_track=False,
                 use_Cf_irradiation=False, min_length=2.15,
                 length_reduction=0.893):

        self.use_projected_track = use_projected_track
        self.use_Cf_irradiation = use_Cf_irradiation
        self.min_length = min_length
        self.length_reduction = length_reduction
        self._kinetic_parameter = None
        self._kinetic_parameter_type = None
        self.annealing_model = None

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    @property
    def kinetic_parameter(self):
        return self._kinetic_parameter

    @kinetic_parameter.setter
    def kinetic_parameter(self, value):
        self._kinetic_parameter = value
    
    @property
    def kinetic_parameter_type(self):
        return self._kinetic_parameter_type

    @kinetic_parameter_type.setter
    def kinetic_parameter_type(self, value):
        self._kinetic_parameter_type = value

    def _get_reduced_length(self, nbins=200):
        # Convert from megayears to seconds
        time = self.history.time * _seconds_in_megayears 
        temperature = self.history.temperature

        self.reduced_lengths, self.first_node = self.annealing_model(
                time, temperature, self.rmr0, nbins)
        return self.reduced_lengths, self.first_node

    def _get_distribution(self, track_l0, nbins=200):
        # Convert from megayears to seconds
        time = self.history.time * _seconds_in_megayears 
        temperature = self.history.temperature
        
        pdf_axis, pdf, cdf = sum_population(
                time, temperature,
                self.reduced_lengths, self.first_node,
                track_l0, self.min_length, nbins,
                self.use_projected_track, self.use_Cf_irradiation
                )
        self.pdf_axis = np.array(pdf_axis)
        self.pdf = np.array(pdf) * 0.1
        self.MTL = np.sum(self.pdf_axis * self.pdf) * 200.0 / self.pdf.shape[0] 

        return self.pdf_axis, self.pdf, self.MTL

    def calculate_age(self, track_l0, nbins=200):
        # Convert from megayears to seconds
        time = self.history.time * _seconds_in_megayears 
        temperature = self.history.temperature

        self._get_reduced_length(nbins)
        self._get_distribution(track_l0, nbins)
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


class Ketcham1999(AnnealingModel):
    
    @staticmethod
    def convert_Dpar_to_rmr0(dpar):
        if dpar <= 1.75: 
            return 0.84
        elif dpar >= 4.58: 
            return 0.
        else: 
            return 1.0 - np.exp(0.647 * (dpar - 1.75) - 1.834)

    @staticmethod
    def convert_Cl_pfu_to_rmr0(clpfu):
        value = np.abs(clpfu - 1.0)
        if value <= 0.130:
            return 0.0
        else:
            return 1.0 - np.exp(2.107 * (1.0 - value) - 1.834)

    @staticmethod
    def convert_Cl_weight_pct(clwpct):
        clwpct *= 0.2978
        return Ketcham1999.convert_Cl_pfu_to_rmr0(clwpct)

    @staticmethod
    def convert_OH_pfu_to_rmr0(ohpfu):
        value = np.abs(ohpfu - 1.0)
        return 0.84 * (1.0 - (1.0 - value)**4.5)

    _kinetic_conversion = {"ETCH_PIT_LENGTH": convert_Dpar_to_rmr0,
                          "CL_PFU": convert_Cl_pfu_to_rmr0,
                          "OH_PFU": convert_OH_pfu_to_rmr0,
                          "RMR0": lambda x: x}

    def __init__(self, use_projected_track=False,
                 use_Cf_irradiation=False, min_length=2.15,
                 length_reduction=0.893):

        super(Ketcham1999, self).__init__(
                use_projected_track,
                use_Cf_irradiation, min_length,
                length_reduction
                )
        
        self.annealing_model = ketcham99_annealing_model

    @property
    def rmr0(self):
        return Ketcham1999._kinetic_conversion[self.kinetic_parameter_type].__func__(self.kinetic_parameter)



class Ketcham2007(AnnealingModel):
    
    @staticmethod
    def convert_Dpar_to_rmr0(dpar, etchant="5.5HNO3"):
        """ Here depends on the etchant (5.5 or 5.0 HNO3)
            This is based on the relation between the fitted rmr0 values and
            the Dpar etched using a 5.5M etchant as published in
            Ketcham et al, 2007,Figure 6b
            We use the linear conversion defined in Ketcham et al 2007 to
            make sure that we are using 5.5M DPar"""
        if etchant == "5.0HNO3": 
             dpar = 0.9231 * dpar + 0.2515
        if dpar <= 1.75:
            return 0.84
        elif dpar >= 4.58:
            return 0
        else:
            return 0.84 * ((4.58 - dpar) / 2.98)**0.21
        
    @staticmethod
    def convert_Cl_pfu_to_rmr0(clpfu):
        """ Relation between fitted rmr0 value from the fanning curvilinear model and
            Cl content is taken from Ketcham et al 2007 Figure 6a """
        value = np.abs(clpfu - 1.0)
        if value <= 0.130:
            return 0.0
        else:
            return 0.83 * ((value - 0.13) / 0.87)**0.23

    @staticmethod
    def convert_Cl_weight_pct(clwpct):
        # Convert %wt to APFU
        return Ketcham2007.convert_Cl_pfu_to_rmr0(clwpct * 0.2978)


    @staticmethod
    def convert_unit_paramA_to_rmr0(paramA):
        if paramA >= 9.51:
            return 0.0
        else:
            return 0.84 * ((9.509 - paramA) / 0.162)**0.175
    
    _kinetic_conversion = {"ETCH_PIT_LENGTH": convert_Dpar_to_rmr0,
                          "CL_PFU": convert_Cl_pfu_to_rmr0,
                          "RMR0": lambda x: x}

    def __init__(self, use_projected_track=False,
                 use_Cf_irradiation=False, min_length=2.15,
                 length_reduction=0.893):

        super(Ketcham2007, self).__init__(
                use_projected_track,
                use_Cf_irradiation, min_length,
                length_reduction
                )
        self.annealing_model = ketcham07_annealing_model

    @property
    def rmr0(self):
        return Ketcham2007._kinetic_conversion[self.kinetic_parameter_type].__func__(self.kinetic_parameter)
