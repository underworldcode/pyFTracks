import numpy as np
from .utilities import draw_from_distrib, drawbinom
from .viewer import Viewer
import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp

# cdef extern from "include/utilities.h":

#     cdef void ketcham_sum_population(
#         int numPDFPts, int numTTNodes, int firstTTNode, int doProject,
#         int usedCf, double *time, double *temperature, double *pdfAxis,
#         double *pdf, double *cdf, double  initLength, double min_length,
#         double  *redLength)
#     cdef void ketcham_calculate_model_age(
#         double *time, double *temperature, double  *redLength,
#         int numTTNodes, int firstNode, double  *oldestModelAge,
#         double *ftModelAge, double stdLengthReduction, double *redDensity)

cdef extern from "include/ketcham1999.h":
    
    cdef void ketch99_reduced_lengths(
        double *time, double *temperature,int numTTNodes, double *redLength,
        double rmr0, int *firstTTNode)

cdef extern from "include/ketcham2007.h":
    
    cdef void ketch07_reduced_lengths(
        double *time, double *temperature,int numTTNodes, double *redLength,
        double rmr0, int *firstTTNode)

cdef correct_observational_bias(double cparlen):
    """
    Does the conversion from length to density for the Ketcham et al., 1999 model.
    
    The routine is also used to estimate bias for population summing.
    
    Assumes we're passing in a c-axis-projected length
    
    The following text is taken from Forward Inverse and Modeling of LT Thermoch Data
    Low-T Thermochronology: Techniques, Interpretations and Applications (ed. Reiners an Ehlers)
    
    The observational bias quantifies the relative probability of observation among different
    fission-track populations calculated by the model. Highly annealed populations are less
    likely to be detected and measured than less-annealed populations for 2 primary reasons.
      - Shorter track are less frequently impinged and thus etched
      - At advanced stage of annealing some proportion of tracks at high angles to the c-axis
        may be lost altogether, even though lower-angle tracks remain long
    Thus the number of detectable tracks in the more annealed population diminishes, at a rate
    dispropportionate to measured mean length (Ketcham 2003b). These 2 factors can be approximated
    in a general way by using an empirical function that relates measured fission-track length to
    fission-track density (e,g. Green 1998). The following is taken from Ketcham et al 2000 
    """
    if (cparlen > 0.765):
        return 1.600 * cparlen - 0.599
    if (cparlen >= 0.13):
        return 9.205 * cparlen * cparlen - 9.157 * cparlen + 2.269
    return 0.0


cdef calculate_reduced_stddev(double redLength, int doProject):
    """Calculates the reduced standard deviation of a track population length
       from the reduced mean length.  Based on Carlson and Donelick"""
    if doProject:
        return(0.1081 - 0.1642 * redLength + 0.1052 * redLength * redLength)
    else:
        return(0.4572 - 0.8815 * redLength + 0.4947 * redLength * redLength)

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

    # def _get_distribution(self, track_l0, nbins=200):
    
    #     cdef double init_length = track_l0
        
    #     time = self.history.time * _seconds_in_megayears 
    #     temperature = self.history.temperature
    #     time = np.ascontiguousarray(time)
    #     temperature = np.ascontiguousarray(temperature)
    #     reduced_lengths = np.ascontiguousarray(self.reduced_lengths)
    
    #     cdef double[::1] time_memview = time
    #     cdef double[::1] temperature_memview = temperature
    #     cdef double[::1] reduced_lengths_memview = reduced_lengths
    #     cdef double[::1] pdfAxis = np.zeros((nbins))
    #     cdef double[::1] cdf = np.zeros((nbins))
    #     cdef double[::1] pdf = np.zeros((nbins))
    #     cdef int first_node = self.first_node
    #     cdef double min_length = self.min_length
    #     cdef int project = self.use_projected_track
    #     cdef int usedCf = self.use_Cf_irradiation
    
    #     ketcham_sum_population(nbins, time_memview.shape[0], first_node,
    #                            <int> project, <int> usedCf, &time_memview[0],
    #                            &temperature_memview[0], &pdfAxis[0], &pdf[0],
    #                            &cdf[0], init_length, min_length,
    #                            &reduced_lengths_memview[0])
        
    #     self.pdf_axis = np.array(pdfAxis)
    #     self.pdf = np.array(pdf) * 0.1
    #     self.MTL = np.sum(self.pdf_axis * self.pdf) * 200.0 / self.pdf.shape[0] 
    
    #     return self.pdf_axis, self.pdf, self.MTL


    def _get_distribution(self, track_l0, nbins=200):
    
        cdef double init_length = track_l0
         
        cdef double[::1] time = np.ascontiguousarray(self.history.time * _seconds_in_megayears)
        cdef double[::1] temperature = np.ascontiguousarray(self.history.temperature)
        cdef double[::1] reduced_lengths = np.ascontiguousarray(self.reduced_lengths)
        cdef double[::1] pdfAxis = np.zeros((nbins))
        cdef double[::1] cdf = np.zeros((nbins))
        cdef double[::1] pdf = np.zeros((nbins))
        cdef int first_node = self.first_node
        cdef double min_length = self.min_length
        cdef int project = self.use_projected_track
        cdef int usedCf = self.use_Cf_irradiation
        cdef int num_points_pdf = nbins
        cdef int numTTNodes = time.shape[0]

        cdef int i, j
        cdef double weight, rLen, rStDev, obsBias, rmLen, calc, z
        cdef double wt1, wt2

        cdef double U238SEC = 4.91575e-18
        cdef double SQRT2PI = 2.50662827463

        for i in range(num_points_pdf):
            pdf[i] = 0.

        for i in range(num_points_pdf):
            pdfAxis[i] = <double>(i * 1.0 + 0.5) * 20.0 / num_points_pdf

        wt1 = exp(U238SEC * time[first_node]) / U238SEC

        for j in range(numTTNodes - 1):

            wt2 = exp(U238SEC * time[j+1]) / U238SEC
            weight = wt1 - wt2
            wt1 = wt2
            if usedCf:
                rmLen = 1.396 * reduced_lengths[j] - 0.4017
            else:
                rmLen = -1.499 * reduced_lengths[j] * reduced_lengths[j] + 4.150 * reduced_lengths[j] - 1.656
            if project:
                rLen = reduced_lengths[j]
            else:
                rLen = rmLen
            rStDev = calculate_reduced_stddev(rLen, project)
            obsBias = correct_observational_bias(reduced_lengths[j])
            calc = weight * obsBias / (rStDev * SQRT2PI)

            if rLen > 0:
                for i in range(num_points_pdf):
                    if pdfAxis[i] >= min_length:
                        z = (rLen - pdfAxis[i] / init_length) / rStDev
                        if z <= 4:
                            pdf[i] += calc * exp(-(z*z) / 2.0)

        cdf[0] = pdf[0]
        for i in range(num_points_pdf):
            cdf[i] = cdf[i-1] = ((pdf[i] + pdf[i-1]) / 2.0) * (pdfAxis[i] - pdfAxis[i-1])
       
        if cdf[num_points_pdf - 1] > 0.:
            for i in range(num_points_pdf):
                pdf[i] = pdf[i] / cdf[num_points_pdf - 1]
                cdf[i] = cdf[i] / cdf[num_points_pdf - 1]

        self.pdf_axis = np.array(pdfAxis)
        self.pdf = np.array(pdf) * 0.1
        self.MTL = np.sum(self.pdf_axis * self.pdf) * 200.0 / self.pdf.shape[0] 
    
        return self.pdf_axis, self.pdf, self.MTL

    def calculate_age(self, track_l0, nbins=200):

        self.annealing_model(nbins)
        self._get_distribution(track_l0, nbins)

        cdef double[::1] time = np.ascontiguousarray(self.history.time * _seconds_in_megayears )
        cdef double[::1] temperature = np.ascontiguousarray(self.history.temperature)
        cdef double[::1] reduced_lengths = np.ascontiguousarray(self.reduced_lengths)
        cdef int first_node = self.first_node
        cdef double std_length_reduction = self.length_reduction

        cdef double oldest_age
        cdef double ft_model_age
        cdef double reduced_density

        cdef int node
        cdef double midLength
        cdef long secinmyr = _seconds_in_megayears

        cdef int numTTNodes = time.shape[0]

        reduced_density = 0.0 
        ft_model_age = 0.0
        oldest_age = time[first_node] / secinmyr

        for node in range(numTTNodes - 2):
            midLength = (reduced_lengths[node] + reduced_lengths[node+1]) / 2.0
            ft_model_age += correct_observational_bias(midLength) * (time[node] - time[node+1])
            reduced_density += correct_observational_bias(midLength) 

        ft_model_age += correct_observational_bias(reduced_lengths[numTTNodes - 2]) * (time[node] - time[node+1])
        reduced_density += correct_observational_bias(reduced_lengths[numTTNodes - 2])
        reduced_density /= std_length_reduction * (numTTNodes-2)

        ft_model_age /= std_length_reduction * secinmyr

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
        
    def annealing_model(self, nbins=200):

        cdef double[::1] time_memview = np.ascontiguousarray(self.history.time * _seconds_in_megayears)
        cdef double[::1] temperature_memview = np.ascontiguousarray(self.history.temperature)
        cdef double[::1] reduced_lengths = np.zeros((nbins))
        cdef double crmr0 = self.rmr0 
        cdef int first_node = 0

        ketch99_reduced_lengths(&time_memview[0], &temperature_memview[0],
                                time_memview.shape[0], &reduced_lengths[0],
                                crmr0, &first_node)

        self.reduced_lengths = np.array(reduced_lengths)
        self.first_node = first_node
        return self.reduced_lengths, self.first_node

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

    def annealing_model(self, nbins=200):

        cdef double[::1] time_memview = np.ascontiguousarray(self.history.time * _seconds_in_megayears)
        cdef double[::1] temperature_memview = np.ascontiguousarray(self.history.temperature)
        cdef double[::1] reduced_lengths = np.zeros((nbins))
        cdef double crmr0 = self.rmr0 
        cdef int first_node = 0

        ketch07_reduced_lengths(&time_memview[0], &temperature_memview[0],
                                time_memview.shape[0], &reduced_lengths[0],
                                crmr0, &first_node)

        self.reduced_lengths = np.array(reduced_lengths)
        self.first_node = first_node
        return self.reduced_lengths, self.first_node

    @property
    def rmr0(self):
        return Ketcham2007._kinetic_conversion[self.kinetic_parameter_type].__func__(self.kinetic_parameter)
