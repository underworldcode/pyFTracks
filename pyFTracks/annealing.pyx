import numpy as np
from .utilities import draw_from_distrib, drawbinom
from .viewer import Viewer
import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, log
from pyFTracks.structures import Sample

_MIN_OBS_RCMOD = 0.55


cdef struct annealModel:
    double c0, c1, c2, c3, a, b

cdef correct_observational_bias(double rcmod):
    """
    Does the conversion from length to density for the Ketcham et al., 1999 model.
    
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
    if (rcmod >= 0.765):
        return 1.600 * rcmod - 0.600
    # because very short fission tracks are undetectable, they should be eliminated from model results.
    # We assumes a minimum detectable length of 2.18 µm, or a reduced length of 0.13, 
    # the shortest track observed in over 38,000 measurements in the Carlson et al. (1999) data set.
    # (Ketcham, 2000)
    if (rcmod >= _MIN_OBS_RCMOD):
        return 9.205 * rcmod * rcmod - 9.157 * rcmod + 2.269
    return 0.0


cdef calculate_reduced_stddev(double redLength, int doProject):
    """Calculates the reduced standard deviation of a track population length
       from the reduced mean length.  Based on Carlson and Donelick"""
    if doProject:
        return(0.1081 - 0.1642 * redLength + 0.1052 * redLength * redLength)
    else:
        return(0.4572 - 0.8815 * redLength + 0.4947 * redLength * redLength)


cdef calculate_mean_reduced_length_ketcham1999(double redLength, int usedCf):
    # Californium irradiation of apatite can be a useful technique for increasing the number
    # of confined tracks. It will however change the biasing of track detection.
    # If it is necessary to calculate the mean rather than c-axis-projected lengths, we
    # use the empirical function provided by Ketcham et al 1999.
    if usedCf:
        return 1.396 * redLength - 0.4017
    else:
        return -1.499 * redLength * redLength + 4.150 * redLength - 1.656


cdef calculate_mean_reduced_length_ketcham2003(double redLength, int usedCf):
    # Californium irradiation of apatite can be a useful technique for increasing the number
    # of confined tracks. It will however change the biasing of track detection.
    # If it is necessary to calculate the mean rather than c-axis-projected lengths, we
    # use the empirical function provided by Ketcham et al 2003.
    if usedCf:
        return -0.4720 + 1.4701 * redLength  
    else:
        return -1.2101 + 3.0864 * redLength - 0.8792 * redLength * redLength


_seconds_in_megayears = 31556925974700

class AnnealingModel():

    def __init__(self, use_projected_track: bool=False,
                 use_Cf_irradiation: bool =False):

        self.use_projected_track = use_projected_track
        self.use_Cf_irradiation = use_Cf_irradiation

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    def _sum_populations(self, track_l0=16.1, nbins=200):
        
        cdef double init_length = track_l0
        cdef double[::1] time = np.ascontiguousarray(self.history.time)
        cdef double[::1] reduced_lengths = np.ascontiguousarray(self.reduced_lengths)
        cdef int first_node = self.first_node

        cdef double[::1] pdfAxis = np.zeros((nbins))
        cdef double[::1] cdf = np.zeros((nbins))
        cdef double[::1] pdf = np.zeros((nbins))
        cdef double min_length = 2.15
        cdef int project = self.use_projected_track
        cdef int usedCf = self.use_Cf_irradiation
        cdef int num_points_pdf = nbins
        cdef int numTTNodes = time.shape[0]

        cdef int i, j
        cdef double weight, rStDev, obsBias, calc, rmLen, z
        cdef double wt1, wt2

        cdef double SQRT2PI = 2.50662827463
        cdef double U238MYR = 1.55125e-4

        for i in range(num_points_pdf):
            pdf[i] = 0.

        for i in range(num_points_pdf):
            pdfAxis[i] = <double>(i * 1.0 + 0.5) * 20.0 / num_points_pdf

        wt1 = exp(U238MYR * time[first_node]) / U238MYR

        for j in range(first_node, numTTNodes - 1):

            wt2 = exp(U238MYR * time[j+1]) / U238MYR
            weight = wt1 - wt2
            wt1 = wt2

            # Californium irradiation of apatite can be a useful technique for increasing the number
            # of confined tracks. It will however change the biasing of track detection.
            # If it is necessary to calculate the mean rather than c-axis-projected lengths, we
            # use the empirical function provided by Ketcham et al 1999.
            rmLen = calculate_mean_reduced_length_ketcham1999(reduced_lengths[j], usedCf)
            
            rStDev = calculate_reduced_stddev(rmLen, project)
            obsBias = correct_observational_bias(rmLen)
            calc = weight * obsBias / (rStDev * SQRT2PI)

            if rmLen > 0:
                for i in range(num_points_pdf):
                    if pdfAxis[i] >= min_length:
                        z = (rmLen - pdfAxis[i] / init_length) / rStDev
                        if z <= 4.:
                            pdf[i] += calc * exp(-(z*z) / 2.0)

        self.pdf_axis = np.array(pdfAxis)
        self.pdf = np.array(pdf)
        self.pdf /= self.pdf.sum()
        self.cdf = self.pdf.cumsum()
        self.MTL = np.sum(self.pdf_axis * self.pdf)
        self.STD = np.sqrt(np.sum(self.pdf_axis**2 * self.pdf) - self.MTL**2)

        return self.pdf_axis, self.pdf, self.MTL

    def calculate_age(self, track_l0=16.1, std_length_reduction=0.893):
        """ Predict the pooled fission-track age 
        
        We assume that each time step of length dt will contribute dt to the
        total fission track age, modified by the amount of track density reduction of
        the population in that time step, relative to the age standard.

        The total age is the sum of all contributions

        std_length_reduction: Estimated fission track density reduction in the age standard.
        The density reduction in the age standard is calculated using its estimated
        track length reduction, using the assumption that density reduction is proportional to
        length reduction, and that spontaneaous fission track are initially as long as induced
        track.
        
        If for a fission-track worker the Durango apatite has a measured present day spontaneous
        mean track length of 14.47 um, and a mean induced track length of 16.21, them the
        estimated length reduction is 14.47/16.21 = 0.893
        """

        self.annealing_model()
        self._sum_populations(track_l0)

        cdef double[::1] time = np.ascontiguousarray(self.history.time * _seconds_in_megayears )
        cdef double[::1] reduced_lengths = np.ascontiguousarray(self.reduced_lengths)
        cdef int first_node = self.first_node

        cdef double cstd_length_reduction = std_length_reduction

        cdef double oldest_age
        cdef double ft_model_age
        cdef double reduced_density

        cdef int node
        cdef double midLength
        cdef long long secinmyr = _seconds_in_megayears

        cdef int numTTNodes = time.shape[0]

        reduced_density = 0.0 
        ft_model_age = 0.0
        oldest_age = time[first_node] / secinmyr

        for node in range(numTTNodes - 2):
            # Take midpoint length as the mean of the endpoints. This is conform to
            # Willett (1992) and is also described in the Ketcham 2000 AFTSolve implementation.
            midLength = (reduced_lengths[node] + reduced_lengths[node+1]) / 2.0
            ft_model_age += correct_observational_bias(midLength) * (time[node] - time[node+1])
            reduced_density += correct_observational_bias(midLength) 

        ft_model_age += correct_observational_bias(reduced_lengths[numTTNodes - 2]) * (time[node] - time[node+1])
        reduced_density += correct_observational_bias(reduced_lengths[numTTNodes - 2])
        reduced_density /= cstd_length_reduction * (numTTNodes-2)

        ft_model_age /= cstd_length_reduction * secinmyr

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

    def generate_synthetic_sample(self, counts=30,  ntl=100):
        tls = self.generate_synthetic_lengths(ntl)
        Ns, Ni = self.generate_synthetic_counts(counts)
        A = np.random.randint(10, 100, Ns.size)
        data = {"Ns": Ns, "Ni": Ni, "A": A}
        sample = Sample(data)
        sample.track_lengths = tls
        sample.pooled_age = self.ft_model_age
        return sample



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

    def __init__(self, kinetic_parameters: dict, use_projected_track: bool =False,
                 use_Cf_irradiation: bool =False):

        self._kinetic_parameters = kinetic_parameters
       
        super(Ketcham1999, self).__init__(
                use_projected_track,
                use_Cf_irradiation)
    
    @property
    def kinetic_parameters(self):
        return self._kinetic_parameters

    @kinetic_parameters.setter
    def kinetic_parameters(self, value):
        self._kinetic_parameters = value
    
    @property
    def rmr0(self):
        kinetic_type = list(self.kinetic_parameters.keys())[0]
        kinetic_value = self.kinetic_parameters[kinetic_type]
        return self._kinetic_conversion[kinetic_type].__func__(kinetic_value)
        
    def annealing_model(self):

        # Must be in seconds (do conversion)
        cdef double[::1] time = np.ascontiguousarray(self.history.time * _seconds_in_megayears)
        # Must be in Kelvin
        cdef double[::1] temperature = np.ascontiguousarray(self.history.temperature)
        cdef int numTTnodes = time.shape[0]
        cdef double[::1] reduced_lengths = np.zeros(time.shape[0] - 1)
        cdef double crmr0 = self.rmr0
        cdef int first_node = 0

        cdef int node, nodeB
        cdef double equivTime
        cdef double timeInt, x1, x2, x3
        cdef double totAnnealLen
        cdef double equivTotAnnLen
        cdef double k
        cdef double calc
        cdef double tempCalc
        cdef double MIN_OBS_RCMOD = _MIN_OBS_RCMOD

        # Fanning Curvilinear Model lcMod FC, See Ketcham 1999, Table 5e
        # The preferred equation presented in Ketcham et al 1999, describes the apatite
        # B2 from the Carlson et al 1999 data set. The Apatite, which is a chlor-hydroxy apatite from
        # Norway, showed the most resistance to annealing """ 
        cdef annealModel modKetch99 = annealModel(
            c0=-19.844,
            c1=0.38951,
            c2=-51.253,
            c3=-7.6423,
            a=-0.12327,
            b=-11.988)

        k = 1 - crmr0

        totAnnealLen = MIN_OBS_RCMOD
        equivTotAnnLen =  pow(totAnnealLen, 1.0 / k) * (1.0 - crmr0) + crmr0

        equivTime = 0.
        tempCalc = log(1.0 / ((temperature[numTTnodes - 2] +  temperature[numTTnodes - 1]) / 2.0))

        for node in range(numTTnodes - 2, -1, -1):
            # We calculate the modeled reduced length (length normalized by
            # initial length of a fission track parallel to the c-axis (Donelick 1999))
            # after an isothermal annealing episode at a temperature T (Kelvin) of
            # duration t (seconds)
            timeInt = time[node] - time[node + 1] + equivTime
            x1 = (log(timeInt) - modKetch99.c2) / (tempCalc - modKetch99.c3)
            x2 = 1.0 + modKetch99.a * (modKetch99.c0 + modKetch99.c1 * x1)

            if x2 < 0:
                reduced_lengths[node] = 0.0
            else:
                reduced_lengths[node] = pow(x2, 1.0 / modKetch99.a)
                if x3 < 0:
                    reduced_lengths[node] = 0.
                else:
                    x3 = 1.0 - modKetch99.b * reduced_lengths[node]
                    reduced_lengths[node] = pow(x3, 1.0 / modKetch99.b)

            if reduced_lengths[node] < equivTotAnnLen:
                reduced_lengths[node] = 0.

            # Check to see if we've reached the end of the length distribution
            # If so, we then do the kinetic conversion.
            if reduced_lengths[node] == 0.0 or node == 0:
                if node > 0:
                    node += 1
                first_node = node

                for nodeB in range(first_node, numTTnodes - 1):
                    if reduced_lengths[nodeB] < crmr0:
                        reduced_lengths[nodeB] = 0.0
                        first_node = nodeB
                    else:
                        # This is equation 8 from Ketcham et al, 1999
                        # Apatite with the composition of B2 are very rare, B2 is
                        # significantly more resistant than the most common variety, near
                        # end member fluorapatite.
                        # Ketcham 1999 showed that the reduced length of any apatite could
                        # be related to the length of an apatite that is relatively more resistant
                        # (hence use of B2)
                        reduced_lengths[nodeB] = pow((reduced_lengths[nodeB] - crmr0) / (1.0 - crmr0), k)
                        if reduced_lengths[nodeB] < totAnnealLen:
                            reduced_lengths[nodeB] = 0.
                            first_node = nodeB
        
                self.reduced_lengths = np.array(reduced_lengths)
                self.first_node = first_node
                return self.reduced_lengths, self.first_node

            # Update tiq for this time step
            if reduced_lengths[node] < 0.999:
                tempCalc = log(1.0 / ((temperature[node-1] + temperature[node]) / 2.0))
                equivTime = pow((1.0 - pow(reduced_lengths[node], modKetch99.b)) / modKetch99.b, modKetch99.a)
                equivTime = ((equivTime - 1.0) / modKetch99.a - modKetch99.c0) / modKetch99.c1
                equivTime = exp(equivTime * (tempCalc - modKetch99.c3) + modKetch99.c2)
        
        self.reduced_lengths = np.array(reduced_lengths)
        self.first_node = first_node
        return self.reduced_lengths, self.first_node


class Generic(AnnealingModel):


    def __init__(self, model_parameters: dict, use_projected_track: bool =False,
                 use_Cf_irradiation: bool =False):

        self.model_parameters = model_parameters

        super(Generic, self).__init__(
                use_projected_track,
                use_Cf_irradiation)

    def annealing_model(self):

        # Must be in seconds (do conversion)
        cdef double[::1] time = np.ascontiguousarray(self.history.time * _seconds_in_megayears)
        # Must be in Kelvin
        cdef double[::1] temperature = np.ascontiguousarray(self.history.temperature)
        cdef int numTTnodes = time.shape[0]
        cdef double[::1] reduced_lengths = np.zeros(time.shape[0] - 1)
        cdef int first_node = 0

        cdef int node, nodeB
        cdef double equivTime
        cdef double timeInt, x1, x2, x3
        cdef double k
        cdef double calc
        cdef double tempCalc

        cdef annealModel model = annealModel(
            c0=self.model_parameters["c0"],
            c1=self.model_parameters["c1"],
            c2=self.model_parameters["c2"],
            c3=self.model_parameters["c3"],
            a=self.model_parameters["a"],
            b=self.model_parameters["b"])

        equivTime = 0.
        tempCalc = 1.0 / ((temperature[numTTnodes - 2] +  temperature[numTTnodes - 1]) / 2.0)

        for node in range(numTTnodes - 2, -1, -1):
            # We calculate the modeled reduced length (length normalized by
            # initial length of a fission track parallel to the c-axis (Donelick 1999))
            # after an isothermal annealing episode at a temperature T (Kelvin) of
            # duration t (seconds)
            timeInt = time[node] - time[node + 1] + equivTime
            x1 = (log(timeInt) - model.c2) / (tempCalc - model.c3)
            x2 = 1.0 + model.a * (model.c0 + model.c1 * x1)

            if x2 < 0:
                reduced_lengths[node] = 0.0
            else:
                reduced_lengths[node] = pow(x2, 1.0 / model.a)
                if x3 < 0:
                    reduced_lengths[node] = 0.
                else:
                    x3 = 1.0 - model.b * reduced_lengths[node]
                    reduced_lengths[node] = pow(x3, 1.0 / model.b)

            # Check to see if we've reached the end of the length distribution
            # If so, we then do the kinetic conversion.
            if reduced_lengths[node] == 0.0 or node == 0:
                if node > 0:
                    node += 1
                first_node = node
                self.reduced_lengths = np.array(reduced_lengths)
                self.first_node = first_node
                return self.reduced_lengths, self.first_node

            # Update tiq for this time step
            if reduced_lengths[node] < 0.999:
                tempCalc = 1.0 / ((temperature[node-1] + temperature[node]) / 2.0)
                equivTime = pow((1.0 - pow(reduced_lengths[node], model.b)) / model.b, model.a)
                equivTime = ((equivTime - 1.0) / model.a - model.c0) / model.c1
                equivTime = exp(equivTime * (tempCalc - model.c3) + model.c2)


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

    def __init__(self, kinetic_parameters: bool, use_projected_track: bool =False,
                 use_Cf_irradiation: bool=False):
        
        self._kinetic_parameters = kinetic_parameters

        super(Ketcham2007, self).__init__(
              use_projected_track,
              use_Cf_irradiation)
    
    @property
    def kinetic_parameters(self):
        return self._kinetic_parameters

    @kinetic_parameters.setter
    def kinetic_parameters(self, value):
        self._kinetic_parameters = value
    
    @property
    def rmr0(self):
        kinetic_type = list(self.kinetic_parameters.keys())[0]
        kinetic_value = self.kinetic_parameters[kinetic_type]
        return self._kinetic_conversion[kinetic_type].__func__(kinetic_value)

    def annealing_model(self):
        cdef double[::1] time = np.ascontiguousarray(self.history.time * _seconds_in_megayears)
        cdef double[::1] temperature = np.ascontiguousarray(self.history.temperature)
        cdef int numTTnodes = time.shape[0]
        cdef double[::1] reduced_lengths = np.zeros(time.shape[0] - 1)
        cdef double crmr0 = self.rmr0
        cdef int first_node = 0

        cdef int node, nodeB
        cdef double equivTime
        cdef double timeInt, x1, x2
        cdef double totAnnealLen
        cdef double equivTotAnnLen
        cdef double k
        cdef double calc
        cdef double tempCalc
        cdef double MIN_OBS_RCMOD = _MIN_OBS_RCMOD

        # Fanning Curvilinear Model lcMod FC, See Ketcham 1999, Table 5e
        cdef annealModel modKetch07 = annealModel(
            c0=0.39528,
            c1=0.01073,
            c2=-65.12969,
            c3=-7.91715,
            a=0.04672,
            b=0)

        k = 1.04 - crmr0

        totAnnealLen = MIN_OBS_RCMOD
        equivTotAnnLen =  pow(totAnnealLen, 1.0 / k) * (1.0 - crmr0) + crmr0

        equivTime = 0.
        tempCalc = log(1.0 / ((temperature[numTTnodes - 2] +  temperature[numTTnodes - 1]) / 2.0))

        for node in range(numTTnodes - 2, -1, -1):
            timeInt = time[node] - time[node + 1] + equivTime
            x1 = (log(timeInt) - modKetch07.c2) / (tempCalc - modKetch07.c3)
            x2 = pow(modKetch07.c0 + modKetch07.c1 * x1, 1.0 / modKetch07.a) + 1.0
            reduced_lengths[node] = 1.0 / x2

            if reduced_lengths[node] < equivTotAnnLen:
                reduced_lengths[node] = 0.
            # Check to see if we've reached the end of the length distribution
            # If so, we then do the kinetic conversion.
            if reduced_lengths[node] == 0.0 or node == 0:
                if node > 0:
                    node += 1
                first_node = node

                for nodeB in range(first_node, numTTnodes - 1):
                    if reduced_lengths[nodeB] < crmr0:
                        reduced_lengths[nodeB] = 0.0
                        first_node = nodeB
                    else:
                        # This is equation 8 from Ketcham et al, 1999
                        reduced_lengths[nodeB] = pow((reduced_lengths[nodeB] - crmr0) / (1.0 - crmr0), k)
                        if reduced_lengths[nodeB] < totAnnealLen:
                            reduced_lengths[nodeB] = 0.
                            first_node = nodeB
        
                self.reduced_lengths = np.array(reduced_lengths)
                self.first_node = first_node
                return self.reduced_lengths, self.first_node

            # Update tiq for this time step
            if reduced_lengths[node] < 0.999:
                tempCalc = log(1.0 / ((temperature[node-1] + temperature[node]) / 2.0))
                equivTime = pow(1.0 / reduced_lengths[node] - 1.0, modKetch07.a)
                equivTime = (equivTime - modKetch07.c0) / modKetch07.c1
                equivTime = exp(equivTime * (tempCalc - modKetch07.c3) + modKetch07.c2)
