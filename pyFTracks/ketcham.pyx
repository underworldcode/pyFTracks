import cython
import numpy as np
cimport numpy as np

cdef extern from "include/utilities.h":

    cdef int refine_history(
        double *time, double *temperature, int npoints,
        double max_temp_per_step, double max_temp_step_near_ta,
        double *new_time, double *new_temperature, int *new_npoints)
    cdef void ketcham_sum_population(
        int numPDFPts, int numTTNodes, int firstTTNode, int doProject,
        int usedCf, double *time, double *temperature, double *pdfAxis,
        double *pdf, double *cdf, double  initLength, double min_length,
        double  *redLength)
    cdef void ketcham_calculate_model_age(
        double *time, double *temperature, double  *redLength,
        int numTTNodes, int firstNode, double  *oldestModelAge,
        double *ftModelAge, double stdLengthReduction, double *redDensity)

cdef extern from "include/ketcham1999.h":
    
    cdef void ketch99_reduced_lengths(
        double *time, double *temperature,int numTTNodes, double *redLength,
        double kinPar, int kinParType, int *firstTTNode, int etchant)

cdef extern from "include/ketcham2007.h":
    
    cdef void ketch07_reduced_lengths(
        double *time, double *temperature,int numTTNodes, double *redLength,
        double kinPar, int kinParType, int *firstTTNode, int etchant)

def isothermal_intervals(time, temperature, max_temp_per_step, max_temp_step_near_ta):

    time = np.ascontiguousarray(time)
    temperature = np.ascontiguousarray(temperature)

    cdef double[::1] time_memview = time
    cdef double[::1] temperature_memview = temperature
    cdef double[::1] new_time = np.ndarray((200))
    cdef double[::1] new_temperature = np.ndarray((200))
    cdef int* new_npoints
    cdef int a=0

    new_npoints = &a

    refine_history(&time_memview[0], &temperature_memview[0], time_memview.shape[0],
                   max_temp_per_step, max_temp_step_near_ta,
                   &new_time[0], &new_temperature[0], new_npoints)
    return np.array(new_time)[:new_npoints[0]], np.array(new_temperature)[:new_npoints[0]]


def ketcham99_annealing_model(time, temperature,
                              int kinetic_parameter_type,
                              double kinetic_parameter_value,
                              int nbins, int etchant):

    time = np.ascontiguousarray(time)
    temperature = np.ascontiguousarray(temperature)

    cdef int* first_node
    cdef int a = 0
    cdef double[::1] time_memview = time
    cdef double[::1] temperature_memview = temperature
    cdef double[::1] reduced_lengths = np.zeros((nbins))

    first_node = &a

    ketch99_reduced_lengths(&time_memview[0], &temperature_memview[0],
                            time_memview.shape[0], &reduced_lengths[0],
                            kinetic_parameter_value,
                            kinetic_parameter_type,
                            first_node, etchant)

    return np.array(reduced_lengths), first_node[0]


def ketcham07_annealing_model(time, temperature,
                              int kinetic_parameter_type,
                              double kinetic_parameter_value,
                              int nbins, int etchant):

    time = np.ascontiguousarray(time)
    temperature = np.ascontiguousarray(temperature)

    cdef int* first_node
    cdef int a = 0
    cdef double[::1] time_memview = time
    cdef double[::1] temperature_memview = temperature
    cdef double[::1] reduced_lengths = np.zeros((nbins))

    first_node = &a

    ketch07_reduced_lengths(&time_memview[0], &temperature_memview[0],
                            time_memview.shape[0], &reduced_lengths[0],
                            kinetic_parameter_value,
                            kinetic_parameter_type,
                            first_node, etchant)

    return np.array(reduced_lengths), first_node[0]


def sum_population(time, temperature, reduced_lengths,
        int first_node, double init_length, double min_length,
        int nbins, int project, int usedCf):

    time = np.ascontiguousarray(time)
    temperature = np.ascontiguousarray(temperature)
    reduced_lengths = np.ascontiguousarray(reduced_lengths)

    cdef double[::1] time_memview = time
    cdef double[::1] temperature_memview = temperature
    cdef double[::1] reduced_lengths_memview = reduced_lengths
    cdef double[::1] pdfAxis = np.zeros((nbins))
    cdef double[::1] cdf = np.zeros((nbins))
    cdef double[::1] pdf = np.zeros((nbins))

    ketcham_sum_population(nbins, time_memview.shape[0], first_node,
                           <int> project, <int> usedCf, &time_memview[0],
                           &temperature_memview[0], &pdfAxis[0], &pdf[0],
                           &cdf[0], init_length, min_length,
                           &reduced_lengths_memview[0])
    
    return (np.array(pdfAxis), np.array(pdf), np.array(cdf))


def calculate_model_age(time, temperature, reduced_lengths,
                         int first_node, double std_length_reduction):

    time = np.ascontiguousarray(time)
    temperature = np.ascontiguousarray(temperature)
    reduced_lengths = np.ascontiguousarray(reduced_lengths)

    cdef double[::1] time_memview = time
    cdef double[::1] temperature_memview = temperature
    cdef double[::1] reduced_lengths_memview = reduced_lengths

    cdef double* oldest_age
    cdef double* ft_model_age
    cdef double* reduced_density
    cdef double val1 = 0.
    cdef double val2 = 0.
    cdef double val3 = 0.

    oldest_age = &val1
    ft_model_age = &val2
    reduced_density = &val3


    ketcham_calculate_model_age(&time_memview[0], &temperature_memview[0],
                                &reduced_lengths_memview[0], time_memview.shape[0],
                                first_node, oldest_age, ft_model_age,
                                std_length_reduction, reduced_density)

    return (oldest_age[0], ft_model_age[0], reduced_density[0])
