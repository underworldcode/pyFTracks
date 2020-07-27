import numpy as np
import cython
import numpy as np
cimport numpy as np

cdef extern from "include/utilities.h":

    cdef int refine_history(
        double *time, double *temperature, int npoints,
        double max_temp_per_step, double max_temp_step_near_ta,
        double *new_time, double *new_temperature, int *new_npoints)


class ThermalHistory(object):
    """Class defining a thermal history"""

    def __init__(self, time, temperature, name="unknown"):
        """ 
         time: list of time points in Myr.
         temperature: list of temperature points in deg Kelvin.
         name: a name for the thermal-history.

        """
        self.name = name
        self.input_time = time
        self.input_temperature = temperature

        self.time = self.input_time
        self.temperature = self.input_temperature

        self.maxT = max(temperature)
        self.minT = min(temperature)
        self.totaltime = max(time) - min(time)
        self.dTdt = np.diff(self.temperature) / np.diff(self.time)
        self.get_isothermal_intervals()

    def get_isothermal_intervals(self, max_temperature_per_step=8.0,
                                 max_temperature_step_near_ta=3.5):
        """
        The more segments a time-temperature path is subdivided into
        the more accurate the numerical solution.
        Issler (1996) demonstrated that time steps should be smaller
        as we approach total annealing temperature.
        For the Ketcham et 1999 model for F-apatite, Ketcham 2000
        found that 0.5 precision is assured if the there is no step
        greater than a 3.5 degrees change within 10 degrees
        of the F-apatite total annealing temperature.

        We set a maximum temperature step of 3.5 degrees C when the model
        temperature is within 10 degrees of the total annealing temperature.
        Before this cutoff the maximum temperature step required is 8C
        """

        time = np.ascontiguousarray(self.input_time)
        temperature = np.ascontiguousarray(self.input_temperature)

        cdef double[::1] time_memview = time
        cdef double[::1] temperature_memview = temperature
        cdef double[::1] new_time = np.ndarray((200))
        cdef double[::1] new_temperature = np.ndarray((200))
        cdef double cmax_temp_per_step = max_temperature_per_step
        cdef double cmax_temp_step_near_ta = max_temperature_step_near_ta
        cdef int* new_npoints
        cdef int a=0

        new_npoints = &a

        refine_history(&time_memview[0], &temperature_memview[0], time_memview.shape[0],
                       cmax_temp_per_step, cmax_temp_step_near_ta,
                       &new_time[0], &new_temperature[0], new_npoints)
        self.time = np.array(new_time)[:new_npoints[0]]
        self.temperature = np.array(new_temperature)[:new_npoints[0]]
        return self.time, self.temperature

# Some useful thermal histories
WOLF1 = ThermalHistory(
    name="wolf1",
    time=[0., 43., 44., 100.],
    temperature=[283.15, 283.15, 403.15, 403.15]
)

WOLF2 = ThermalHistory(
    name="wolf2",
    time=[0., 100.],
    temperature=[283.15, 403.15]
)

WOLF3 = ThermalHistory(
    name="wolf3",
    time=[0., 19., 19.5, 100.],
    temperature=[283.15, 283.15, 333.15, 333.15]
)

WOLF4 = ThermalHistory(
    name="wolf4",
    time=[0., 24., 76., 100.],
    temperature=[283.15, 333.15, 333.15, 373.15]
)

WOLF5 = ThermalHistory(
    name="wolf5",
    time=[0., 5., 100.],
    temperature=[283.15, 373.15, 291.15]
)


FLAXMANS1 = ThermalHistory(
    name="Flaxmans1",
    #time=[109.73154362416108, 95.97315436241611, 65.10067114093958, 42.95302013422818, 27.069351230425042, 0.223713646532417], 
    #temperature=[10.472716661803325, 50.21343115594648, 90.20426028596441, 104.6346242027596, 124.63170619867442, 125.47709366793116]
    time=[0., 27, 43, 65, 96, 110], 
    temperature=[125, 126, 105, 90, 50, 10.]
)

VROLIJ = ThermalHistory(
    name="Vrolij",
    #time=[112.84098861592778, 108.92457659225633, 101.04350962294087, 95.96509833052357, 4.910255922414279, -0.196743768208961],
    #temperature=[10.036248368327097, 14.455174285000524, 14.971122078085369, 18.945174136102615, 11.984858737246478, 11.027412104738094]
    time=[0., 5., 96., 101, 109, 113],
    temperature=[11, 12, 19, 15, 14, 10.]
)
