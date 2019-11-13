import numpy as np
from .ketcham import isothermal_intervals
import pint

u = pint.UnitRegistry()


class ThermalHistory(object):
    """Class defining a thermal history"""

    def __init__(self, time, temperature, name="unknown"):
        self.name = name
        self.input_time = time.to(u.megayears)
        self.input_temperature = temperature.to_base_units()

        self.time = self.input_time.magnitude
        self.temperature = self.input_temperature.magnitude

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
        self.time, self.temperature = isothermal_intervals(
            self.input_time, self.input_temperature, max_temperature_per_step,
            max_temperature_step_near_ta)
        return


# Some useful thermal histories
WOLF1 = ThermalHistory(
    name="wolf1",
    time=u.Quantity([0., 43., 44., 100.], u.megayears),
    temperature=u.Quantity([10., 10., 130., 130.], u.degC)
)

WOLF2 = ThermalHistory(
    name="wolf2",
    time=u.Quantity([0., 100.], u.megayears),
    temperature=u.Quantity([10., 130], u.degC)
)

WOLF3 = ThermalHistory(
    name="wolf3",
    time=u.Quantity([0., 19., 19.5, 100.], u.megayears),
    temperature=u.Quantity([10., 10., 60., 60.], u.degC)
)

WOLF4 = ThermalHistory(
    name="wolf4",
    time=u.Quantity([0., 24., 76., 100.], u.megayears),
    temperature=u.Quantity([10., 60., 60., 100],  u.degC)
)

WOLF5 = ThermalHistory(
    name="wolf5",
    time=u.Quantity([0., 5., 100.], u.megayears),
    temperature=u.Quantity([10., 64., 18.], u.degC)
)
