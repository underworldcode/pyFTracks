import numpy as np
from ketcham import isothermal_intervals
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib as mpl
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


class ThermalHistoryViewer(object):

    def __init__(self, time, temperature):

        self.input_time = time
        self.input_temperature = temperature
        self.original_time = np.copy(time)
        self.original_temperature = np.copy(temperature)

        # figure.subplot.right
        mpl.rcParams['figure.subplot.right'] = 0.8

        # set up a plot
        self.fig, self.ax1 = plt.subplots(1, 1, figsize=(9.0, 8.0), sharex=True)
        self.pind = None  # active point
        self.epsilon = 5  # max pixel distance

        self.ax1.plot(
            self.original_time,
            self.original_temperature,
            'k--', label='original')

        self.l, = self.ax1.plot(
            self.input_time,
            self.input_temperature,
            color='k', linestyle='none',
            marker='o', markersize=8)

        self.m, = self.ax1.plot(self.input_time, self.input_temperature, 'r-')

        self.ax1.set_yscale('linear')
        self.ax1.set_xlim(np.max(self.input_time), 0.)
        self.ax1.set_ylim(np.max(self.input_temperature), 0.)
        self.ax1.set_xlabel('Time (Myr)')
        self.ax1.set_ylabel('Temperature (C)')
        self.ax1.grid(True)
        self.ax1.yaxis.grid(True, which='minor', linestyle='--')
        self.ax1.legend(loc=4, prop={'size': 10})

        self.axres = plt.axes([0.84, 0.8-((self.original_time.shape[0])*0.05), 0.12, 0.02])
        self.bres = Button(self.axres, 'Test')
        self.bres.on_clicked(self.reset)

        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.fig.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

        plt.show()

    def reset(self, event):
        """ Reset the values """
        self.input_temperature = np.copy(self.original_temperature)
        self.input_time = np.copy(self.original_time)
        self.l.set_ydata(self.input_temperature)
        self.m.set_ydata(self.input_temperature)
        self.l.set_xdata(self.input_time)
        self.m.set_xdata(self.input_time)
        plt.draw()

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self.pind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        """ Whenever a mouse button is released """
        if event.button != 1:
            return
        self.pind = None

    def get_ind_under_point(self, event):
        """
           Get the index of the vertex under point if within epsilon tolerance
        """
        # display coords
        t = self.ax1.transData.inverted()
        tinv = self.ax1.transData
        #xy = t.transform([event.x, event.y])
        xr = np.reshape(self.input_time, (np.shape(self.input_time)[0], 1))
        yr = np.reshape(self.input_temperature, (np.shape(self.input_temperature)[0], 1))
        xy_vals = np.append(xr, yr, 1)
        xyt = tinv.transform(xy_vals)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def motion_notify_callback(self, event):
        'on mouse movement'
        if self.pind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        self.input_temperature[self.pind] = event.ydata
        self.input_time[self.pind] = event.xdata
        self.l.set_ydata(self.input_temperature)
        self.m.set_ydata(self.input_temperature)
        self.l.set_xdata(self.input_time)
        self.m.set_xdata(self.input_time)
        self.fig.canvas.draw_idle()


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
    time=u.Quantity([0., 19.5, 19., 100.], u.megayears),
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
