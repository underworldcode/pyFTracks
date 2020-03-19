import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib.backend_bases import MouseButton
from .thermal_history import ThermalHistory


class Viewer(object):

    def __init__(self,history=None,
                 forward_model=None,
                 sample=None,
                 kinetic_parameter_type="ETCH_PIT_LENGTH",
                 kinetic_parameter_value=1.65,
                 track_l0=16.3,
                 present_temperature=293.15):

        if history:
            self.history = history
            self.time = np.array(self.history.input_time)
            self.temperature = np.array(self.history.input_temperature)
            self.present_temperature = self.temperature[-1]
        else:
            self.present_temperature = present_temperature
            self.time = np.array([0.])
            self.temperature = np.array([self.present_temperature])
            self.history = ThermalHistory(self.time, self.temperature) 

        self.fwd_model = forward_model
        if not self.fwd_model:
            from .annealing import Ketcham2007
            self.fwd_model = Ketcham2007(self.history)

        self.original_time = np.copy(self.time)
        self.original_temperature = np.copy(self.temperature)
        self.kinetic_parameter_value = kinetic_parameter_value
        self.kinetic_parameter_type = kinetic_parameter_type
        self.track_l0 = track_l0
        self.sample = sample

        self.pind = None  # active point
        self.epsilon = 1
        self.init_plot()

    def init_plot(self):

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(9.0, 8.0))

        self.ax1.plot(
            self.original_time,
            self.original_temperature,
            'k--', label='original')

        self.l, = self.ax1.plot(
            self.time,
            self.temperature,
            color='k', linestyle="-",
            marker='o', markersize=8)

        self.ax1.set_yscale('linear')
        self.ax1.set_title("Thermal History")
        if np.max(self.time) > 0.0:
            self.ax1.set_xlim(np.max(self.time), 0.)
            self.ax1.set_ylim(np.max(self.temperature) + 50, 273.15)
        else:
            self.ax1.set_xlim(100, 0.)
            self.ax1.set_ylim(500., 273.15)
        self.ax1.set_xlabel('Time (Myr)')
        self.ax1.set_ylabel('Temperature (C)')
        self.ax1.grid(True)
        self.ax1.yaxis.grid(True, which='minor', linestyle='--')
        self.ax1.legend(loc=4, prop={'size': 10})

        if self.fwd_model:
            self.fwd_model.solve(self.track_l0, self.kinetic_parameter_type, self.kinetic_parameter_value)
            self.m2, = self.ax2.plot(self.fwd_model.pdf_axis, self.fwd_model.pdf, color="r")
            age_label = self.fwd_model.ft_model_age
            MTL_label = self.fwd_model.MTL
        else:
            self.ax2.plot()
            age_label = 0.0
            MTL_label = 0.0

        self.age_label = self.ax2.text(0.05, 0.95, "AFT age:{0:5.2f} Myr".format(age_label),
                                       horizontalalignment='left', verticalalignment='center',
                                       transform=self.ax2.transAxes)
        self.MTL_label = self.ax2.text(0.05, 0.90, "MTL:{0:5.2f} $\mu$m".format(MTL_label),
                                       horizontalalignment='left', verticalalignment='center',
                                       transform=self.ax2.transAxes)
        self.ax2.set_title("Fission Track prediction")
        self.ax2.set_ylim(0., 0.05)
        self.ax3 = self.ax2.twinx()
        self.ax3.set_ylim(0., 40)
        #self._synthetic_lengths = self.fwd_model.generate_synthetic_lengths(100)
        #self.ax3.hist(self._synthetic_lengths, range=(0., 20.), bins=20, rwidth=0.8)

        self.axres = plt.axes([0.84, 0.05, 0.12, 0.02])
        self.bres = Button(self.axres, 'Reset')
        self.bres.on_clicked(self.reset)

        self.fig.canvas.mpl_connect('button_press_event',
                                    self.on_press)
        self.fig.canvas.mpl_connect('button_release_event',
                                    self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',
                                    self.on_motion)

    def update_plot(self):
        self.l.set_ydata(self.temperature)
        self.l.set_xdata(self.time)

        if self.history:
            self.m2.set_ydata(self.fwd_model.pdf)
            age_label = self.fwd_model.ft_model_age
            MTL_label = self.fwd_model.MTL
        else:
            self.ax2.plot()
            age_label = 0.0
            MTL_label = 0.0

        self.age_label.set_text("AFT age:{0:5.2f} Myr".format(age_label))
        self.MTL_label.set_text("MTL:{0:5.2f} $\mu$m".format(MTL_label))
        self.fig.canvas.draw_idle()

    def reset(self, event):
        self.temperature = np.copy(self.original_temperature)
        self.time = np.copy(self.original_time)
        self.fwd_model.pdf *= 0.
        self.refresh_data()
        self.update_plot()

    def on_press(self, event):
        if event.inaxes is None:
            return
        if event.button == MouseButton.LEFT:
            d, self.pind = self.find_closest_point(event)
            if d[self.pind] > self.epsilon:
                self.add_point(event)
        if event.button == MouseButton.RIGHT:
            d, self.pind = self.find_closest_point(event)
            if d[self.pind] >= self.epsilon:
                self.pind = None
            self.delete_point()
        self.refresh_data()
        self.update_plot()

    def on_release(self, event):
        if event.button != 1:
            return
        self.pind = None
        self.refresh_data()
        self.update_plot()

    def on_motion(self, event):
        if self.pind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        if self.pind == 0:
            self.temperature[self.pind] = event.ydata

        if self.pind > 0:
            if self.pind < self.time.shape[0] - 1:
                if (event.xdata < self.time[self.pind + 1] and
                   event.xdata > self.time[self.pind - 1]):
                    self.temperature[self.pind] = event.ydata
                    self.time[self.pind] = event.xdata
            else:
                if event.xdata > self.time[self.pind - 1]:
                    self.temperature[self.pind] = event.ydata
                    self.time[self.pind] = event.xdata

        self.update_plot()

    def add_point(self, event):
        self.time = np.insert(self.time, 0, event.xdata)
        self.temperature = np.insert(self.temperature, 0, event.ydata)
        indices = np.argsort(self.time)
        self.time = np.sort(self.time)
        self.temperature = self.temperature[indices]

    def find_closest_point(self, event):
        d = np.abs(self.time - event.xdata)
        ind = d.argmin()
        return d, ind

    def refresh_data(self):
        self.fwd_model.history.input_time = np.copy(self.time)
        self.fwd_model.history.input_temperature = np.copy(self.temperature)
        if self.time.shape[0] > 1:
            self.fwd_model.history.get_isothermal_intervals()
            self.fwd_model.solve(self.track_l0, self.kinetic_parameter_type, self.kinetic_parameter_value)
        #self._synthetic_lengths = self.fwd_model.generate_synthetic_lengths(100)

    def delete_point(self):
        if not self.pind:
            return
        self.time = np.delete(self.time, self.pind)
        self.temperature = np.delete(self.temperature, self.pind)

