import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib.backend_bases import MouseButton


class Viewer(object):

    def __init__(self, forward_model=None,
                 sample=None,
                 kinetic_parameter_type="ETCH_PIT_LENGTH",
                 kinetic_parameter_value=1.65,
                 track_l0=16.3):

        if forward_model:
            self.time = np.array(forward_model.history.input_time)
            self.temperature = np.array(forward_model.history.input_temperature)
        else:
            self.time = np.empty()
            self.temperature = np.empty()

        self.original_time = np.copy(self.time)
        self.original_temperature = np.copy(self.temperature)
        self.fwd_model = forward_model
        self.kinetic_parameter_value = kinetic_parameter_value
        self.kinetic_parameter_type = kinetic_parameter_type
        self.track_l0 = track_l0
        self.sample = sample

        self.pind = None  # active point
        self.epsilon = 20  # max pixel distance
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
        self.ax1.set_xlim(np.max(self.time), 0.)
        self.ax1.set_ylim(np.max(self.temperature) + 50, 273.15)
        self.ax1.set_xlabel('Time (Myr)')
        self.ax1.set_ylabel('Temperature (C)')
        self.ax1.grid(True)
        self.ax1.yaxis.grid(True, which='minor', linestyle='--')
        self.ax1.legend(loc=4, prop={'size': 10})

        self.fwd_model.solve(self.track_l0, self.kinetic_parameter_type, self.kinetic_parameter_value)
        self._synthetic_lengths = self.fwd_model.generate_synthetic_lengths(100)
        self.ax2.hist(self._synthetic_lengths, range=(0., 20.), bins=20, rwidth=0.8)
        self.ax2.set_ylim(0., 40)
        self.ax3 = self.ax2.twinx()
        self.m2, = self.ax3.plot(self.fwd_model.pdf_axis, self.fwd_model.pdf, color="r")
        self.age_label = self.ax3.text(0.05, 0.9, "AFT age:{0:5.2f}".format(self.fwd_model.ft_model_age),
                                       horizontalalignment='left', verticalalignment='center',
                                       transform=self.ax3.transAxes)
        self.MTL_label = self.ax3.text(0.05, 0.85, "MTL:{0:5.2f}".format(self.fwd_model.MTL),
                                       horizontalalignment='left', verticalalignment='center',
                                       transform=self.ax3.transAxes)
        self.ax3.set_title("Fission Track prediction")
        self.ax3.set_ylim(0., 0.05)

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
        if self.time.shape[0] >= 2:
            self.refresh_data()
        self.l.set_ydata(self.temperature)
        self.l.set_xdata(self.time)
        self.m2.set_ydata(self.fwd_model.pdf)
        self.ax2.cla()
        if self.time.shape[0] >= 2:
            self.ax2.hist(self._synthetic_lengths, range=(0., 20.), bins=20, rwidth=0.8)
        self.ax2.set_ylim(0., 40)
        self.age_label.set_text("AFT age:{0:5.2f}".format(self.fwd_model.ft_model_age))
        self.MTL_label.set_text("MTL:{0:5.2f}".format(self.fwd_model.MTL))
        self.fig.canvas.draw_idle()

    def reset(self, event):
        self.temperature = np.copy(self.original_temperature)
        self.time = np.copy(self.original_time)
        self.update_plot()

    def on_press(self, event):
        if event.inaxes is None:
            return
        if event.button == MouseButton.LEFT:
            d, self.pind = self.find_closest_point(event)
            if d[self.pind] >= self.epsilon:
                self.pind = None
        if event.button == MouseButton.RIGHT:
            d, self.pind = self.find_closest_point(event)
            if d[self.pind] >= self.epsilon:
                self.pind = None
            self.delete_point()
        if event.dblclick:
            d, self.pind = self.find_closest_point(event)
            self.add_point(event)

    def on_release(self, event):
        if event.button != 1:
            return
        self.pind = None

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
            if (event.xdata < self.time[self.pind + 1] and
               event.xdata > self.time[self.pind - 1]):
                self.temperature[self.pind] = event.ydata
                self.time[self.pind] = event.xdata

        self.update_plot()

    def add_point(self, event):
        xclosest = self.time[self.pind]
        if self.pind:
            if self.pind == 0:
                self.time = (
                    np.insert(self.time, self.pind + 1, event.xdata)
                )
                self.temperature = (
                    np.insert(self.temperature, self.pind + 1, event.ydata)
                )
                self.update_plot()

            if self.pind == self.time.shape[0] - 1:
                self.time = (
                    np.insert(self.time, self.pind, event.xdata)
                )
                self.temperature = (
                    np.insert(self.temperature, self.pind, event.ydata)
                )
                self.update_plot()

            if (event.xdata < self.time[self.pind + 1] and
               event.xdata > self.time[self.pind - 1]):

                if event.xdata < xclosest:
                    self.time = (
                        np.insert(self.time, self.pind, event.xdata)
                    )
                    self.temperature = (
                        np.insert(self.temperature, self.pind, event.ydata)
                    )
                else:
                    self.time = (
                        np.insert(self.time, self.pind + 1, event.xdata)
                    )
                    self.temperature = (
                        np.insert(self.temperature, self.pind + 1, event.ydata)
                    )
                self.update_plot()

    def find_closest_point(self, event):
        tinv = self.ax1.transData
        xy_vals = np.ndarray((self.time.shape[0], 2))
        xy_vals[:, 0] = self.time
        xy_vals[:, 1] = self.temperature
        xyt = tinv.transform(xy_vals)
        mouse = (event.x, event.y)
        d, = cdist([mouse], xyt)
        ind = d.argmin()
        return d, ind

    def refresh_data(self):
        self.fwd_model.history.input_time = np.copy(self.time)
        self.fwd_model.history.input_temperature = np.copy(self.temperature)
        self.fwd_model.history.get_isothermal_intervals()
        self.fwd_model.solve(self.track_l0, self.kinetic_parameter_type, self.kinetic_parameter_value)
        self._synthetic_lengths = self.fwd_model.generate_synthetic_lengths(100)

    def delete_point(self):
        if not self.pind:
            return
        self.time = np.delete(self.time, self.pind)
        self.temperature = np.delete(self.temperature, self.pind)
        self.update_plot()

