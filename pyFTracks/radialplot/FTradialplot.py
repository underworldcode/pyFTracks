from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from matplotlib.patches import Arc
from matplotlib import collections  as mc
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import MaxNLocator
from .radialplot import ZAxis, Radialplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .age_calculations import calculate_central_age, calculate_pooled_age, calculate_ages

LAMBDA = 1.55125e-4
G = 0.5

class ZAxisFT(ZAxis):
    
    def _add_radial_axis(self):
        # Get min and max angle

        theta1 = self.ax._t2axis_angle(self.zlim[0])
        theta2 = self.ax._t2axis_angle(self.zlim[1])

        # The circle is always centered around 0.
        # Width and height are equals (circle)
        # Here the easiest is probably to use axis coordinates. The Arc
        # is always centered at (0.,0.) and 

        height = width = 2.0 * self.radius
        arc_element = Arc(
            (0, 0.5), width, height, angle=0., theta1=theta1,
            theta2=theta2, linewidth=1, zorder=0, color="k",
            transform=self.ax.transAxes)

        self.ax.add_patch(arc_element)
        
        # Add ticks
        self.ticks()
        self.labels()
        self.set_zlabel("Age Estimates (Myr)")
        self.add_values_indicators()
    
    def _get_radial_ticks_z(self):
        # Let's build the ticks of the Age axis
        za = self.ticks_locator()
        zr = self.ax._t2z(np.array(za)) - self.ax.z0
        return za
    
    def labels(self):
        # text label
        ticks = self.ticks_locator()
        angles = np.array([self.ax._t2axis_angle(val) for val in ticks])
        x = 1.02 * self.radius * np.cos(np.deg2rad(angles))
        y = 1.02 * self.radius * np.sin(np.deg2rad(angles)) + 0.5

        for idx, val in enumerate(ticks):
            self.ax.text(x[idx], y[idx], str(val), transform=self.ax.transAxes)

    def ticks(self):

        ticks = self.ticks_locator()
        angles = np.array([self.ax._t2axis_angle(val) for val in ticks])
        starts = np.ndarray((len(angles), 2))
        ends = np.ndarray((len(angles), 2))
        starts[:,0] = self.radius * np.cos(np.deg2rad(angles))
        starts[:,1] = self.radius * np.sin(np.deg2rad(angles)) + 0.5
        ends[:,0] = 1.01 * self.radius * np.cos(np.deg2rad(angles))
        ends[:,1] = 1.01 * self.radius * np.sin(np.deg2rad(angles)) + 0.5

        segments = np.stack((starts, ends), axis=1)
        lc = mc.LineCollection(segments, colors='k', linewidths=1, transform=self.ax.transAxes)
        self.ax.add_collection(lc)
    
    def ticks_locator(self, ticks=None):
        if not ticks:
            ages = self.ax._z2t(self.ax.z)
            start, end = np.int(np.rint(min(ages))), np.int(np.rint(max(ages)))
            loc = MaxNLocator()
            ticks = loc.tick_values(start, end)
        return ticks

class FTRadialplot(Radialplot):

    """A RadiaPlot for fission track counts
    
    Returns:
        FTRadialPlot: Radialplot
    """

    name = "fission_track_radialplot"

    def radialplot(self, Ns, Ni, zeta, zeta_err, rhod, rhod_err, 
                   Dpars=None, name="unknown", 
                   transform="logarithmic", **kwargs):
       
        self.Ns = np.array(Ns)
        self.Ni = np.array(Ni)
        Ns = self.Ns[(self.Ns > 0) & (self.Ni > 0)]
        Ni = self.Ni[(self.Ns > 0) & (self.Ni > 0)]
        self.Ns = Ns
        self.Ni = Ni
        # Zeta and Zeta err have units of 10e-6 cm2
        self.zeta = zeta
        self.zeta_err = zeta_err
        self.rhod = rhod
        self.rhod_err = rhod_err
        self.Dpars = Dpars
        self.name = name
        self.transform = transform

        # Prepare the plot Area
        # Left spine
        self.set_ylim(-8, 8)
        self.set_yticks([-2, -1, 0, 1, 2])
        self.spines["left"].set_bounds(-2, 2)
        self.yaxis.set_ticks_position('left')
       
        self.set_xlim()
        self.set_xticks()
        
        self.spines["top"].set_visible(False)
        self.spines["right"].set_visible(False)
            
        im=self.scatter(self.x, self.y, c=Dpars, cmap="YlOrRd", **kwargs)
        if Dpars:
            divider = make_axes_locatable(self)
            if self.transform == "logarithmic":
                divider = make_axes_locatable(self.taxis)
            cax = divider.new_vertical(size="5%", pad=0.8, pack_start=True)
            self.figure.add_axes(cax)
            self.figure.colorbar(im, cax=cax, orientation="horizontal", label=r'Dpars ($\mu$m)')

        self._add_sigma_lines()
        self._add_shaded_area()
        self._add_central_line()
        self._add_stats()

        self.zaxis = ZAxisFT(self)
        self.zaxis._add_radial_axis()

        # Apply some default labels:
        self.set_ylabel("Standardised estimate y")

    def _second_axis(self):
        
        def tick_function(x):
            with np.errstate(divide='ignore'):
                v = 1./ x
            return ["{0}%".format(int(val*100)) if val != np.inf else "" for val in v]

        twin_axis = self.twiny()
        twin_axis.set_xlim(self.get_xlim())
        
        loc = MaxNLocator(5)
        ticks = loc.tick_values(0, self.max_x)
        twin_axis.spines["bottom"].set_bounds(ticks[0], ticks[-1])

        twin_axis.xaxis.set_ticks_position("bottom")
        twin_axis.xaxis.set_label_position("bottom")
        twin_axis.tick_params(axis="x", direction="in", pad=-15)
        twin_axis.spines["bottom"].set_position(("axes", 0.))
        twin_axis.set_frame_on(True)
        twin_axis.patch.set_visible(False)
        for key, sp in twin_axis.spines.items():
            sp.set_visible(False)
        twin_axis.spines["bottom"].set_visible(True)

        twin_axis.set_xticks(ticks)
        twin_axis.set_xticklabels(tick_function(ticks))
        twin_axis.set_xlabel(r'$\sigma / t$', labelpad=-30)
        
        self.taxis = twin_axis
        return

    def set_xticks(self, ticks=None):
        if ticks:
            super(Radialplot, self).set_xticks(ticks)
        else:
            if self.transform == "linear":
                loc = MaxNLocator(5)
                ticks = loc.tick_values(0., self.max_x)
                ticks2 = loc.tick_values(min(self.sez), max(self.sez))
                ticks2 = ticks2[::-1]
                ticks2[-1] = min(self.sez)
                super(Radialplot, self).set_xticks(1.0 / ticks2)
                labels = [str(int(val)) for val in ticks2]
                self.xaxis.set_ticklabels(labels)
                self.spines["bottom"].set_bounds(0., 1. / ticks2[-1])
                self.set_xlabel(r'$\sigma$ (Myr)')
            elif self.transform == "logarithmic":
                loc = MaxNLocator(5)
                ticks = loc.tick_values(0., self.max_x)
                super(Radialplot, self).set_xticks(ticks)
                self.spines["bottom"].set_bounds(ticks[0], ticks[-1])
                self.set_xlabel(r'$t / \sigma$')
                self._second_axis()
            elif self.transform == "arcsine":
                loc = MaxNLocator(5)
                ticks = loc.tick_values(0., self.max_x)
                super(Radialplot, self).set_xticks(ticks)
                labels = [str(int(val**2/4.0)) for val in ticks]
                self.xaxis.set_ticklabels(labels)
                self.spines["bottom"].set_bounds(ticks[0], ticks[-1])
                self.set_xlabel("Ns + Ni")

    @property
    def z(self):
        """ Return transformed z-values"""
        if self.transform == "linear":
            return  1.0 / LAMBDA * np.log(1.0 + G * self.zeta * LAMBDA * self.rhod * (self.Ns / self.Ni))

        if self.transform == "logarithmic":
            return np.log(G * self.zeta * LAMBDA * self.rhod * (self.Ns / self.Ni))
           
        if self.transform == "arcsine":
            return np.arcsin(np.sqrt((self.Ns + 3.0/8.0) / (self.Ns + self.Ni + 3.0 / 4.0)))
        
    @property
    def sez(self):
        """Return standard errors"""
        
        if self.transform == "linear":
            return self.z * np.sqrt( 1.0 / self.Ns + 1.0 / self.Ni)

        if self.transform == "logarithmic":
            return np.sqrt(1.0 / self.Ns + 1.0 / self.Ni)

        if self.transform == "arcsine":
            return 1.0 / (2.0 * np.sqrt(self.Ns + self.Ni))
        
    @property
    def z0(self):
        """ Return central age"""
        
        if self.transform == "linear":
            return np.sum(self.z / self.sez**2) / np.sum(1 / self.sez**2)

        if self.transform == "logarithmic":
            totalNs = np.sum(self.Ns)
            totalNi = np.sum(self.Ni)
            return np.log(G * self.zeta * LAMBDA * self.rhod * (totalNs / totalNi))

        if self.transform == "arcsine":
            return np.arcsin(np.sqrt(np.sum(self.Ns) / np.sum(self.Ns + self.Ni)))
    
    def _z2t(self, z):
        
        if self.transform == "linear":
            t = z
            return t
        elif self.transform == "logarithmic":
            NsNi = np.exp(z) / (self.zeta * G * LAMBDA * self.rhod)
        elif self.transform == "arcsine":
            NsNi = np.sin(z)**2 / (1.0 - np.sin(z)**2)
    
        t = 1.0 / LAMBDA * np.log(1.0 + G * self.zeta * LAMBDA * self.rhod * (NsNi))
        return t
    
    def _t2z(self, t):

        if t == 0:
            return 0

        if self.transform == "linear":
            return t
        elif self.transform == "logarithmic":
            return np.log(np.exp(LAMBDA * t) - 1)
        elif self.transform == "arcsine":
            return np.arcsin(
                    1.0 / np.sqrt(
                        1.0 + LAMBDA * self.zeta * G * self.rhod / (np.exp(LAMBDA * t) - 1.0)
                        )
                    )
   
    def get_central_age(self):
        data = calculate_central_age(self.Ns, self.Ni, self.zeta, self.zeta_err, self.rhod, self.rhod_err)
        self.central_age = data["Central"]
        self.central_age_error = data["se"]
        return data

    def get_pooled_age(self):
        data = calculate_pooled_age(self.Ns, self.Ni, self.zeta, self.zeta_err, self.rhod, self.rhod_err)
        self.pooled_age = data["Pooled Age"]
        self.pooled_age_error = data["se"]
        return data

    def get_single_ages(self):
        data = calculate_ages(self.Ns, self.Ni, self.zeta, self.zeta_err, self.rhod, self.rhod_err)
        return data

    def _add_stats(self):
        
        self.get_central_age()
        self.get_pooled_age()
        data = self.get_single_ages()
        
        self.mean_age = np.mean(data["Age(s)"])
        self.mean_age_error = np.mean(data["se(s)"])

        text =  "{name} (n={n}) \n".format(name=self.name, n=len(self.Ns))
        text += "Central Age = {central_age:5.2f} +/- {central_age_error:5.2f} (1$\sigma$) \n".format(
                central_age=self.central_age, central_age_error=self.central_age_error
                )
        text += "Pooled Age = {pooled_age:5.2f} +/- {pooled_age_error:5.2f} (1$\sigma$) \n".format(
                pooled_age=self.pooled_age, pooled_age_error=self.pooled_age_error
                )
        text += "Mean Age = {mean_age:5.2f} +/- {mean_age_error:5.2f} (1$\sigma$) \n".format(
                mean_age=self.mean_age, mean_age_error=self.mean_age_error
                )
        text += "Dispersion = {dispersion} % \n".format(dispersion=0.)
        text += "P($\chi^2$) = {chi2}".format(chi2=0.)
        self.text(0., 0.95, text,
                  horizontalalignment="left", verticalalignment="top",
                  transform=self.transAxes)
        return

register_projection(FTRadialplot)

def radialplot(Ns=None, Ni=None, zeta=None, zeta_err=0., rhod=None, rhod_err=0., file=None,
               Dpars=None, name="unknown", transform="logarithmic", **kwargs):
    """Plot Fission Track counts using a RadialPlot (Galbraith Plot)
    
    Args:
        Ns (list or numpy array, optional): 
            Spontaneous counts. 
            Defaults to None.
        Ni (list or numpy array, optional): 
            Induced counts. 
            Defaults to None.
        zeta (float, optional): 
            Zeta calibration parameter.
            Defaults to None.
        zeta_err (float, optional): 
            Uncertainty on Zeta calibration parameter.
            Defaults to 0.
        rhod (float, optional): 
            Rhod calibration parameter.
            Defaults to None.
        rhod_err (float, optional): 
            Uncertainty on Rhod calibration parameter.
            Defaults to None.
        file (string, optional): 
            Data File, for now pyRadialPlot only accepts
            file format similar to RadialPlotter.
            Defaults to None.
        Dpars (list or numpy array or float, optional): 
            Dpars values associated with the grain counts.
            Defaults to None.
        transform (str, optional): 
            Transformation used.
            Options are "linear", "logarithmic", "arcsine".
            Defaults to "logarithmic".
        kwargs: Matplotlib additional parameters.
    
    Returns:
        matplotlib.Axes: 
            A Matplotlib Axes object.
    """

    fig = plt.figure(figsize=(6,6))
    if file:
        from .utilities import read_radialplotter_file
        data = read_radialplotter_file(file)
        Ns = data["Ns"]
        Ni = data["Ni"]
        zeta = data["zeta"]
        zeta_err = data["zeta_err"]
        rhod = data["rhod"]
        rhod_err = data["rhod_err"]
        if Dpars:
            Dpars = data["dpars"]
    
    if not Dpars and not "color" in kwargs.keys():
        kwargs["color"] = "black"

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="fission_track_radialplot")
    ax.radialplot(Ns, Ni, zeta, zeta_err, rhod, rhod_err, Dpars, name=name,
                  transform=transform, **kwargs)
    return ax
