from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from matplotlib.patches import Arc
from matplotlib import collections  as mc
import numpy as np
import math
import matplotlib.pyplot as plt

LAMBDA = 1.55125e-10
G = 0.5


class Radialplot(Axes):

    """
    A RadialPlot or Galbraith Plot
    """

    name = "radialplot"

    LAMBDA = 1.55125e-4
    ZETA = 350
    RHOD = 1.304

    def radialplot(self, Ns, Ni, zeta, rhod, Dpars=None, transform="Logarithmic"):
        
        self.Ns = np.array(Ns)
        self.Ni = np.array(Ni)
        self.zeta = zeta
        self.rhod = rhod
        self.Dpars = Dpars
        self.transform = transform
        
        # Prepare the plot Area
        
        # Left spine
        self.set_ylim(-8, 8)
        self.set_yticks([-2, -1, 0, 1, 2])
        self.spines["left"].set_bounds(-2, 2)
       
        # Bottom spine
        xticks_max = np.int(np.rint(self.max_x))
        xticks_interval = 2
        xticks = range(0, xticks_max + xticks_interval, xticks_interval)
        self.set_xticks(xticks)
        self.set_xlim(0, self.max_x + 0.4 * self.max_x)
        self.spines["bottom"].set_bounds(0, xticks_max)
        
        self.spines["top"].set_visible(False)
        self.spines["right"].set_visible(False)
            
        # Only show ticks on the left and bottom spines
        self.yaxis.set_ticks_position('left')
        self.xaxis.set_ticks_position('bottom')
        self.xaxis.set_tick_params(direction="out", pad=-15)

        im=self.scatter(self.x, self.y, c=Dpars)
        if Dpars:
            plt.gcf().colorbar(im, ax=self, orientation="horizontal")
        self._add_radial_axis()
        self._add_values_indicators()

    @property
    def max_x(self):
        return max(self.x)
        
    @property
    def z(self):
        """ Return transformed z-values"""

        if self.transform == "Linear":
            return 1.0 / LAMBDA * np.log(1.0 + G * self.zeta * LAMBDA * self.rhod * (self.Ns / self.Ni))
            
        if self.transform == "Logarithmic":
            return np.log(G * self.zeta * LAMBDA * self.rhod * (self.Ns / self.Ni))
           
        if self.transform == "arcsine":
            return np.asin(np.sqrt((self.Ns + 3.0/8.0) / (self.Ns + self.Ni + 3.0 / 4.0)))
        
    @property
    def sez(self):
        """Return standard errors"""
        
        if self.transform == "Linear":
            return z * np.sqrt( 1.0 / self.Ns + 1.0 / self.Ni)

        if self.transform == "Logarithmic":
            return np.sqrt(1.0 / self.Ns + 1.0 / self.Ni)

        if self.transform == "arcsine":
            return 1.0 / (2.0 * np.sqrt(self.Ns + self.Ni))
        
    @property
    def z0(self):
        """ Return central age"""
        
        if self.transform == "Linear":
            return np.sum(self.z / self.sez**2) / np.sum(1 / self.sez**2)

        if self.transform == "Logarithmic":
            totalNs = np.sum(self.Ns)
            totalNi = np.sum(self.Ni)
            return np.log(G * self.zeta * LAMBDA * self.rhod * (totalNs / totalNi))

        if self.transform == "arcsine":
            return np.asin(np.sqrt(np.sum(self.Ns) / np.sum(self.Ns + self.Ni)))
    
    @property
    def x(self):            
        return  1 / self.sez
    
    @property
    def y(self):
        return (self.z - self.z0) / self.sez
    
    def _z2t(self, z):
        
        if self.transform == "Linear":
            return z
        if self.transform == "Logarithmic":
            NsNi = np.exp(z) / (self.zeta * G * LAMBDA * self.rhod)
    
        t = 1.0 / LAMBDA * np.log(1.0 + G * self.zeta * LAMBDA * self.rhod * (NsNi))
        return t
    
    def _t2z(self, t):
        
        if self.transform == "Linear":
            return t
        if self.transform == "Logarithmic":
            return np.log(np.exp(LAMBDA * t) - 1)
    
    def _add_radial_axis(self):
        # Get min and max angle
        zr = self._get_radial_ticks_z()
        theta1 = np.rad2deg(np.min(zr))
        theta2 = np.rad2deg(np.max(zr))

        width = 2.0 * (self.max_x + 0.2 * self.max_x)
        height = width

        # The circle is always centered around 0.
        # Width and height are equals (circle)
        arc_element = Arc(
            (0, 0), width, height, theta1=theta1,
            theta2=theta2, linewidth=1, zorder=0, color="k")

        self.add_patch(arc_element)
        
        # Add ticks
        self._add_radial_ticks()
        self._add_radial_ticks_labels()
        
    def _add_radial_ticks(self, nticks=10):

        zr = self._get_radial_ticks_z()

        # Lets build a line collection
        R1 = (1 + 0.2) * self.max_x
        R2 = R1 + 0.01 * R1
        x1 = R1 * np.cos(zr)
        y1 = R1 * np.sin(zr)
        x2 = R2 * np.cos(zr)
        y2 = R2 * np.sin(zr)
        
        starts = list(zip(x1, y1))
        ends = list(zip(x2, y2))
        segments = zip(starts, ends)

        lc = mc.LineCollection(segments, colors='k', linewidths=2)
        self.add_collection(lc)
        
    def _get_radial_ticks_z(self):
        # Let's build the ticks of the Age axis
        za = self._get_radial_ticks_ages()
        zr = self._t2z(np.array(za) * 1e6) - self.z0
        return zr
    
    def _get_radial_ticks_ages(self, nticks=10):
        ages = self._z2t(self.z) * 1e-6
        start, end = np.int(np.rint(min(ages))), np.int(np.rint(max(ages)))
        start, end, interval = nice_bounds(start, end, nticks)
        za = range(int(start), int(end) + int(interval), int(interval))
        return za
        
    def _add_values_indicators(self):
        R1 = (1.0 + 0.2 - 0.02) * self.max_x
        R2 = (1.0 + 0.2 - 0.01) * self.max_x
        ratio = self.y / self.x
        x1 = R1 * np.cos(ratio)
        y1 = R1 * np.sin(ratio)
        x2 = R2 * np.cos(ratio)
        y2 = R2 * np.sin(ratio)

        starts = list(zip(x1, y1))
        ends = list(zip(x2, y2))
        segments = zip(starts, ends)

        lc = mc.LineCollection(segments, colors='k', linewidths=2)
        self.add_collection(lc) 
        
    def _add_radial_ticks_labels(self):
        # text label
        R3 = (1 + 0.2) * self.max_x
        R3 += 0.02 * (1 + 0.2) * self.max_x
        za = self._get_radial_ticks_ages()
        labels = self._t2z(np.array(za) * 1e6)
        labels -= self.z0
        x1 = R3 * np.cos(labels)
        y1 = R3 * np.sin(labels)

        for idx, val in enumerate(za):
            self.text(x1[idx], y1[idx], str(val)+ "Ma") 
            
def nice_number(value, round_=False):
    '''nice_number(value, round_=False) -> float'''
    exponent = math.floor(math.log(value, 10))
    fraction = value / 10 ** exponent

    if round_:
        if fraction < 1.5: nice_fraction = 1.
        elif fraction < 3.: nice_fraction = 2.
        elif fraction < 7.: nice_fraction = 5.
        else: niceFraction = 10.
    else:
        if fraction <= 1: nice_fraction = 1.
        elif fraction <= 2: nice_fraction = 2.
        elif fraction <= 5: nice_fraction = 5.
        else: nice_fraction = 10.

    return nice_fraction * 10 ** exponent

def nice_bounds(axis_start, axis_end, num_ticks=10):
    '''
    nice_bounds(axis_start, axis_end, num_ticks=10) -> tuple
    @return: tuple as (nice_axis_start, nice_axis_end, nice_tick_width)
    '''
    axis_width = axis_end - axis_start
    if axis_width == 0:
        nice_tick = 0
    else:
        nice_range = nice_number(axis_width)
        nice_tick = nice_number(nice_range / (num_ticks -1), round_=True)
        axis_start = math.floor(axis_start / nice_tick) * nice_tick
        axis_end = math.ceil(axis_end / nice_tick) * nice_tick

    return axis_start, axis_end, nice_tick

        
register_projection(Radialplot)


def radialplot(Ns, Ni, zeta, rhod, Dpars=None, transform="Logarithmic"):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="radialplot")
    ax.radialplot(Ns, Ni, zeta, rhod, Dpars, transform)
    return ax
