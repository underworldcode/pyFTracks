import pylab as plt
import random
import numpy as np
from matplotlib import collections  as mc
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

def _radius(a, se, f):
    """Return Ellipse radius at max precision position"""
    r = max(np.sqrt(se**2 + (f*a)**2))
    return r

def _f(xlim, ylim):
    # Calculate conversion factor for plot coordinates
    return (max(xlim) - min(xlim)) / (max(ylim)- min(ylim))

def _zlim(values):
    zspan = (np.mean(values) *0.5) / (np.std(values) * 100)
    zspan = 0.9 if zspan > 1.0 else zspan
    zlim = (np.floor((0.9 - zspan)*min(values)),
            np.ceil((1.1 + zspan)*max(values)))
    return zlim

def plot_line(val, central, r, f, **kwargs):
    ax = plt.gca()
    x1,y1 = (0,0)
    x2 = r / np.sqrt(1 + f**2*(val-central)**2)
    y2 = x2*(val-central)
    ax.plot((x1,x2), (y1, y2), **kwargs)
    
def get_ellipse_coord(val,r,f,central):
    x = r / np.sqrt(1 + f**2*(val - central)**2)
    y = x * (val - central)
    return x, y

def plot_ticks(x1,y1,x2,y2):
    ax = plt.gca()
    A = list(zip(x1, y1))
    B = list(zip(x2, y2))
    segments = list(zip(A, B))
    lc = mc.LineCollection(segments)
    ax.add_collection(lc)
    lc.set_color("k")

def plot_rugs(val, central, f, r):
    Rin = 0.980 * r
    Rout = 0.993 * r
    x1, y1 = get_ellipse_coord(val, Rin, f, central)
    x2, y2 = get_ellipse_coord(val, Rout, f, central)
    plot_ticks(x1,y1,x2,y2)

def upper_axis_formatter(x, pos):
    if x != 0:
        return '%1.1f'% (1/x)
    else:
        return ''

class Radialplot():

    def __init__(self, values, errors, central):

        self.values = values
        self.errors = errors
        self.central = central

        # Calculate standard estimates and z-values
        self.se = 1.0 / errors
        self.z = (values - central) / errors
        
        # Define Axes limits
        self.zticks_major = 1.015
        self.zticks_minor = 1.007
        self.zlabels = 1.02
        self.rellipse = 1.1
        
        minx = 0.0
        maxx = 1.2*max(self.se)
        miny = -15
        maxy = 15
        
        self.xlim=(minx,maxx)
        self.ylim=(miny,maxy)
        
        self.f = _f(self.xlim, self.ylim)
        # Now we need to create a z-axis
        self.r = self.rellipse*_radius(self.z, self.se, self.f)
        # Calculate z-span
        self.zlim = _zlim(values)
        
        
        # Create z-ticks
        locator = MaxNLocator(integer="true", nbins=5, prune="both", symetric="True")
        self.ticks_values_major = locator.tick_values(*self.zlim)
        locator = MaxNLocator(integer="true", nbins=20, prune="both", symetric="True")
        self.ticks_values_minor = locator.tick_values(*self.zlim)
        
        # Calculate major z-ticks coordinates
        self.ticks_x1_major, self.ticks_y1_major = get_ellipse_coord(self.ticks_values_major, 
                                                                     self.r, self.f, 
                                                                     self.central)
        self.ticks_x2_major, self.ticks_y2_major = get_ellipse_coord(self.ticks_values_major, 
                                                                     self.zticks_major*self.r,
                                                                     self.f,
                                                                     self.central)
        
        # Calculate minor z-ticks coordinates
        self.ticks_x1_minor, self.ticks_y1_minor = get_ellipse_coord(self.ticks_values_minor, 
                                                                     self.r,self.f,
                                                                     self.central)
        self.ticks_x2_minor, self.ticks_y2_minor = get_ellipse_coord(self.ticks_values_minor, 
                                                                     self.zticks_minor*self.r,
                                                                     self.f,
                                                                     self.central)
        
        # Calculate z-labels positions
        self.xlab, self.ylab = get_ellipse_coord(self.ticks_values_major,self.zlabels*self.r,
                                                 self.f, self.central)
        
        # Ellipse
        self.ellipse_values = np.linspace(min(np.hstack((self.ticks_values_major,
                                                    self.ticks_values_minor))), 
                                    max(np.hstack((self.ticks_values_major, 
                                                   self.ticks_values_minor))), 500)
        
        self.ellipse_x, self.ellipse_y =  get_ellipse_coord(self.ellipse_values,
                                                            self.r, self.f, 
                                                            self.central)
   
    def plot(self):
        ## Plotting
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.25)
        ax = fig.add_subplot(111)
       
        # Plot 2 sigma rectangle
        rect1 = mpatches.Rectangle((0,-2),self.xlim[-1], 4, color="pink")
        ax.add_patch(rect1)
        inner_x, inner_y =  get_ellipse_coord(self.ellipse_values,
                                              self.r, self.f, 
                                              self.central)
        inners = list(zip(inner_x, inner_y))
        outer_x, outer_y =  get_ellipse_coord(self.ellipse_values,
                                              2*self.r, self.f, 
                                              self.central)
        outers = list(reversed(list(zip(outer_x, outer_y))))

        polygon = inners + outers
        polygon = mpatches.Polygon(polygon, color="white")
        ax.add_patch(polygon)
        ## end of 2 sigma rectangle

        plot_rugs(self.values, self.central, self.f, self.r)
        
        ax.set_yticks([-2,0,2])
        ax.spines["left"].set_bounds(-2,2)
        ax.spines["bottom"].set_bounds(self.xlim[0], max(self.se))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(direction="in", pad=-15)
        
        ax.plot(self.ellipse_x, self.ellipse_y, c="k")
        plot_ticks(self.ticks_x1_major, self.ticks_y1_major,
                   self.ticks_x2_major, self.ticks_y2_major)
        plot_ticks(self.ticks_x1_minor, self.ticks_y1_minor,
                   self.ticks_x2_minor, self.ticks_y2_minor)
        
        
        # Add labels z-axis
        for index, label in enumerate(zip(self.xlab, self.ylab)):
            ax.text(*(label+"{}".format(lab=self.ticks_values_major[index])))
        
        # Plot data
        ax.plot(self.se, self.z, marker="o", linestyle="")
        
        # Plot central value line
        plot_line(self.central, self.central, self.r, self.f, c="k")

        
        ax.plot(self.ellipse_x, self.ellipse_y, c="k")
        plot_ticks(self.ticks_x1_major, self.ticks_y1_major,
                   self.ticks_x2_major, self.ticks_y2_major)
        plot_ticks(self.ticks_x1_minor, self.ticks_y1_minor,
                   self.ticks_x2_minor, self.ticks_y2_minor)
        
        
        ax.set_ylim(self.ylim)
        ax.set_xlim(self.xlim)
        # Change locator for ticks
        locator = MaxNLocator(nbins=5, prune="upper")
        ticks = locator.tick_values(*self.xlim)
        ax.xaxis.set_ticks(ticks)
        ax.set_ylabel("Standardised estimates") 
        ax.spines["bottom"].set_bounds(self.xlim[0], max(ticks))
        ax.set_adjustable("box-forced")
        formatter = FuncFormatter(upper_axis_formatter)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("Standard Error", labelpad=-40)

        newax = ax.twiny()
        newax.set_xlim(self.xlim)
        newax.set_ylim(self.ylim)
        newax.spines["top"].set_visible(False)
        newax.spines["right"].set_visible(False)
        newax.yaxis.set_ticks_position('left')
        newax.xaxis.set_ticks_position('bottom')
        newax.xaxis.set_tick_params(direction="out", pad=15)
        newax.spines['bottom'].set_position(('outward', 20))
        locator = MaxNLocator(nbins=5, prune="upper")
        ticks = locator.tick_values(*self.xlim)
        newax.xaxis.set_ticks(ticks)
        #newax.set_xlabel("Precision", labelpad=20)
        newax.spines["bottom"].set_bounds(self.xlim[0], max(ticks))
        newax.set_adjustable("box-forced")
        ax.set_aspect(self.f)
        newax.set_aspect(self.f)


if __name__== "__main__":

    central = 100.
    values = np.random.normal(central,10, size=1000)
    errors = np.random.normal(10,1, size=1000)

    radialplot = Radialplot(values, errors, central)
    radialplot.plot()
    plt.show()
