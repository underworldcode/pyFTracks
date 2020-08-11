import numpy as np
from pyFTracks.thermal_history import ThermalHistory
import scipy

class MonteCarloPathGenerator(object):
    
    def __init__(self, time_range, temperature_range, npaths=1000, inbetween_points=2):
        
        time_range = np.array(time_range)
        temperature_range = np.array(temperature_range)

        # If time is not increasing, reverse arrays
        if not np.all(np.diff(time_range) > 0):
            time_range = time_range[::-1]

        if np.any(temperature_range < 273.):
            print("It looks like you have entered temperature in Celsius...Converting temperature to Kelvin")
            temperature_range = temperature_range + 273.15  

        self.time_range = np.array(time_range)
        self.fact_time = self.time_range[-1]
        self.time_range = self.time_range / self.fact_time

        self.temperature_range = np.array(temperature_range)
        self.fact_temperature = np.diff(self.temperature_range)
        self.temperature_range = (self.temperature_range - 273.15) / self.fact_temperature
        
        self.inbetween_points = self.n = inbetween_points
        self.npaths = npaths

        self.constraints = []
        
        self.add_constraint({'time': (0., 0.), 'temperature': (0., 20.)})
        self.TTPaths = None
        self._annealing_model = None
        self.goodness_of_fit_values = None

    @property
    def annealing_model(self):
        return self._annealing_model

    @annealing_model.setter
    def annealing_model(self, value):
        self._annealing_model = value
        
    def add_constraint(self, constraint):

        def convert_time(time):
            time = np.array(time)
            if np.all(time < 0.):
                time = time[::-1]
            return time

        def convert_temperature(temperature):
            temperature = np.array(temperature) 
            if np.any(temperature < 273):
                temperature = temperature + 273.15
            return temperature
        
        if isinstance(constraint, list):
            self.constraints += constraint
            for item in constraint:
                item["time"] = convert_time(item["time"])
                item["temperature"] = convert_temperature(item["temperature"])
        else:    
            constraint["time"] = convert_time(constraint["time"])
            constraint["temperature"] = convert_temperature(constraint["temperature"])
            self.constraints.append(constraint)
        return self.constraints
    
    def clear_constraints(self):
        self.constraints = []
        
    def generate_paths(self):
        
        nconstraints = len(self.constraints)
        npoints = nconstraints * (1 + (2**self.n - 1))

        time = np.random.rand(self.npaths, npoints)
        time = (1.0 - time)
        # Final time is always present time
        time[:, -1] = 0.

        for index, constrain in enumerate(self.constraints):
            constrain_time = constrain['time'] / self.fact_time
            mask = ~np.any((time >= min(constrain_time)) & (time <= max(constrain_time)), axis=1)
            time[mask, index] = np.random.rand(np.count_nonzero(mask),) * (max(constrain_time) - min(constrain_time)) + min(constrain_time)

        time = np.sort(time, axis=1)    

        temperature = np.random.rand(self.npaths, npoints)

        for index, constrain in enumerate(self.constraints):
            constrain_temp = (constrain['temperature'] - 273.15) / self.fact_temperature
            constrain_time = constrain['time'] / self.fact_time
            i, j = np.where((time >= min(constrain_time)) & (time <= max(constrain_time)))
            shape = i.shape[0]
            temperature[i, j] = np.random.rand(shape,) * (max(constrain_temp) - min(constrain_temp)) + min(constrain_temp) 
            
        self.TTPaths = np.ndarray((self.npaths, npoints, 2))
        self.TTPaths[:, :, 0] = time * self.fact_time
        self.TTPaths[:, :, 1] = temperature * self.fact_temperature + 273.15
        return self.TTPaths

    def run(self, measured_lengths, measured_age, measured_age_error):

        if not self.annealing_model:
            raise ValueError("""Please provide an Annealing Model""")

        self.goodness_of_fit_values = []

        for path in self.TTPaths:
            time, temperature = path[:, 0], path[:, 1]
            history = ThermalHistory(time, temperature)
            self.annealing_model.history = history
            self.annealing_model.calculate_age()
            self.goodness_of_fit_values.append(self.merit_function(measured_lengths, measured_age, measured_age_error))

        # sort TTPaths
        self.goodness_of_fit_values = np.array(self.goodness_of_fit_values)
        self.TTPaths = self.TTPaths[np.argsort(self.goodness_of_fit_values)][::-1]
        self.goodness_of_fit_values = self.goodness_of_fit_values[np.argsort(self.goodness_of_fit_values)][::-1]

        return

    def merit_function(self, measured_lengths, age=None, age_error=None):
        # We first evaluated goodness of fit for the track length distribution using 
        # a Kolmogorov-Smirnov test.
        # The test relies on 2 parameters:
        # 1) The maximum separation between 2 cumulative distribution functions which represent the
        # measured and modelled track length.
        # 2) The number of measuruments

        # The result of the test is the probability that a set of samples taken randomly
        # from the known modelled distribution would have a greater maximum separation from it on 
        # a cdf plot than is observed for the sample distribution being tested.

        # The number of tracks counted is the statistical constrain on how well-defined the fission
        # track length distribution is, we assume that the model distribution is completely known and
        # we test the measured distribution against it

        # A K-S probability of 0.05 means that, if N random samples were taken from the distribution
        # described by the calculation result, where N is the number of FT length actually measured, there would be a 5% chance that
        # the resulting distribution would have a greater maximum separation from the model on a cdf plot than
        # is observed between the data and the model.
        KS_test_lengths = scipy.stats.rv_discrete(values=(self.annealing_model.pdf_axis, self.annealing_model.pdf))
        KS_test_lengths = scipy.stats.kstest(measured_lengths, KS_test_lengths.cdf)[1]
        
        # Now do the age
        norm = scipy.stats.norm()
        value = (self.annealing_model.ft_model_age - age) / age_error
        KS_test_age = 1.0 - scipy.stats.kstest(np.array([value]), norm.cdf)[1]
        return min(KS_test_age, KS_test_lengths)


    def plot_paths(self, new=False):
        
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        import matplotlib as mpl

        #Create a new colormap
        cmap = mpl.colors.ListedColormap(["pink", "green", "grey"])
        bounds = [0., 0.05, 0.5, 1.0]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        if self.goodness_of_fit_values is not None and not new:
            colors = cmap(norm(self.goodness_of_fit_values))
        else:
            colors = "grey"

        if new:
            self.generate_paths()
        fig, ax = plt.gcf(), plt.gca()
        ax.set_xlim(self.TTPaths[:, :, 0].max(), self.TTPaths[:, :, 0].min())
        ax.set_ylim(self.TTPaths[:, :, 1].max(), self.TTPaths[:, :, 1].min())
        
        lines = LineCollection(self.TTPaths, linestyle='solid', colors=colors)
        ax.add_collection(lines)       
        
        patches = []
        
        for constrain in self.constraints:
            dx = abs(constrain["time"][1] - constrain["time"][0])
            dy = abs(constrain["temperature"][1] - constrain["temperature"][0])
            x = constrain["time"][0]
            y = constrain["temperature"][0]
            patches.append(Rectangle([x, y], dx, dy))
            
        rectangles = PatchCollection(patches, color="red",  facecolor='none', zorder=20)
        ax.add_collection(rectangles)
        
        ax.set_title('Time Temperature Paths')
        ax.set_xlabel('Time in Myr')
        ax.set_ylabel('Tempeature in Kelvins')
        return ax