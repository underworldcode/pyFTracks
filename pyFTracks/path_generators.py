import numpy as np

class MonteCarloPathGenerator(object):
    
    def __init__(self, time_range, temperature_range, npaths=1000, inbetween_points=2):
        
        self.time_range = np.array(time_range)
        self.fact_time = self.time_range[-1]
        self.time_range = self.time_range / self.fact_time

        self.temperature_range = np.array(temperature_range)
        self.fact_temperature = self.temperature_range[-1]
        self.temperature_range = self.temperature_range / self.fact_temperature
        
        self.inbetween_points = self.n = inbetween_points
        self.npaths = npaths

        self.constraints = []
        
        self.add_constraint({'time': (0., 0.), 'temperature': (0., 20.)})
        self.TTPaths = None
        
    def add_constraint(self, constraint):
        
        if isinstance(constraint, list):
            self.constraints += constraint
        else:    
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

        temperature = np.random.rand(self.npaths, npoints)

        for index, constrain in enumerate(self.constraints):
            constrain_time = constrain['time'] / self.fact_time
            mask = ~np.any((time >= min(constrain_time)) & (time <= max(constrain_time)), axis=1)
            time[mask, index] = np.random.rand(np.count_nonzero(mask),) * (max(constrain_time) - min(constrain_time)) + min(constrain_time)

        time = np.sort(time, axis=1)    

        for index, constrain in enumerate(self.constraints):
            constrain_temp = constrain['temperature'] / self.fact_temperature
            constrain_time = constrain['time'] / self.fact_time
            i, j = np.where((time >= min(constrain_time)) & (time <= max(constrain_time)))
            shape = i.shape[0]
            temperature[i, j] = np.random.rand(shape,) * (max(constrain_temp) - min(constrain_temp)) + min(constrain_temp) 
            
        self.TTPaths = np.ndarray((self.npaths, npoints, 2))
        self.TTPaths[:, :, 0] = time * self.fact_time
        self.TTPaths[:, :, 1] = temperature * self.fact_temperature
        return self.TTPaths
    
    def plot_paths(self, new=False):
        
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection

        if new:
            self.generate_paths()
        fig, ax = plt.subplots()
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
        return ax