from itertools import count
import numpy as np
from collections import OrderedDict
from pandas import Series, DataFrame
from .radialplot import radialplot
import pandas as pd
from .utilities import read_mtx_file, h5load, h5store
from .age_calculations import calculate_ages
from .age_calculations import calculate_pooled_age
from .age_calculations import calculate_central_age
from .age_calculations import chi_square

unprojected_coefs = {"ETCH_PIT_LENGTH": {"m": 0.283, "b": 15.63},
               "CL_PFU": {"m": 0.544, "b": 16.18},
               "OH_PFU": {"m": 0.0, "b": 16.18},
               "CL_WT_PCT": {"m": 0.13824, "b": 16.288}}

projected_coefs = {"ETCH_PIT_LENGTH": {"m": 0.205, "b": 16.10},
             "CL_PFU": {"m": 0.407, "b": 16.49},
             "OH_PFU": {"m": 0.000, "b": 16.57},
             "CL_WT_PCT": {"m": 0.17317, "b": 16.495}}


class Grain(Series):

    _metadata = ["_track_length_distribution", "track_length_distribution"]

    def __init__(self, *args, **kwargs):
        Series.__init__(self, *args, **kwargs)
        
        self._track_length_distribution = pd.DataFrame(columns=["bins", "lengths"])

    @property
    def _constructor(self):
        return Grain

    @property
    def _constructor_expanddim(self):
        return Sample
    
    @property
    def track_length_distribution(self):
        return self._track_length_distribution

    @track_length_distribution.setter
    def track_length_distribution(self, values):
        self._track_length_distribution = values

    
class Sample(DataFrame):

    # Let pandas know what properties are added
    _metadata = ['name', 'zeta', 'zeta_error', 
            'pooled_age', 'pooled_age_se', 'central_age',
            'central_age_se', 'central_age_se',
            'rhod', 'nd', 'depth', 'elevation',
            'stratigraphic_age', 'stratigraphic_age_name',
            'unit_area_graticule',
            'deposition_temperature', 'present_day_temperature', 'id',
            '_track_length_distribution', 'track_length_distribution']
            
    def __init__(self, *args, **kwargs):
       
        #for element in self._metadata:
        #    if element in kwargs.keys():
        #        setattr(self, element, kwargs.pop(element))
        #    else:
        #        setattr(self, element, None)

        super(Sample, self).__init__(*args, **kwargs)

        self._track_length_distribution = pd.DataFrame(columns=["bins", "lengths"])

        if self.empty:
            self.insert(loc=0, column="Ns", value=None)
            self.insert(loc=1, column="Ni", value=None)
            self.insert(loc=2, column="A", value=None)
            self.insert(loc=3, column="Ns/Ni", value=None)
            self.insert(loc=4, column="RhoS", value=None)
            self.insert(loc=5, column="RhoI", value=None)

    @property
    def _constructor(self):
        return Sample

    @property
    def _constructor_sliced(self):
        return Grain

    def read_from_hdf5(self, filename):
        with pd.HDFStore(filename) as store:
            data, metadata = h5load(store)
        for val in self._metadata:
            try:
                key = val.replace("_", " ")
                setattr(self, val, metadata.pop(key))
            except:
                pass
        self.__init__(data)
        self._calculate_statistics()
        return self

    def _calculate_statistics(self):
        self["Ns/Ni"] = self.Ns / self.Ni
        if not hasattr(self, "unit_area_graticule"):
            self.unit_area_graticule = 1.0
        if not hasattr(self, "A"):
            self.A = 1
        self["RhoS"] = self.Ns / (self.A * self.unit_area_graticule)
        self["RhoI"] = self.Ni / (self.A * self.unit_area_graticule)
        self.calculate_ages()

    def read_from_radialplotter(self, filename):
        from pyRadialPlot import read_radialplotter_file
        data = read_radialplotter_file(filename)
         
        self.__init__({"Ns": data["Ns"], "Ni": data["Ni"]})
        self.zeta = data["zeta"]
        self.rhod = data["rhod"]
        self._calculate_statistics()
    
    def calculate_l0_from_Dpars(self, projected=True):
        if projected:
            m = projected_coefs["ETCH_PIT_LENGTH"]["m"]
            b = projected_coefs["ETCH_PIT_LENGTH"]["b"]
        else:
            m = unprojected_coefs["ETCH_PIT_LENGTH"]["m"]
            b = unprojected_coefs["ETCH_PIT_LENGTH"]["b"]
        if not hasattr(self, "Dpars"):
            raise ValueError("Cannot find Dpars column")
        self["l0"] = m * self.Dpars + b

    def calculate_ages(self):
        data =  calculate_ages(
                self.Ns, self.Ni, self.zeta, 
                self.zeta_error, self.rhod, self.nd)
        self["Ages"] = data["Age(s)"]
        self["Ages Errors"] = data["se(s)"]

    @property
    def track_length_distribution(self):
        return self._track_length_distribution

    @track_length_distribution.setter
    def track_length_distribution(self, values):
        self._track_length_distribution = values

    def calculate_pooled_age(self):
        data = calculate_pooled_age(
                self.Ns, self.Ni, self.zeta,
                self.zeta_error, self.rhod, self.nd)
        self.pooled_age = data["Pooled Age"]
        self.pooled_age_se = data["se"]

    def calculate_central_age(self):
        data = calculate_central_age(
                self.Ns, self.Ni, self.zeta,
                self.zeta_error, self.rhod, self.nd
                )
        self.central_age = data["Central"]
        self.central_age_se = data["se"]
        self.central_age_sigma = data["sigma"]

    def calculate_chi_square(self):
        self.chi2 = chi_square(self.Ns, self.Ni)
        return self.chi2
    
    def _repr_html_(self):
        """_repr_html_

        HTML table describing the Sample.
        For integration with Jupyter notebook.
        """
        params = OrderedDict()
        #params["Name"] = self.name
        #params["Depth"] = self.depth
        #params["Elevation"] = self.elevation
        #params["Stratigraphic Age Range Upper/Lower"] = self.stratigraphic_age
        #params["Stratigraphic Age Name"] = self.stratigraphic_age_name
        #params["Deposition Temperature"] = self.deposition_temperature
        #params["Present Day Temperature"] = self.present_day_temperature
        #params["Total Ns"] = sum(self.Ns)
        #params["Total Ni"] = sum(self.Ni)
    
        html = ""

        for key, val in params.items():
            if not val: val = ""
            html += "<div>{0}: {1}</div>".format(key, val)

        return html + DataFrame._repr_html_(self)

    def apply_forward_model(self, fwd_model, name):
        self.kinetic_parameter_type = "ETCH_PIT_LENGTH"
        def func1(row):
            _, ft_age, reduced_density = fwd_model.solve(
                row["l0"],
                self.kinetic_parameter_type,
                row["Dpars"])
            return pd.Series({"ft_age": ft_age, "reduced_density": reduced_density})
        df = self.apply(func1, axis=1)
        self[name] = df["ft_age"]

    def save(self, filename):
        h5store(filename, *self.data_info)

    def radialplot(self, transform="logarithmic"):
        return radialplot(Ns=self.Ns, Ni=self.Ni, zeta=self.zeta, zeta_err=self.zeta_error,
                          rhod=self.rhod, transform=transform)

    


