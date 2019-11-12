from itertools import count
import numpy as np
from collections import OrderedDict
from pandas import Series, DataFrame
from pyRadialPlot import radialplot
import pandas as pd
from .utilities import read_mtx_file, h5load, h5store
from .age_calculations import calculate_ages
from .age_calculations import calculate_pooled_age
from .age_calculations import calculate_central_age

unprojected_coefs = {"ETCH_PIT_LENGTH": {"m": 0.283, "b": 15.63},
               "CL_PFU": {"m": 0.544, "b": 16.18},
               "OH_PFU": {"m": 0.0, "b": 16.18},
               "CL_WT_PCT": {"m": 0.13824, "b": 16.288}}

projected_coefs = {"ETCH_PIT_LENGTH": {"m": 0.205, "b": 16.10},
             "CL_PFU": {"m": 0.407, "b": 16.49},
             "OH_PFU": {"m": 0.000, "b": 16.57},
             "CL_WT_PCT": {"m": 0.17317, "b": 16.495}}


class Grain(Series):

    def __init__(self, *args, **kwargs):
        Series.__init__(self, *args, **kwargs)

    @property
    def _constructor(self):
        return Grain

    @property
    def _constructor_expanddim(self):
        return Sample

    
class Sample(DataFrame):

    # Let pandas know what properties are added
    _metadata = ['name', 'zeta', 'zeta_error', 
            'pooled_age', 'pooled_age_se', 'central_age',
            'central_age_se', 'central_age_se',
            'rhod', 'nd', 'depth', 'elevation',
            'stratigraphic_age', 'stratigraphic_age_name',
            'deposition_temperature', 'present_day_temperature', 'id']

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


    def read_from_radialplotter(self, filename):
        from pyRadialPlot import read_radialplotter_file
        data = read_radialplotter_file(filename)
         
        self.__init__({"Ns": data["Ns"], "Ni": data["Ni"]})
        self.zeta = data["zeta"]
        self.rhod = data["rhod"]
    
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

    def save(self, filename):
        h5store(filename, *self.data_info)

    def radialplot(self, transform="Logarithmic"):
        return radialplot(self.Ns, self.Ni, self.zeta, self.rhod, transform=transform)

    


