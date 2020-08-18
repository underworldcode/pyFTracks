from itertools import count
import numpy as np
from collections import OrderedDict
from pandas import DataFrame, Series
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

    _metadata = ['_track_lengths', 'track_lengths']

    def __init__(self, *args, **kwargs):
        Series.__init__(self, *args, **kwargs)

    @property
    def _constructor(self):
        return Grain

    @property
    def _constructor_expanddim(self):
        return Sample

    @property
    def zeta(self):
        return self._zeta

    @zeta.setter
    def zeta(self, value):
        self._zeta = value
    
    
class Sample(DataFrame):

    # Let pandas know what properties are added
    _metadata = ['name', 'zeta', 'zeta_error', 
            'pooled_age', 'pooled_age_se', 'central_age',
            'central_age_se', 'central_age_se',
            'rhod', 'nd', 'depth', 'elevation',
            'stratigraphic_age', 'stratigraphic_age_name',
            'unit_area_graticule', 'description',
            'deposition_temperature', 'present_day_temperature', 'id',
            '_track_lengths', 'track_lengths']
            
    def __init__(self, data=None, central_age=None, pooled_age=None, zeta=None, zeta_error=None, rhod=None, nd=None, name: str=None,
                 elevation=None, depth=None, stratigraphic_age=None, stratigraphic_age_name:str=None,
                 description=None, deposition_temperature=None,
                 present_day_temperature=None, *args, **kwargs):

        self.name = name
        self.depth = depth
        self.elevation = elevation
        self.stratigraphic_age = stratigraphic_age
        self.stratigraphic_age_name = stratigraphic_age_name
        self.deposition_temperature = deposition_temperature
        self.present_day_temperature = present_day_temperature
        self.description = description
        self.zeta = zeta
        self.zeta_error = zeta_error
        self.rhod = rhod
        self.nd = nd
        self.central_age = central_age
        self.pooled_age = pooled_age


        if isinstance(data, DataFrame):
            data = data.to_dict() 
       
        super(Sample, self).__init__(columns=["Ns", "Ni", "A"], data=data, *args, **kwargs)
        self._track_lengths = None

    @property
    def _constructor(self):
        return DataFrame

    @property
    def _constructor_sliced(self):
        return Grain

    def read_from_hdf5(self, filename):
        with pd.HDFStore(filename) as store:
            data, metadata = h5load(store)
        for val in self._metadata:
            try:
                setattr(self, val, metadata.pop(val))
            except:
                pass

        try:
            self.calculate_ages()
            self.calculate_ratios()
        except:
            pass


        if not self.central_age:
            try:
                self.calculate_central_age()
            except:
                pass

        if not self.pooled_age:
            try:
                self.calculate_pooled_age()
            except:
                pass

        super(Sample, self).__init__(data=data)
        return self

    def calculate_ratios(self):
        if not hasattr(self, "Ns"):
            raise ValueError("Cannot find Ns counts")
        if not hasattr(self, "Ni"):
            raise ValueError("Cannot find Ns counts")

        self["Ns/Ni"] = self.Ns / self.Ni
        if not hasattr(self, "unit_area_graticule"):
            self.unit_area_graticule = 1.0
        if not hasattr(self, "A"):
            self.A = 1
        self["RhoS"] = self.Ns / (self.A * self.unit_area_graticule)
        self["RhoI"] = self.Ni / (self.A * self.unit_area_graticule)
        return self

    def read_from_radialplotter(self, filename):
        from pyRadialPlot import read_radialplotter_file
        data = read_radialplotter_file(filename)
         
        self.__init__({"Ns": data["Ns"], "Ni": data["Ni"]})
        self.zeta = data["zeta"]
        self.rhod = data["rhod"]
        self._calculate_statistics()
    
    def calculate_l0_from_Dpars(self, projected=True):
        if not hasattr(self, "Dpars"):
            raise ValueError("Cannot find Dpars column")
        if projected:
            m = projected_coefs["ETCH_PIT_LENGTH"]["m"]
            b = projected_coefs["ETCH_PIT_LENGTH"]["b"]
        else:
            m = unprojected_coefs["ETCH_PIT_LENGTH"]["m"]
            b = unprojected_coefs["ETCH_PIT_LENGTH"]["b"]
        self["l0"] = m * self.Dpars + b

    def calculate_ages(self):
        required = ["Ns", "Ni", "zeta", "zeta_error", "rhod", "nd"]
        for arg in required:
            if arg is None:
                raise ValueError("""Cannot find {0}""".format(arg))

        data =  calculate_ages(
                self.Ns, self.Ni, self.zeta, 
                self.zeta_error, self.rhod, self.nd)
        self["Ages"] = data["Age(s)"]
        self["Ages Errors"] = data["se(s)"]
        return {"Ages": list(self["Ages"]), 
                "Ages Errors": list(self["Ages Errors"])}

    @property
    def track_lengths(self):
        return self._track_lengths

    @track_lengths.setter
    def track_lengths(self, values):
        self._track_lengths = values

    def calculate_pooled_age(self):
        required = ["Ns", "Ni", "zeta", "zeta_error", "rhod", "nd"]
        for arg in required:
            if arg is None:
                raise ValueError("""Cannot find {0}""".format(arg))
        data = calculate_pooled_age(
                self.Ns, self.Ni, self.zeta,
                self.zeta_error, self.rhod, self.nd)
        self.pooled_age = data["Pooled Age"]
        self.pooled_age_se = data["se"]
        return {"Pooled Age": self.pooled_age,
                "se": self.pooled_age_se}

    def calculate_central_age(self):
        required = ["Ns", "Ni", "zeta", "zeta_error", "rhod", "nd"]
        for arg in required:
            if arg is None:
                raise ValueError("""Cannot find {0}""".format(arg))
        data = calculate_central_age(
                self.Ns, self.Ni, self.zeta,
                self.zeta_error, self.rhod, self.nd
                )
        self.central_age = data["Central"]
        self.central_age_se = data["se"]
        self.central_age_sigma = data["sigma"]
        return {"Central": self.central_age,
                "se": self.central_age_se,
                "sigma": self.central_age_sigma}

    def calculate_chi_square(self):
        self.chi2 = chi_square(self.Ns, self.Ni)
        return self.chi2
    
    def _repr_html_(self):
        """_repr_html_

        HTML table describing the Sample.
        For integration with Jupyter notebook.
        """
        params = OrderedDict()
        params["Name"] = self.name
        params["Description"] = self.description
        params["Depth"] = self.depth
        params["Elevation"] = self.elevation
        params["Stratigraphic Age Range Upper/Lower"] = self.stratigraphic_age
        params["Stratigraphic Age Name"] = self.stratigraphic_age_name
        params["Deposition Temperature"] = self.deposition_temperature
        params["Present Day Temperature"] = self.present_day_temperature
        params["Total Ns"] = sum(self.Ns)
        params["Total Ni"] = sum(self.Ni)
        params["rhoD"] = self.rhod
        params["nd"] = self.nd
        params["Zeta"] = f"{self.zeta} ({self.zeta_error})"
    
        #html = "<div style='margin-bottom:0.5cm;margin.top:0.5cm;margin-left:4cm;font-size:large;font-weight:bold'>Metadata</div>"
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
        data = pd.DataFrame()
        data["Ns"] = self.Ns
        data["Ni"] = self.Ni
        data["A"] = self.A
        metadata = {}
        for val in self._metadata:
            if not val.startswith("_"):
                try:
                    metadata[val] = getattr(self, val)
                except:
                    pass
        h5store(filename, data, **metadata)
    

    save_to_hdf = save

    def radialplot(self, transform="logarithmic"):
        return radialplot(Ns=self.Ns, Ni=self.Ni, zeta=self.zeta, zeta_err=self.zeta_error,
                          rhod=self.rhod, transform=transform)

    


