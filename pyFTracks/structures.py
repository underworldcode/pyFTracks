from itertools import count
import numpy as np
from collections import OrderedDict
from pandas import Series, DataFrame
import pandas as pd
from .utilities import read_mtx_file, h5load, h5store

class Grain(Series):

    # The Following parameters are taken from Carlson et al 1999
    # Equations 1..4
    unprojected = {"ETCH_PIT_LENGTH": {"m": 0.283, "b": 15.63},
                   "CL_PFU": {"m": 0.544, "b": 16.18},
                   "OH_PFU": {"m": 0.0, "b": 16.18},
                   "CL_WT_PCT": {"m": 0.13824, "b": 16.288}}

    projected = {"ETCH_PIT_LENGTH": {"m": 0.205, "b": 16.10},
                 "CL_PFU": {"m": 0.407, "b": 16.49},
                 "OH_PFU": {"m": 0.000, "b": 16.57},
                 "CL_WT_PCT": {"m": 0.17317, "b": 16.495}}

    _ids = count(0)
    
    _added_properties = ['id', 'l0_user_defined', 'l0', '_l0']
    _internal_names = Series._internal_names + _added_properties
    _internal_names_set = set(_internal_names)
    
    def __init__(self, *args, **kwargs):

        self.id = next(self._ids)
        Series.__init__(self, *args, **kwargs)

    @property
    def _constructor(self):
        return Grain

    @property
    def _constructor_expanddim(self):
        return Sample

    @property
    def l0(self):
        if not self.l0_user_defined:
            m = Grain.unprojected[self.kinetic_parameter_type]["m"]
            b = Grain.unprojected[self.kinetic_parameter_type]["b"]
            self._l0 = m * self.kinetic_parameter_value + b
        return self._l0

    @l0.setter
    def l0(self, length):
        if length:
            self.l0_user_defined = True
            self._l0 = length
        else:
            self._l0 = None
#
#    @property
#    def l0_projected(self):
#        if not self.l0_projected_user_defined:
#            m = Grain.projected[self.kinetic_parameter_type]["m"]
#            b = Grain.projected[self.kinetic_parameter_type]["b"]
#            self._l0_projected = m * self.kinetic_parameter_value + b
#        return self._l0_projected
#    
#    @l0_projected.setter
#    def l0_projected(self, length):
#        if length:
#            self.l0_projected_user_defined = True
#            self._l0_projected = length
#        else:
#            self._l0_projected = None
    
Population = Apatite = Crystal = Grain


class Sample(DataFrame):

    _ids = count(0)

    # Let pandas know what properties are added
    _added_properties = ['name', 'zeta', 'zeta_error', 
            'rhod', 'nd', 'depth', 'elevation',
            'stratigraphic_age', 'stratigraphic_age_name',
            'deposition_temperature', 'present_day_temperature', 'id']
    _internal_names = DataFrame._internal_names + _added_properties
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        return Sample

    @property
    def _constructor_sliced(self):
        return Grain

    def __init__(self,
                 file=None,
                 name=None, 
                 zeta=None,
                 zeta_error=None,
                 rhod=None,
                 nd=None,
                 depth=None, 
                 elevation=None,
                 stratigraphic_age=None, 
                 stratigraphic_age_name=None, 
                 deposition_temperature=None,
                 present_day_temperature=None,
                 grains=None,
                 counts=None,
                 Ns=None,
                 Ni=None, A=None, track_lengths=None, l0=None,
                 Dpars=None, Cl=None, projected=True,
                 description=None
                 ):

        self.id = next(self._ids)

        if file:
            with pd.HDFStore(file) as store:
                data, metadata = h5load(store)
            for val in self._added_properties:
                try:
                    key = val.replace("_", " ")
                    setattr(self, val, metadata.pop(key))
                except:
                    pass
            DataFrame.__init__(self, data)
        else:
            self.name = name
            self.zeta = zeta
            self.zeta_error = zeta_error
            self.rhod = rhod
            self.nd = nd
            self.depth = depth
            self.elevation = elevation
            self.stratigraphic_age = stratigraphic_age
            self.stratigraphic_age_name = stratigraphic_age_name
            self.deposition_temperature = deposition_temperature
            self.present_day_temperature = present_day_temperature
            DataFrame.__init__(self)

    def _repr_html_(self):
        """_repr_html_

        HTML table describing the Sample.
        For integration with Jupyter notebook.
        """
        params = OrderedDict()
        params["Name"] = self.name
        params["Depth"] = self.depth
        params["Elevation"] = self.elevation
        params["Stratigraphic Age Range Upper/Lower"] = self.stratigraphic_age
        params["Stratigraphic Age Name"] = self.stratigraphic_age_name
        params["Deposition Temperature"] = self.deposition_temperature
        params["Present Day Temperature"] = self.present_day_temperature
        params["Total Ns"] = sum(self.Ns)
        params["Total Ni"] = sum(self.Ni)
    
        html = ""

        for key, val in params.items():
            if not val: val = ""
            html += "<div>{0}: {1}</div>".format(key, val)

        return html + DataFrame._repr_html_(self)

    def save(self, filename):
        h5store(filename, *self.data_info)
    


