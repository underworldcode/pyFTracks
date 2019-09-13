from itertools import count
from collections import OrderedDict
import numpy as np
from .sample import Sample
from pandas import Series

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

    @property
    def _constructor(self):
        return Grain

    @property
    def _constructor_expanddim(self):
        return Sample

    def __init__(self, Ns=None, Ni=None,
                 lengths=None, name="Undefined", l0=None,
                 Dpars=None, Cl=None, projected=True):
        """
          Grain
        
          counts: list of (Ns, Ni) counts.
          Ns: number of spontaneous tracks
          Ni: number of induced tracks
          track_lengths: track length measurements
          Dpars: Dpar values
          Cl: Chlorine content
          name: optional name
          l0: User defined initial track length

        """

        self.id = next(self._ids)
        data = {"Ns": Ns, "Ni": Ni, "lengths": lengths, "Dpars": Dpars}
        Series.__init__(self, data)

        if l0:
            self.l0_user_defined = True
            self.l0 = l0

        if Dpars:
            self.kinetic_parameter_type = "ETCH_PIT_LENGTH"
        elif Cl:
            self.kinetic_parameter_type = "CL_PFU"

    
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

