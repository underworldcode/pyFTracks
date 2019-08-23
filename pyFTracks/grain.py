from itertools import count
from collections import OrderedDict
import numpy as np

class Grain(object):

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

    def __init__(self, counts=None, Ns=None, Ni=None,
                 track_lengths=None, name="Undefined", l0=None,
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

        self._name = name
        self.id = next(self._ids)

        if counts:
            self.counts = counts
            self.Ns = counts[0]
            self.Ni = counts[1]

        if Ns and Ni:
            self.Ns = Ns
            self.Ni = Ni

        self.add_length(track_lengths, True)

        if Dpars:
            self.kinetic_parameter_type = "ETCH_PIT_LENGTH"
            self.kinetic_parameter_value = np.mean(Dpars)
        elif Cl:
            self.kinetic_parameter_type = "CL_PFU"
            self.kinetic_parameter_value = np.mean(Cl)

        self.l0_projected_user_defined = False
        self.l0_user_defined = False
        
        if l0 and projected:
            self.l0_projected = l0 
        else:
            self.l0 = l0

        self.description = ""
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
    
    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

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

    @property
    def l0_projected(self):
        if not self.l0_projected_user_defined:
            m = Grain.projected[self.kinetic_parameter_type]["m"]
            b = Grain.projected[self.kinetic_parameter_type]["b"]
            self._l0_projected = m * self.kinetic_parameter_value + b
        return self._l0_projected
    
    @l0_projected.setter
    def l0_projected(self, length):
        if length:
            self.l0_projected_user_defined = True
            self._l0_projected = length
        else:
            self._l0_projected = None
    
    def add_length(self, length, reset=False):
        if reset:
            self.TL = self.track_lengths = []
        if length:
            length = [length] if not isinstance(length, list) else length
            self.TL += length
            self.mean_track_lengths = self.MTL = np.mean(self.TL)
        return

    def _repr_html_(self):
        """_repr_html_

        HTML table describing the Sample.
        For integration with Jupyter notebook.
        """
        params = OrderedDict()
        params["Name"] = self.name
        params["Description"] = self.description
        params["Ns"] = self.Ns
        params["Ni"] = self.Ni
        params["Kinetic Parameter"] = self.kinetic_parameter_type
        params["Kinetic Parameter Value"] = self.kinetic_parameter_value 
    
        header = "<table>"
        footer = "</table>"
        html = ""

        for key, val in params.items():
            html += "<tr><td>{0}</td><td>{1}</td></tr>".format(key, val)

        return header + html + footer

Population = Apatite = Crystal = Grain

