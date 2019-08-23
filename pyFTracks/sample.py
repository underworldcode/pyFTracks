from itertools import count
from collections import OrderedDict
from .grain import Grain
import numpy as np
from .utilities import read_mtx_file

class Sample(Grain):

    _ids = count(0)

    def __init__(self, 
                 name=None, 
                 depth=None, 
                 elevation=None,
                 stratigraphic_age=None, 
                 stratigraphic_age_name=None, 
                 deposition_temperature=None,
                 present_day_temperature=None,
                 grains=None,
                 counts=None,
                 Ns=None,
                 Ni=None, track_lengths=None, l0=None,
                 Dpars=None, Cl=None, projected=True
                 ):

        """
          Sample

          grains: list of grains
          coordinates: coordinates of the sample location
          elevation: elevation of the sample location
          name: sample name
        """

        super(Sample, self).__init__(counts, Ns, Ni, track_lengths,
                                     name, l0, Dpars, Cl, projected)

        self._name = name
        self._id = self._number = next(self._ids)
        self._depth = depth
        self._elevation = elevation

        self._stratigraphic_age = stratigraphic_age
        self._stratigraphic_age_name = stratigraphic_age_name

        self._deposition_temperature = deposition_temperature
        self._present_day_temperature = present_day_temperature

        if grains:
            grains = list(grains)
            for grain in grains:
                self.add_grain(grain)

        self._description = ""
        
        self.add_count(counts, Ns, Ni, True)
        self.add_length(track_lengths, True)

        if Dpars:
            self.kinetic_parameter_type = "ETCH_PIT_LENGTH"
            self.kinetic_parameter_value = np.mean(Dpars)
        elif Cl:
            self.kinetic_parameter_type = "CL_PFU"
            self.kinetic_parameter_value = np.mean(Cl)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def number(self):
        return self._number

    @property
    def depth(self):
        return self._depth
    
    @depth.setter
    def depth(self, depth):
        self._depth = depth

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, elevation):
        self._elevation = elevation

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def stratigraphic_age(self):
        return self._stratigraphic_age

    @stratigraphic_age.setter
    def stratigraphic_age(self, stratigraphic_age):
        stratigraphic_age = tuple(stratigraphic_age)
        if len(stratigraphic_age) != 2 or stratigraphic_age[-1] >= stratigraphic_age[0]:
            raise ValueError("Wrong format input for stratigraphic age: Format is (Upper, Lower)")
        self._stratigraphic_age = stratigraphic_age
    
    @property
    def stratigraphic_age_name(self):
        return self._stratigraphic_age

    @stratigraphic_age.setter
    def stratigraphic_age_name(self, name):
        self._stratigraphic_age_name = name

    @property
    def deposition_temperature(self):
        return self._deposition_temperature

    @deposition_temperature.setter
    def deposition_temperature(self, deposition_temperature):
        self._deposition_temperature = deposition_temperature

    @property
    def present_day_temperature(self):
        return self._present_day_temperature

    @present_day_temperature.setter
    def present_day_temperature(self, present_day_temperature):
        self._present_day_temperature = present_day_temperature

    def add_grain(self, grain):
        if not isinstance(grain, Grain):
            raise ValueError("""You must provide a Grain object""")
        self.grains.append(Grain)
    
    def add_count(self, count=None, Ns=None, Ni=None, reset=False):
        if reset:
            self.Ns = self.spontaneous = []
            self.Ni = self.induced = []
            self.counts = []

        if count:
            count = [count] if not isinstance(count, list) else count
            self.Ns += [val[0] for val in count]
            self.Ni += [val[1] for val in count]
            self.counts += count
            return
        
        if Ns is not None and Ni is not None:
            Ns = [Ns] if not isinstance(Ns, list) else Ns
            Ni = [Ni] if not isinstance(Ni, list) else Ni
            if not len(Ns) == len(Ni):
                raise ValueError("""Ns and Ni do not have the same lengths""")
            self.Ns += Ns
            self.Ni += Ni
            self.counts += zip(self.Ns, self.Ni)
            return

    def _repr_html_(self):
        """_repr_html_

        HTML table describing the Sample.
        For integration with Jupyter notebook.
        """
        params = OrderedDict()
        params["Name"] = self.name
        params["Number"] = self.number
        params["Depth"] = self.depth
        params["Elevation"] = self.elevation
        params["Stratigraphic Age Range Upper/Lower"] = self.stratigraphic_age
        params["Stratigraphic Age Name"] = self.stratigraphic_age_name
        params["Deposition Temperature"] = self.deposition_temperature
        params["Present Day Temperature"] = self.present_day_temperature
        params["Description"] = self.description
        params["Total Ns"] = sum(self.Ns)
        params["Total Ni"] = sum(self.Ni)
    
        header = "<table>"
        footer = "</table>"
        html = ""

        for key, val in params.items():
            html += "<tr><td>{0}</td><td>{1}</td></tr>".format(key, val)

        return header + html + footer

    def save(self, filename, zeta=384.1, rhod=9000):
    
        f = open(filename, "w")
        f.write("{name:s}\n".format(name=self.name))
        f.write("{value:s}\n".format(value=str(-999)))
        f.write("{nconstraints:d} {ntl:d} {nc:d} {zeta:5.1f} {rhod:12.1f} {totco:d}\n".format(
                 nconstraints=0, ntl=len(self.TL), nc=len(self.Ns), zeta=zeta, rhod=rhod,
                 totco=2000))
        f.write("{age:5.1f} {age_error:5.1f}\n".format(age=self.FTage,
                                                       age_error=FTage_error))
        TLmean = (float(sum(TL))/len(TL) if len(TL) > 0 else float('nan'))
        TLmean_sd = np.std(TL)
    
        f.write("{mtl:5.1f} {mtl_error:5.1f}\n".format(mtl=TLmean,
                                                       mtl_error=TLmean*0.05))
        f.write("{mtl_std:5.1f} {mtl_std_error:5.1f}\n".format(mtl_std=TLmean_sd,
                                                               mtl_std_error=TLmean_sd*0.05))
        for i in range(NS.size):
            f.write("{ns:d} {ni:d}\n".format(ns=NS[i], ni=NI[i]))
    
        for track in TL:
            f.write("{tl:4.1f}\n".format(tl=track))
    
        f.close()
        return 0

    def read_from_file(self, filename):
        data = read_mtx_file(filename)
        self.Ns = data["Ns"]


