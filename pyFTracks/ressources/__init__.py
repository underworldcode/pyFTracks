import pandas as pd
from pathlib import Path

#Miller1995 = pd.read_hdf((Path(__file__).parent / "Miller1995.h5"), "data")
#Gleadow = pd.read_hdf((Path(__file__).parent / "Gleadow.h5"), "data")

from pyFTracks import Sample

Miller1995 = Sample().read_from_hdf5(Path(__file__).parent / "Miller.h5")
Gleadow = Sample().read_from_hdf5(Path(__file__).parent / "Gleadow.h5")
