import pandas as pd
from pathlib import Path

Miller1995 = pd.read_hdf((Path(__file__).parent / "Miller1995.h5"), "data")
