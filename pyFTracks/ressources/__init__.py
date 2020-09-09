import pandas as pd
from pathlib import Path
from pyFTracks import Sample

Miller = Sample().read_from_hdf5(Path(__file__).parent / "Miller.h5")
Gleadow = Sample().read_from_hdf5(Path(__file__).parent / "Gleadow.h5")

#  Read a bunch of tables from the literature.

Ketcham_et_al_1999_Table_3a = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_1999_Table3a.csv")
Ketcham_et_al_1999_Table_3b = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_1999_Table3b.csv")
Ketcham_et_al_1999_Table_4a = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_1999_Table4a.csv")
Ketcham_et_al_1999_Table_4b = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_1999_Table4b.csv")
Ketcham_et_al_1999_Table_5a = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_1999_Table5a.csv")
Ketcham_et_al_1999_Table_5b = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_1999_Table5b.csv")
Ketcham_et_al_1999_Table_5c = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_1999_Table5c.csv")
Ketcham_et_al_1999_Table_5d = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_1999_Table5d.csv")

Ketcham_et_al_2007_Table_1a = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_2007_Table1a.csv")
Ketcham_et_al_2007_Table_1b = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_2007_Table1b.csv")
Ketcham_et_al_2007_Table_2a = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_2007_Table2a.csv")
Ketcham_et_al_2007_Table_2b = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_2007_Table2b.csv")
Ketcham_et_al_2007_Table_3a = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_2007_Table3a.csv")
Ketcham_et_al_2007_Table_3b = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_2007_Table3b.csv")
Ketcham_et_al_2007_Table_5a = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_2007_Table5a.csv")
Ketcham_et_al_2007_Table_5b = pd.read_csv(
    Path(__file__).parent / "Ketcham_et_al_2007_Table5b.csv")
