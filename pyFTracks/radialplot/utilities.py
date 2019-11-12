
import csv

def fission_track_parser(filename):
    """ Parser P. Vermeesh RadialPlotter csv file

        returns: Ns, Ni, dpars as python lists"""
    
    with open(filename, "r") as f:
        file = csv.reader(f)
        name = next(file)
        line = next(file)
        zeta, zeta_err = float(line[0]), float(line[1])
        line = next(file)
        rhod, rhod_err = float(line[0]), int(float(line[1]))
        
        Ns = []
        Ni = []
        dpars = []
        
        for line in file:
            Ns.append(int(line[0]))
            Ni.append(int(line[1]))
            if len(line) > 2:
                dpars.append(float(line[2]))

    return {"Ns": Ns,
            "Ni": Ni,
            "zeta": zeta,
            "zeta_err": zeta_err,
            "rhod": rhod,
            "rhod_err": rhod_err,
            "dpars": dpars}

def read_radialplotter_file(filename):
    """ Parser P. Vermeesh RadialPlotter csv file"""

    with open(filename, "r") as f:
        file = csv.reader(f)
        name = next(file)

    if name[1] == "F":
        return fission_track_parser(filename)
    else:
        return generic_parser(filename)

def generic_parser(filename):
    """ Parser P. Vermeesh RadialPlotter csv file"""

    with open(filename, "r") as f:
        file = csv.reader(f)
        name = next(file)
        
        estimates = []
        standard_errors = []
        
        for line in file:
            estimates.append(float(line[0]))
            standard_errors.append(float(line[1]))

    return {"Estimates": estimates,
            "Standard Errors": standard_errors}
