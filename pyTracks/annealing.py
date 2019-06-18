from collections import namedtuple

# Collection of annealing models

AnnealModel = namedtuple("AnnealModel", "c0 c1 c2 c3 a b lmin")

KetchamEtAl = AnnealModel(
    c0=-19.844,
    c1=0.38951,
    c2=-51.253,
    c3=-7.6423,
    a=-0.12327,
    b=-11.988,
    lmin=0.0)

TILm = AnnealModel(
    c0=-1.66965,
    c1=0.0000241755,
    c2=-12.4864,
    c3=0.000843004,
    a=0.675508,
    b=4.16615,
    lmin=0.0)

TILc = AnnealModel(
    c0=-2.36910,
    c1=0.0000603834,
    c2=-8.65794,
    c3=0.000972676,
    a=0.404700,
    b=1.65355,
    lmin=9.0)

CrowDur = AnnealModel(
    c0=-3.202,
    c1=0.00009367,
    c2=-19.6328,
    c3=0.0004200,
    a=0.49,
    b=3.00,
    lmin=0.0)

CrowFAp = AnnealModel(
    c0=-1.508,
    c1=0.00002076,
    c2=-10.3227,
    c3=0.0009967,
    a=0.76,
    b=4.30,
    lmin=0.0)

CrowSrAp = AnnealModel(
    c0=-1.123,
    c1=0.00001055,
    c2=-5.0085,
    c3=0.001195,
    a=0.97,
    b=4.16,
    lmin=0.0)

LasDur = AnnealModel(
    c0=-4.87,
    c1=0.000168,
    c2=-28.12,
    c3=0.0,
    a=0.35,
    b=2.7,
    lmin=0.0)

