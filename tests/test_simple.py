import pyFTracks as FT
import pytest
import numpy as np

def test_import_sample_datasets():
    from pyFTracks.ressources import Miller, Gleadow

def test_calculate_central_age():
    Ns = [31, 19, 56, 67, 88, 6, 18, 40, 36, 54, 35, 52, 51, 47, 27, 36, 64, 68, 61, 30]
    Ni = [41, 22, 63, 71, 90, 7, 14, 41, 49, 79, 52, 76, 74, 66, 39, 44, 86, 90, 91, 41]
    zeta = 350. # megayears * 1e6 * u.cm2
    zeta_err = 10. / 350.
    rhod = 1.304 # 1e6 * u.cm**-2
    rhod_err = 0.
    Nd = 2936
    data = FT.central_age(Ns, Ni, zeta, zeta_err, rhod, Nd)
    assert(data["Central"] == pytest.approx(175.5672, rel=1e-2))
    assert(data["se"] == pytest.approx(8.5101, rel=1e-2))

def test_calculate_pooled_age():
    Ns = [31, 19, 56, 67, 88, 6, 18, 40, 36, 54, 35, 52, 51, 47, 27, 36, 64, 68, 61, 30]
    Ni = [41, 22, 63, 71, 90, 7, 14, 41, 49, 79, 52, 76, 74, 66, 39, 44, 86, 90, 91, 41]
    zeta = 350. # megayears * 1e6 * u.cm2
    zeta_err = 10. / 350.
    rhod = 1.304 # 1e6 * u.cm**-2
    rhod_err = 0.
    Nd = 2936
    data = FT.pooled_age(Ns, Ni, zeta, zeta_err, rhod, Nd)
    assert(data["Pooled Age"] == pytest.approx(175.5672, rel=1e-2))
    assert(data["se"] == pytest.approx(9.8784, rel=1e-2))

def test_calculate_single_grain_ages():
    Ns = [31, 19, 56, 67, 88, 6, 18, 40, 36, 54, 35, 52, 51, 47, 27, 36, 64, 68, 61, 30]
    Ni = [41, 22, 63, 71, 90, 7, 14, 41, 49, 79, 52, 76, 74, 66, 39, 44, 86, 90, 91, 41]
    zeta = 350. # megayears * 1e6 * u.cm2
    zeta_err = 10. / 350.
    rhod = 1.304 # 1e6 * u.cm**-2
    rhod_err = 0.
    Nd = 2936
    data = FT.single_grain_ages(Ns, Ni, zeta, zeta_err, rhod, Nd)
    actual_values = np.array([170.27277726, 194.12922188, 199.71847392, 211.82501094,
        219.35417906, 192.69120201, 286.91906312, 218.87597033,
        165.51402406, 154.12751783, 151.7948727 , 154.27595627,
        155.38512454, 160.49155828, 156.07978109, 184.05633985,
        167.62488355, 170.15231471, 151.18251036, 164.84973261])
    actual_errors = np.array([ 40.93820562,  61.15626671,  37.30358567,  36.79102639,
         33.72043367, 107.40365187, 102.70782874,  49.20922289,
         36.7660512 ,  27.71454167,  33.5872093 ,  28.25634367,
         28.7686159 ,  31.11427488,  39.43447289,  41.83466447,
         28.25353735,  27.94537164,  25.54017193,  40.0012906 ])
    assert np.testing.assert_allclose(data["Age(s)"], actual_values, rtol=0.01) is None
    assert np.testing.assert_allclose(data["se(s)"], actual_errors, rtol=0.01) is None

def test_chi2_test():
    Ns = [31, 19, 56, 67, 88, 6, 18, 40, 36, 54, 35, 52, 51, 47, 27, 36, 64, 68, 61, 30]
    Ni = [41, 22, 63, 71, 90, 7, 14, 41, 49, 79, 52, 76, 74, 66, 39, 44, 86, 90, 91, 41]
    chi2 = FT.chi2_test(Ns, Ni)
    assert chi2 == pytest.approx(0.9292, rel=1e-3)

#def test_miller_sample():
#    from pyFTracks.ressources import Miller
#    assert Miller.central_age == pytest.approx(175.5672, rel=0.001)
#    assert Miller.central_age_se == pytest.approx(8.51013)
#    assert Miller.central_age_sigma == pytest.approx(5.1978e-5, rel=1e-7)
