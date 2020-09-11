import pyFTracks as FT
import pytest
import numpy as np

def clean_ketcham_et_al_1999_Table3b(model):
    return {"c0": float(model["C0"].split()[0]),
            "c1": float(model["C1"].split()[0]),
            "c2": float(model["C2"].split()[0]),
            "c3": float(model["C3"].split()[0]),
            "a": float(model["α"]), 
            "b": float(model["β"])}


def test_thermal_history():
    thermal_history = FT.ThermalHistory(name="My Thermal History",
        time=[0., 43., 44., 100.],
        temperature=[283., 283., 403., 403.])
    assert(np.all(thermal_history.input_time == np.array([0.0, 43.0, 44.0, 100.0])))
    assert(np.all(thermal_history.input_temperature == np.array([283.0, 283.0, 403.0, 403.0])))

def test_thermal_history_entered_as_degC():
    thermal_history = FT.ThermalHistory(name="My Thermal History",
        time=[0., 43., 44., 100.],
        temperature=[10., 10., 130., 130.])
    assert(np.all(thermal_history.input_time == np.array([0.0, 43.0, 44.0, 100.0])))
    assert(np.all(thermal_history.input_temperature == np.array([283.15, 283.15, 403.15, 403.15])))

def test_get_isothermal_interval():
    thermal_history = FT.ThermalHistory(name="My Thermal History",
        time=[0., 43., 44., 100.],
        temperature=[283., 283., 403., 403.])
    assert(np.all(np.isclose(thermal_history.time, np.array([
        100., 99., 98., 97.,
        96., 95., 94., 93.,
        92., 91., 90., 89.,
        88., 87., 86., 85.,
        84., 83., 82., 81.,
        80., 79., 78., 77.,
        76., 75., 74., 73.,
        72., 71., 70., 69.,
        68., 67., 66., 65.,
        64., 63., 62., 61.,
        60., 59., 58., 57.,
        56., 55., 54., 53.,
        52., 51., 50., 49.,
        48., 47., 46., 45.,
        44., 43.93333333, 43.86666667, 43.8,
        43.73333333,  43.66666667, 43.6,  43.53333333,
        43.46666667,  43.4, 43.33333333, 43.26666667,
        43.2,  43.13333333, 43.06666667, 43.,
        42., 41., 40., 39.,
        38., 37., 36., 35.,
        34., 33., 32., 31.,
        30., 29., 28., 27.,
        26., 25., 24., 23.,
        22., 21., 20., 19.,
        18., 17., 16., 15.,
        14., 13., 12., 11.,
        10.,  9.,  8.,  7.,
         6.,  5.,  4.,  3.,
         2.,  1.,  0.]))))

def test_isothermal_max_temperature_wolfs():
    from pyFTracks.thermal_history import WOLF1, WOLF2, WOLF3, WOLF4, WOLF5
    thermal_histories = [WOLF1, WOLF2, WOLF3, WOLF4, WOLF5]
    for thermal_history in thermal_histories:
        assert(np.diff(thermal_history.temperature).max() <= 8.0)

def test_isothermal_max_temperature_wolfs2():
    from pyFTracks.thermal_history import WOLF1, WOLF2, WOLF3, WOLF4, WOLF5
    thermal_histories = [WOLF1, WOLF2, WOLF3, WOLF4, WOLF5]
    for thermal_history in thermal_histories:
        thermal_history.get_isothermal_intervals(max_temperature_per_step=3.0)
        assert(np.diff(thermal_history.temperature).max() <= 3.0)

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

def test_ketcham_1999_Dpar_to_rmr0():
    from pyFTracks.annealing import Ketcham1999
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    assert model.convert_Dpar_to_rmr0(1.0) == pytest.approx(0.84)
    assert model.convert_Dpar_to_rmr0(1.74) == pytest.approx(0.84)
    assert model.convert_Dpar_to_rmr0(5.0) == pytest.approx(0.)
    assert model.convert_Dpar_to_rmr0(2.1) == pytest.approx(0.79962206086744075)

def test_ketcham_1999_clapfu_to_rmr0():
    from pyFTracks.annealing import Ketcham1999
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    assert(model.convert_Cl_pfu_to_rmr0(1.0) == pytest.approx(0.))
    assert(model.convert_Cl_pfu_to_rmr0(0.7) == pytest.approx(0.30169548259180623))
    assert(model.convert_Cl_pfu_to_rmr0(0.4) == pytest.approx(0.6288689335789452))

def test_ohapfu_to_rmr0():
    from pyFTracks.annealing import Ketcham1999
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    assert(model.convert_OH_pfu_to_rmr0(0.9) == pytest.approx(0.3171578660452086))
    assert(model.convert_OH_pfu_to_rmr0(0.7) == pytest.approx(0.6712590592085016))
    assert(model.convert_OH_pfu_to_rmr0(0.4) == pytest.approx(0.8263996762391478))

def test_age_wolf1_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import WOLF1
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF1
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(44.9, abs=1.0))

def test_age_wolf2_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import WOLF2
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF2
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(66.5, abs=1.0))

def test_age_wolf3_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import WOLF3
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF3
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(87.9, abs=1.0))

def test_age_wolf4_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import WOLF4
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF4
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(85.8, abs=1.0))

def test_age_wolf5_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import WOLF5
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF5
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(26.0, abs=1.0))

def test_age_vrolij_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import VROLIJ
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = VROLIJ
    _, model_age, reduced = model.calculate_age(16.1)
    assert(model_age == pytest.approx(113.0, abs=1.0))

def test_age_flaxmans_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import FLAXMANS1
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = FLAXMANS1
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(0.04, abs=1.0))

def test_old_age_wolf1_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import WOLF1
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF1
    old, _, _ = model.calculate_age(16.1)
    assert(old == pytest.approx(44.0, abs=1.0))

def test_old_age_wolf2_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import WOLF2
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF2
    old, _, _ = model.calculate_age(16.1)
    assert(old == pytest.approx(79.5, abs=1.0))

def test_old_age_wolf3_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import WOLF3
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF3
    old, _, _ = model.calculate_age(16.1)
    assert(old == pytest.approx(100., abs=1.0))

def test_old_age_wolf4_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import WOLF4
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF4
    old, _,  _ = model.calculate_age(16.1)
    assert(old == pytest.approx(100., abs=1.0))

def test_old_age_wolf5_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import WOLF5
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF5
    old, _, _ = model.calculate_age(16.1)
    assert(old == pytest.approx(100., abs=1.0))

def test_old_age_vrolij_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import VROLIJ
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = VROLIJ
    old, _, _ = model.calculate_age(15.1)
    assert(old == pytest.approx(113., abs=1.0))

def test_old_age_flaxmans_ketcham_1999():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import FLAXMANS1
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = FLAXMANS1
    old, _, _ = model.calculate_age(16.1)
    assert(old == pytest.approx(0.05, abs=1.0))

def test_age_wolf1_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import WOLF1
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF1
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(44.7, abs=1.0))

def test_age_wolf2_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import WOLF2
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF2
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(61.9, abs=1.0))

def test_age_wolf3_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import WOLF3
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF3
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(84.8, abs=1.0))

def test_age_wolf4_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import WOLF4
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF4
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(81.2, abs=1.0))

def test_age_wolf5_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import WOLF5
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF5
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(7.47, abs=1.0))

def test_age_vrolij_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import VROLIJ
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = VROLIJ
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(112.0, abs=1.0))

def test_age_flaxmans_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import FLAXMANS1
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = FLAXMANS1
    _, model_age, _ = model.calculate_age(16.1)
    assert(model_age == pytest.approx(0.03, abs=1.0))

#def test_miller_sample():
#    from pyFTracks.ressources import Miller
#    assert Miller.central_age == pytest.approx(175.5672, rel=0.001)
#    assert Miller.central_age_se == pytest.approx(8.51013)
#    assert Miller.central_age_sigma == pytest.approx(5.1978e-5, rel=1e-7)

def test_old_age_wolf1_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import WOLF1
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF1
    old, _, _ = model.calculate_age(16.1)
    assert(old == pytest.approx(44.0, abs=1.0))

def test_old_wolf3_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import WOLF3
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF3
    old, _, _ = model.calculate_age(16.1)
    assert(old == pytest.approx(100.0, abs=1.0))

def test_old_age_wolf4_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import WOLF4
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF4
    old, _, _ = model.calculate_age(16.1)
    assert(old == pytest.approx(100.0, abs=1.0))

def test_old_age_wolf5_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import WOLF5
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF5
    old, _, _ = model.calculate_age(16.1)
    assert(old == pytest.approx(19.5, abs=1.0))

def test_old_age_vrolij_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import VROLIJ
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = VROLIJ
    old, _, _ = model.calculate_age(16.1)
    assert(old == pytest.approx(113., abs=1.0))

def test_old_age_flaxmans_ketcham_2007():
    from pyFTracks.annealing import Ketcham2007
    from pyFTracks.thermal_history import FLAXMANS1
    model = Ketcham2007({"ETCH_PIT_LENGTH": 1.65})
    model.history = FLAXMANS1
    old, _, _ = model.calculate_age(16.1)
    assert(old == pytest.approx(0.05, abs=1.0))

def test_generate_synthetic_sample_wolf1():
    from pyFTracks.annealing import Ketcham1999
    from pyFTracks.thermal_history import WOLF1
    model = Ketcham1999({"ETCH_PIT_LENGTH": 1.65})
    model.history = WOLF1
    model.calculate_age()
    sample = model.generate_synthetic_sample()
    sample.save("WOLF1.h5")
    assert isinstance(sample, FT.Sample)


def test_ketcham_1999_model_for_reduced_lengths():
    from pyFTracks.annealing import calculate_mean_reduced_length_ketcham1999
    assert calculate_mean_reduced_length_ketcham1999(1.0, False) == pytest.approx(0.995, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.9, False) == pytest.approx(0.865, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.8, False) == pytest.approx(0.705, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.75, False) == pytest.approx(0.613, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.7, False) == pytest.approx(0.514, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.65, False) == pytest.approx(0.408, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.6, False) == pytest.approx(0.294, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.55,  False) == pytest.approx(0.173, abs=0.01)
    assert calculate_mean_reduced_length_ketcham1999(1.0, True) == pytest.approx(0.994, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.9, True) == pytest.approx(0.855, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.8, True) == pytest.approx(0.715, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.75, True) == pytest.approx(0.645, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.7, True) == pytest.approx(0.575, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.65, True) == pytest.approx(0.506, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.6, True) == pytest.approx(0.436, abs=0.001)
    assert calculate_mean_reduced_length_ketcham1999(0.55,  True) == pytest.approx(0.366, abs=0.001)


def test_ketcham_2003_C3_model_for_lengths():
    from pyFTracks.annealing import calculate_mean_length_ketcham2003
    assert calculate_mean_length_ketcham2003(16, False) == pytest.approx(15.47, abs=0.01)
    assert calculate_mean_length_ketcham2003(15, False) == pytest.approx(14.06, abs=0.01)
    assert calculate_mean_length_ketcham2003(14, False) == pytest.approx(12.55, abs=0.01)
    assert calculate_mean_length_ketcham2003(13, False) == pytest.approx(10.93, abs=0.01)
    assert calculate_mean_length_ketcham2003(12, False) == pytest.approx(9.21, abs=0.01)
    assert calculate_mean_length_ketcham2003(11, False) == pytest.approx(7.38, abs=0.01)
    assert calculate_mean_length_ketcham2003(10, False) == pytest.approx(5.44, abs=0.01)
    assert calculate_mean_length_ketcham2003(9.3,  False) == pytest.approx(4.02, abs=0.01)
    assert calculate_mean_length_ketcham2003(16.0, True) == pytest.approx(15.43, abs=0.01)
    assert calculate_mean_length_ketcham2003(15.0, True) == pytest.approx(13.99, abs=0.01)
    assert calculate_mean_length_ketcham2003(14.0, True) == pytest.approx(12.55, abs=0.01)
    assert calculate_mean_length_ketcham2003(13.0, True) == pytest.approx(11.11, abs=0.01)
    assert calculate_mean_length_ketcham2003(12.0, True) == pytest.approx(9.67, abs=0.01)
    assert calculate_mean_length_ketcham2003(11.0, True) == pytest.approx(8.22, abs=0.01)
    assert calculate_mean_length_ketcham2003(10.0, True) == pytest.approx(6.78, abs=0.01)
    assert calculate_mean_length_ketcham2003(9.3,  True) == pytest.approx(5.77, abs=0.01)


def test_ketcham_2003_C3_model_for_reduced_lengths():
    from pyFTracks.annealing import calculate_mean_reduced_length_ketcham2003
    assert calculate_mean_reduced_length_ketcham2003(1.0, False) == pytest.approx(0.997, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.9, False) == pytest.approx(0.856, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.8, False) == pytest.approx(0.696, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.75, False) == pytest.approx(0.610, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.7, False) == pytest.approx(0.520, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.65, False) == pytest.approx(0.425, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.6, False) == pytest.approx(0.325, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.55,  False) == pytest.approx(0.221, abs=0.01)
    assert calculate_mean_reduced_length_ketcham2003(1.0, True) == pytest.approx(0.998, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.9, True) == pytest.approx(0.851, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.8, True) == pytest.approx(0.704, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.75, True) == pytest.approx(0.631, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.7, True) == pytest.approx(0.557, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.65, True) == pytest.approx(0.484, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.6, True) == pytest.approx(0.410, abs=0.001)
    assert calculate_mean_reduced_length_ketcham2003(0.55,  True) == pytest.approx(0.336, abs=0.001)


cooling_100C_per_myr = FT.ThermalHistory(name="100C per Myr Cooling",
                  time=[0., 2.],
                  temperature=[293.15, 493.15])

cooling_10C_per_myr = FT.ThermalHistory(name="10C per Myr Cooling",
                  time=[0., 20.],
                  temperature=[293.15, 493.15])

cooling_1C_per_myr = FT.ThermalHistory(name="1C per Myr Cooling",
                  time=[0., 200.],
                  temperature=[293.15, 493.15])

# def test_FA_apatite_RN_Ketcham_et_al_1999_Total_Annealing_Temperature():
#     from pyFTracks.annealing import FanningLinear
#     from pyFTracks.ressources import Ketcham_et_al_1999_Table_3b as data
#     data = data[(data["Data"] == "lm") & (data["Model"] == "F.A.") & (data["Apatite"] == "RN")]
#     model_parameters = clean_ketcham_et_al_1999_Table3b(data.to_dict(orient="records")[0]) 
#     model = FT.FanningLinear(model_parameters=model_parameters) 
#     model.history = cooling_100C_per_myr
#     model.calculate_age()
#     TA = model.history.get_temperature_at_time(model.oldest_age) - 273.15
#     assert TA == pytest.approx(0, abs=1)

## Crowley Fluoro0Apatite


def test_age_wolf1_crowley_1990():
    model_age = 51.2
    assert(model_age == pytest.approx(51.2, abs=1.0))

def test_old_age_wolf1_crowley_1990():
    old_age = 61.0
    assert(old_age == pytest.approx(61.0, abs=1.0))

def test_track_length_wolf1_crowley_1990():
    l0 = 16.3 #constant
    track_length = 14.31
    track_length_error = 2.39 
    assert(track_length == pytest.approx(14.31, abs=0.1))

def test_age_wolf2_crowley_1990():
    model_age = 88.3
    assert(model_age == pytest.approx(88.3, abs=1.0))

def test_old_age_wolf2_crowley_1990():
    old_age = 100.0
    assert(old_age == pytest.approx(100., abs=1.0))

def test_track_length_wolf2_crowley_1990():
    l0 = 16.3 #constant
    track_length = 13.44
    track_length_error = 1.88 
    assert(track_length == pytest.approx(13.44, abs=0.1))

def test_age_wolf3_crowley_1990():
    model_age = 94.6
    assert(model_age == pytest.approx(94.6, abs=1.0))

def test_old_age_wolf3_crowley_1990():
    old_age = 100.0
    assert(old_age == pytest.approx(100., abs=1.0))

def test_track_length_wolf3_crowley_1990():
    l0 = 16.3 #constant
    track_length = 13.92
    track_length_error = 1.26 
    assert(track_length == pytest.approx(13.92, abs=0.1))

def test_age_wolf4_crowley_1990():
    model_age = 93.6
    assert(model_age == pytest.approx(93.6, abs=1.0))

def test_old_age_wolf4_crowley_1990():
    old_age = 100.0
    assert(old_age == pytest.approx(100., abs=1.0))

def test_track_length_wolf4_crowley_1990():
    l0 = 16.3 #constant
    track_length = 13.79
    track_length_error = 1.27 
    assert(track_length == pytest.approx(13.79, abs=0.1))

def test_age_wolf5_crowley_1990():
    model_age = 82.1
    assert(model_age == pytest.approx(82.1, abs=1.0))

def test_old_age_wolf5_crowley_1990():
    old_age = 100.0
    assert(old_age == pytest.approx(100.0, abs=1.0))

def test_track_length_wolf5_crowley_1990():
    l0 = 16.3 #constant
    track_length = 12.20
    track_length_error = 1.37 
    assert(track_length == pytest.approx(12.20, abs=0.1))

def test_age_flaxmans_crowley_1990():
    model_age = 30.5
    assert(model_age == pytest.approx(30.5, abs=1.0))

def test_old_age_flaxmans_crowley_1990():
    old_age = 110.0
    assert(old_age == pytest.approx(110., abs=1.0))

def test_track_length_flaxmans_crowley_1990():
    l0 = 16.3 #constant
    track_length = 8.09
    track_length_error = 2.50 
    assert(track_length == pytest.approx(8.09, abs=0.1))

def test_age_vrolij_crowley_1990():
    model_age = 115
    assert(model_age == pytest.approx(115, abs=1.0))

def test_old_age_vrolij_crowley_1990():
    old_age = 113
    assert(old_age == pytest.approx(113., abs=1.0))

def test_track_length_vrolij_crowley_1990():
    l0 = 16.3 #constant
    track_length = 14.87
    track_length_error = 1.08 
    assert(track_length == pytest.approx(14.87, abs=0.1))


# Laslett 1987 Durango


def test_age_wolf1_laslett_1987():
    model_age = 46.9
    assert(model_age == pytest.approx(46.9, abs=1.0))

def test_old_age_wolf1_laslett_1987():
    old_age = 44.0
    assert(old_age == pytest.approx(44.0, abs=1.0))

def test_track_length_wolf1_laslett_1987():
    l0 = 16.3 #constant
    track_length = 15.57
    track_length_error = 1.14 
    assert(track_length == pytest.approx(15.57, abs=0.1))

def test_age_wolf2_laslett_1987():
    model_age = 75.1
    assert(model_age == pytest.approx(75.1, abs=1.0))

def test_old_age_wolf2_laslett_1987():
    old_age = 87.0
    assert(old_age == pytest.approx(87., abs=1.0))

def test_track_length_wolf2_laslett_1987():
    l0 = 16.3 #constant
    track_length = 13.71
    track_length_error = 2.24 
    assert(track_length == pytest.approx(13.71, abs=0.1))

def test_age_wolf3_laslett_1987():
    model_age = 93.5
    assert(model_age == pytest.approx(93.5, abs=1.0))

def test_old_age_wolf3_laslett_1987():
    old_age = 100.0
    assert(old_age == pytest.approx(100., abs=1.0))

def test_track_length_wolf3_laslett_1987():
    l0 = 16.3 #constant
    track_length = 13.83
    track_length_error = 1.52 
    assert(track_length == pytest.approx(13.83, abs=0.1))

def test_age_wolf4_laslett_1987():
    model_age = 85.8
    assert(model_age == pytest.approx(85.8, abs=1.0))

def test_old_age_wolf4_laslett_1987():
    old_age = 100.0
    assert(old_age == pytest.approx(100., abs=1.0))

def test_track_length_wolf4_laslett_1987():
    l0 = 16.3 #constant
    track_length = 12.88
    track_length_error = 1.63 
    assert(track_length == pytest.approx(12.88, abs=0.1))

def test_age_wolf5_laslett_1987():
    model_age = 56.1
    assert(model_age == pytest.approx(56.1, abs=1.0))

def test_old_age_wolf5_laslett_1987():
    old_age = 100.0
    assert(old_age == pytest.approx(100.0, abs=1.0))

def test_track_length_wolf5_laslett_1987():
    l0 = 16.3 #constant
    track_length = 10.33
    track_length_error = 2.17 
    assert(track_length == pytest.approx(10.33, abs=0.1))

def test_age_flaxmans_laslett_1987():
    model_age = 0.30
    assert(model_age == pytest.approx(0.30, abs=1.0))

def test_old_age_flaxmans_laslett_1987():
    old_age = 0.60
    assert(old_age == pytest.approx(0.60, abs=1.0))

def test_track_length_flaxmans_laslett_1987():
    l0 = 16.3 #constant
    track_length = 8.49
    track_length_error = 2.47 
    assert(track_length == pytest.approx(8.49, abs=0.1))

def test_age_vrolij_laslett_1987():
    model_age = 119
    assert(model_age == pytest.approx(119, abs=1.0))

def test_old_age_vrolij_laslett_1987():
    old_age = 113
    assert(old_age == pytest.approx(113., abs=1.0))

def test_track_length_vrolij_laslett_1987():
    l0 = 16.3 #constant
    track_length = 15.38
    track_length_error = 1.10
    assert(track_length == pytest.approx(15.38, abs=0.1))