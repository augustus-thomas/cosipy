import numpy as np
from numpy import array_equal as arr_eq
from histpy import Histogram, Axes, Axis
from scoords import SpacecraftFrame
from astropy.coordinates import SkyCoord
import astropy.units as u
import h5py as h5
from astropy.time import Time
from mhealpy import HealpixBase, HealpixMap

from cosipy import test_data
from cosipy.response.FullDetectorResponse import cosi_response
from cosipy.response import PointSourceResponse, FullDetectorResponse

from threeML import DiracDelta, Constant, Line, Quadratic, Cubic, Quartic, Model 
from threeML import StepFunction, StepFunctionUpper, Cosine_Prior, Uniform_prior, PhAbs, Gaussian, TbAbs
from cosipy.threeml.custom_functions import SpecFromDat

import pytest

# init/load
response_path = test_data.path/"test_full_detector_response.h5"

with FullDetectorResponse.open(response_path) as response:
    exposure_map = HealpixMap(base=response,
                                    unit=u.s,
                                    coordsys=SpacecraftFrame())

    ti = Time('1999-01-01T00:00:00.123456789')
    tf = Time('2010-01-01T00:00:00')
    dt = (tf-ti).to(u.s)

    exposure_map[:4] = dt/4
    psr = response.get_point_source_response(exposure_map = exposure_map)

def test_photon_energy_axis():
    assert psr.photon_energy_axis == psr.axes['Ei']

def test_get_expectation():
    # supported spectral functions
    ## see astromodels.functions.function.Function1D.[function_name]()
    ## normalization units make expectation have units of counts
    norm = 1 / (u.keV * u.cm**2 * u.s)
    ## 3 sig-figs of precision
    ## Constant
    const = Constant(k=1e-1)
    const.k.unit = norm
    exp = psr.get_expectation(const)
    assert isinstance(exp, Histogram) == True
    assert 2.19e+11 < np.sum(exp.contents) < 2.20e+11

    ## Line
    line = Line(a=1e-1, b=-4e-5)
    line.a.unit, line.b.unit = norm, norm
    exp = psr.get_expectation(line)
    assert isinstance(exp, Histogram) == True
    assert 2.61e+10 < np.sum(exp.contents) < 2.62e+10
    # spectra = \
    #     [Constant(k=1e-1),
    #     Line(a=1e-1, b=-4e-5),
    #     Quadratic(a=1e-1, b=-4e-5, c=1e-9),
    #     Cubic(a=1e-1, b=-4e-5, c=1e-9, d=-4e-13),
    #     Quartic(a=1e-1, b=-4e-5, c=1e-9, d=-4e-13, e=1e-17),
    #     StepFunction(upper_bound=3e2, lower_bound=0, value=1),
    #     StepFunctionUpper(upper_bound=3e2, lower_bound=0, value=1),
    #     PhAbs(NH=1e-2, redshift=1),
    #     DiracDelta(zero_point=511, value=1000),
    #     Gaussian(F=1000, mu=1e3, sigma=1e1),
    #     Cosine_Prior(upper_bound=1e3, lower_bound=0, value=1),
    #     Uniform_prior(upper_bound=1e3, lower_bound=0, value=1)]
    # for hypo in spectra:
    #     hypo.k.unit = norm
    #     exp = psr.get_expectation(hypo)
    #     assert isinstance(exp, Histogram) == True
    #     print(str(hypo) + ":" + str(np.sum(exp.contents)))
    # # hypo.k.unit = norm
    # exp = psr.get_expectation(hypo)
    # assert isinstance(exp, Histogram) == True
    # sum = np.sum(exp.contents)
    # assert 2.19e11 < sum < 2.20e11
    # assert 631 < np.sum(psr.get_expectation(hypo).contents) < 632
    # assert 75.2 < np.sum(psr.get_expectation(Line(a=1e-1, b=-4e-5, unit=norm)).contents) < 75.3
    # assert 116 < np.sum(psr.get_expectation(Quadratic(a=1e-1, b=-4e-5, c=1e-9, unit=norm)).contents) < 117
    # assert 59.4 < np.sum(psr.get_expectation(Cubic(a=1e-1, b=-4e-5, c=1e-9, d=-4e-13, unit=norm)).contents) < 59.5
    # assert 64.8 < np.sum(psr.get_expectation(Quartic(a=1e-1, b=-4e-5, c=1e-9, d=-4e-13, e=1e-17, unit=norm)).contents) < 64.9
    # assert 39.9 < np.sum(psr.get_expectation(StepFunction(upper_bound=3e2, lower_bound=0, value=1, unit=norm)).contents) < 40.0
    # assert 39.9 < np.sum(psr.get_expectation(StepFunctionUpper(upper_bound=3e2, lower_bound=0, value=1, unit=norm)).contents) < 40.0
    # assert 6310 < np.sum(psr.get_expectation(PhAbs(NH=1e-2, redshift=1, unit=norm)).contents) < 6320
    # assert 1570 < np.sum(psr.get_expectation(DiracDelta(zero_point=511, value=1000, unit=norm)).contents) < 1580
    # assert 2160 < np.sum(psr.get_expectation(Gaussian(F=1000, mu=1e3, sigma=1e1, unit=norm)).contents) < 2170
    # assert 2.220 < np.sum(psr.get_expectation(Cosine_Prior(upper_bound=1e3, lower_bound=0, value=1, unit=norm))) < 2.230
    # assert 1280 < np.sum(psr.get_expectation(Uniform_prior(upper_bound=1e3, lower_bound=0, value=1, unit=norm)).contents) < 1290

    # ## example implicitly supported :py:class:`threeML.Model` with units
    # assert 6290 < np.sum(psr.get_expectation(TbAbs(NH=3, redshift=15)).contents) < 6300

    # ## generic unsupported :py:class:`threeML.Model` without units
    # with pytest.raises(RuntimeError) as pytest_wrapped_exp:
    #     expectation = psr.get_expectation(Model())
    # assert pytest_wrapped_exp.type == RuntimeError