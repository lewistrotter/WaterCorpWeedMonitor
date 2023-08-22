
import os
import numpy as np
import xarray as xr

from scipy.stats import kendalltau

from scripts import shared


# deprecated
def get_opcs(n):
    if n == 3:
        slope = [-1, 0, 1]
        lin_ss, lin_c = 2, 1
        curve = [1, -2, 1]
        crv_ss, crv_c = 6, 3
    elif n == 4:
        slope = [-3, -1, 1, 3]
        lin_ss, lin_c = 20, 2
        curve = [1, -1, -1, 1]
        crv_ss, crv_c = 4, 1
    elif n == 5:
        slope = [-2, -1, 0, 1, 2]
        lin_ss, lin_c = 10, 1
        curve = [2, -1, -2, -1, 2]
        crv_ss, crv_c = 14, 1
    elif n == 6:
        slope = [-5, -3, -1, 1, 3, 5]
        lin_ss, lin_c = 70, 2
        curve = [5, -1, -4, -4, -1, 5]
        crv_ss, crv_c = 84, 1.5
    elif n == 7:
        slope = [-3, -2, -1, 0, 1, 2, 3]
        lin_ss, lin_c = 28, 1
        curve = [5, 0, -3, -4, -3, 0, 5]
        crv_ss, crv_c = 84, 1
    elif n == 8:
        slope = [-7, -5, -3, -1, 1, 3, 5, 7]
        lin_ss, lin_c = 168, 2
        curve = [7, 1, -3, -5, -5, -3, 1, 7]
        crv_ss, crv_c = 168, 1
    elif n == 9:
        slope = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        lin_ss, lin_c = 60, 1
        curve = [28, 7, -8, -17, -20, -17, -8, 7, 28]
        crv_ss, crv_c = 2772, 3
    elif n == 10:
        slope = [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]
        lin_ss, lin_c = 330, 2
        curve = [6, 2, -1, -3, -4, -4, -3, -1, 2, 6]
        crv_ss, crv_c = 132, 0.5
    elif n == 11:
        slope = [-5 - 4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        lin_ss, lin_c = 110, 1
        curve = [15, 6, -1, -6, -9, -10, -9, -6, -1, 6, 15]
        crv_ss, crv_c = 858, 1
    elif n == 12:
        slope = [-11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11]
        lin_ss, lin_c = 572, 2
        curve = [55, 25, 1, -17, -29, -35, -35, -29, -17, 1, 25, 55]
        crv_ss, crv_c = 12012, 3
    else:
        raise ValueError('Number not supported.')

    return slope, lin_ss, lin_c, curve, crv_ss, crv_c


# deprecated
def apply_fit(vec, coeffs, ss, c):
    return np.sum((vec * coeffs)) / ss * c


# deprecated
def generate_trend_rgb_xr(
        ds: xr.Dataset,
        var_name: str
) -> xr.Dataset:

    # extract netcdf attributes
    ds_attrs = ds.attrs
    ds_band_attrs = ds[list(ds)[0]].attrs
    ds_spatial_ref_attrs = ds['spatial_ref'].attrs

    # subset xr by given var
    ds_tmp = ds[[var_name]]

    # get number of years from input xr
    num_years = len(np.unique(ds_tmp['time.year']))

    try:
        # get opcs for linear and curvature fits and create keyword args
        lin_coeffs, lin_ss, lin_c, crv_coeffs, crv_ss, crv_c = get_opcs(num_years)

        # get linear slope per pixel vector with linear coeffs
        kwargs = {'coeffs': lin_coeffs, 'ss': lin_ss, 'c': lin_c}
        da_slp = xr.apply_ufunc(apply_fit,
                                ds_tmp,
                                input_core_dims=[['time']],
                                vectorize=True,
                                kwargs=kwargs)

        # get curvature per pixel vector with curve coeffs
        kwargs = {'coeffs': crv_coeffs, 'ss': crv_ss, 'c': crv_c}
        da_crv = xr.apply_ufunc(apply_fit,
                                ds_tmp,
                                input_core_dims=[['time']],
                                vectorize=True,
                                kwargs=kwargs)

        # modify to highlight inc, dec and recovery trends
        da_inc = (da_slp > 0) * np.abs(da_slp)  # # was <
        da_dec = (da_slp < 0) * np.abs(da_slp)  # pos values are decreasing trend # was >
        da_rec = (da_crv > 0) * np.abs(da_crv)  # recent recovery has neg curve # was <

        # standardise to 0-255
        da_inc = (da_inc / da_inc.max() * 255)
        da_dec = (da_dec / da_dec.max() * 255)
        da_rec = (da_rec / da_rec.max() * 255)

        # rename variables
        da_inc = da_inc.rename({var_name: 'inc'})
        da_dec = da_dec.rename({var_name: 'dec'})
        da_rec = da_rec.rename({var_name: 'rec'})

        # combine into one
        ds_tnd = xr.merge([da_dec, da_rec, da_inc])

        # add attributes back on
        ds_tnd.attrs = ds_attrs
        ds_tnd['spatial_ref'].attrs = ds_spatial_ref_attrs
        for tnd_var in ds_tnd:
            ds_tnd[tnd_var].attrs = ds_band_attrs

    except Exception as e:
        raise e

    return ds_tnd
