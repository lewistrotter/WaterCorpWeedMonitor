
import os
import json
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import arcpy

from osgeo import gdal
#from scipy.stats import zscore


from scripts import shared


def execute(
        parameters
        # messages # TODO: implement
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    #in_project_file = parameters[0].valueAsText
    # start date

    # TODO: uncomment these when testing
    in_project_file = r'C:\Users\Lewis\Desktop\testing\project_1\meta.json'

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE ENVIRONMENT

    arcpy.SetProgressor('default', 'Preparing environment...')

    if arcpy.CheckExtension('Spatial') != 'Available':
        arcpy.AddError('Spatial Analyst license is unavailable.')
        raise  # return
    else:
        arcpy.CheckOutExtension('Spatial')

    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = False

    # TODO: enable if move to non-memory temp files
    # TODO: ensure all the input/out pathes changed too if do this
    #tmp_folder = os.path.join(in_project_file, 'tmp')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK PROJECT FOLDER STRUCTURE AND FILES

    arcpy.SetProgressor('default', 'Checking project folders...')

    if not os.path.exists(in_project_file):
        arcpy.AddError('Project file does not exist.')
        raise  # return

    in_project_folder = os.path.dirname(in_project_file)

    sub_folders = ['grid', 'uav_captures', 'sat_captures', 'tmp', 'visualise']
    for sub_folder in sub_folders:
        sub_folder = os.path.join(in_project_folder, sub_folder)
        if not os.path.exists(sub_folder):
            arcpy.AddError('Project folder is missing expected folders.')
            raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ AND CHECK METADATA

    arcpy.SetProgressor('default', 'Reading and checking metadata...')

    with open(in_project_file, 'r') as fp:
        meta = json.load(fp)

    if len(meta['data']) == 0:
        arcpy.AddError('Project capture data does not exist.')
        raise  # return

    classed_meta_items = []
    for item in meta['data']:
        if item['classified']:
            classed_meta_items.append(item['capture_date'])

    if len(classed_meta_items) == 0:
        arcpy.AddError('Could not find classified capture data in metadata file.')
        raise  # return

    latest_classed_date = sorted(classed_meta_items)[-1]

    meta_item = None
    for item in meta['data']:
        if item['capture_date'] == latest_classed_date:
            meta_item = item

    if meta_item is None:
        arcpy.AddError('Could not find selected capture in metadata file.')
        raise  # return

    # TODO: other checks

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ AND CHECK SENTINEL 2 DATA

    # arcpy.SetProgressor('default', 'Reading and checking fractional data...')
    #
    # sat_folder = os.path.join(in_project_folder, 'sat_captures')
    # monthly_meds_nc = os.path.join(sat_folder, 'monthly_meds.nc')
    #
    # if not os.path.exists(monthly_meds_nc):
    #     arcpy.AddError('Project file does not exist.')
    #     raise  # return
    #
    # with xr.open_dataset(monthly_meds_nc) as ds:
    #     ds.load()
    #
    # if 'time' not in ds or 'x' not in ds or 'y' not in ds:
    #     arcpy.AddError('Project data is not compatible.')
    #     raise  # return
    #
    # if len(ds['time']) == 0:
    #     arcpy.AddError('No dates detected in data.')
    #     raise  # return
    #
    visual_folder = os.path.join(in_project_folder, 'visualise')
    if not os.path.exists(visual_folder):
        arcpy.AddError('Visualise folder does not exist.')
        raise  #

    # TODO: check bands

    # extract netcdf attributes
    # ds_attrs = ds.attrs
    # ds_band_attrs = ds[list(ds)[0]].attrs
    # ds_spatial_ref_attrs = ds['spatial_ref'].attrs

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CONVERT AND DISPLAY TO NDVI

    # arcpy.SetProgressor('default', 'Generating NDVI from Sentinel 2 data...')
    #
    # ndvi = (ds['nbart_nir_1'] - ds['nbart_red']) / (ds['nbart_nir_1'] + ds['nbart_red'])
    # ndvi = ndvi.to_dataset(name='ndvi')
    #
    # # append attributes back on
    # ndvi.attrs = ds_attrs
    # ndvi['spatial_ref'].attrs = ds_spatial_ref_attrs
    # for var in ndvi:
    #     ndvi[var].attrs = ds_band_attrs
    #
    # # export combined monthly median as new netcdf
    # out_nc = os.path.join(sat_folder, 'monthly_ndvi.nc')
    # ndvi.to_netcdf(out_nc)
    # ndvi.close()
    #
    # # create a crf version of ndvi netcdf
    # out_crf = os.path.join(visual_folder, 'monthly_ndvi.crf')
    # arcpy.management.CopyRaster(in_raster=out_nc,
    #                             out_rasterdataset=out_crf,
    #                             format='CRF')
    #
    # # delete ndvi netcdf
    # arcpy.management.Delete(out_nc)
    #
    # try:
    #     # add new layer to map
    #     aprx = arcpy.mp.ArcGISProject('CURRENT')
    #     mp = aprx.activeMap
    #     mp.addDataFromPath(out_crf)
    #
    # except Exception as e:
    #     arcpy.AddError('Could not read generate NDVI time series data. See messages.')
    #     arcpy.AddMessage(str(e))
    #     #raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region Prepare fractional data

    arcpy.SetProgressor('default', 'Preparing fractional data...')

    # TODO: select specific capture date for fractions?
    captures_folder = os.path.join(in_project_folder, 'uav_captures')
    capture_folder = os.path.join(captures_folder, meta_item['capture_folder'])
    fractions_folder = os.path.join(capture_folder, 'fractions')

    if not os.path.exists(fractions_folder):
        arcpy.AddError('Fractions folder does not exist.')
        raise  # return

    if len(os.listdir(fractions_folder)) == 0:
        arcpy.AddError('No fractional layers available.')
        raise  # return

    fraction_ncs = []
    fraction_dates = sorted(os.listdir(fractions_folder))
    for fraction_date in fraction_dates:
        fraction_folder = os.path.join(fractions_folder, fraction_date)

        fraction_map = {
            'native': None,
            'weed': None,
            'other': None
        }

        files = os.listdir(fraction_folder)
        for file in files:
            if file.endswith('.tif'):
                if 'native' in file:
                    fraction_map['native'] = os.path.join(fraction_folder, file)
                elif 'weed' in file:
                    fraction_map['weed'] = os.path.join(fraction_folder, file)
                elif 'other' in file:
                    fraction_map['other'] = os.path.join(fraction_folder, file)

        if None in list(fraction_map.values()):
            arcpy.AddError('Not all fractional layers exist.')
            raise  # return

        ncs = []
        for k, v in fraction_map.items():

            # convert tif to nc
            tmp_nc = os.path.join(fraction_folder, f'tmp_{k}.nc')
            arcpy.management.CopyRaster(in_raster=fraction_map[k],
                                        out_rasterdataset=tmp_nc)

            # read netcdf as xr
            with xr.open_dataset(tmp_nc) as ds_tmp:
                ds_tmp.load()

            # set up crs info
            for band in ds_tmp:
                if len(ds_tmp[band].shape) == 0:
                    crs_name = band
                    crs_wkt = str(ds_tmp[band].attrs.get('spatial_ref'))
                    ds_tmp = ds_tmp.drop_vars(crs_name)
                    break

            ds_tmp = ds_tmp.assign_coords({'spatial_ref': 32750})
            ds_tmp['spatial_ref'].attrs = {
                'spatial_ref': crs_wkt,
                'grid_mapping_name': crs_name
            }

            if 'time' not in ds_tmp:
                dt = pd.to_datetime(fraction_date + '-01', format='%Y-%m-%d')
                ds_tmp = ds_tmp.assign_coords({'time': dt.to_numpy()})
                ds_tmp = ds_tmp.expand_dims('time')

            for dim in ds_tmp.dims:
                if dim in ['x', 'y', 'lat', 'lon']:
                    ds_tmp[dim].attrs = {
                        'resolution': np.mean(np.diff(ds_tmp[dim])),
                        'crs': f'EPSG:{32750}'
                    }

            for i, band in enumerate(ds_tmp):
                ds_tmp[band].attrs = {
                    'units': '1',
                    'crs': f'EPSG:{32570}',
                    'grid_mapping': 'spatial_ref',
                }

                ds_tmp = ds_tmp.rename({band: k})

            ds_tmp.attrs = {
                'crs': f'EPSG:{32570}',
                'grid_mapping': 'spatial_ref'
            }

            # append
            ncs.append(ds_tmp)

            # delete nc
            arcpy.management.Delete(tmp_nc)

        # combine datasets into one
        ds = xr.merge(ncs)
        ds.close()

        # append it
        fraction_ncs.append(ds)

    # combien fraction ncs into one
    ds = xr.concat(fraction_ncs, 'time').sortby('time')

    # extract netcdf attributes
    ds_attrs = ds.attrs
    ds_band_attrs = ds[list(ds)[0]].attrs
    ds_spatial_ref_attrs = ds['spatial_ref'].attrs

    # export combined fractionals as new netcdf
    #out_nc = os.path.join(visual_folder, 'monthly_fractionals.nc')
    #ds.to_netcdf(out_nc)
    #ds.close()

    # create a crf version of fractional netcdf
    #out_crf = os.path.join(visual_folder, 'monthly_fractionals.crf')
    #arcpy.management.CopyRaster(in_raster=out_nc,
                                #out_rasterdataset=out_crf,
                                #format='CRF')
    # delete ndvi netcdf
    #arcpy.management.Delete(out_nc)

    #try:
        # add new layer to map
        #aprx = arcpy.mp.ArcGISProject('CURRENT')
        #mp = aprx.activeMap
        #mp.addDataFromPath(out_crf)

    #except Exception as e:
        #arcpy.AddError('Could not read generate fractional time series data. See messages.')
        #arcpy.AddMessage(str(e))
        #raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region Restrict fractional data dates

    arcpy.SetProgressor('default', 'Restricting fractional data dates...')

    # TODO: get start rehab from json
    start_rehab_date = '2019-01-17'

    # restrict to 2017 or user rehab date
    if start_rehab_date < '2017-01-01':
        start_rehab_date = '2017-01-01'

    # strip day and add 01
    start_rehab_date = start_rehab_date[:7] + '-' + '01'

    # restrict dataset to start rehab or 2017-01-01
    ds = ds.sel(time=slice(start_rehab_date, None))

    # check if we have 3 or more years
    num_years = len(np.unique(ds['time.year']))
    if num_years < 3:
        arcpy.AddError('Need more than three years since capture date to calculate trend.')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region Calculate slope and curvature via ortho. polynomial coefficients

    arcpy.SetProgressor('default', 'Calculating trends per month over years...')

    # get opcs for linear and curvature fits
    lin_coeffs, lin_ss, lin_c, crv_coeffs, crv_ss, crv_c = shared.get_opcs(num_years)

    # ...

    for month in np.arange(1, 13):
        da = ds.where(ds['time.month'] == month, drop=True)

        # # get linear slope per pixel vector
        # kwargs = {'coeffs': lin_coeffs, 'ss': lin_ss, 'c': lin_c}
        # da_slp = xr.apply_ufunc(shared.apply_fit,
        #                         da,
        #                         input_core_dims=[['time']],
        #                         vectorize=True,
        #                         kwargs=kwargs)
        #
        # # get curvature per pixel vector
        # kwargs = {'coeffs': crv_coeffs, 'ss': crv_ss, 'c': crv_c}
        # da_crv = xr.apply_ufunc(shared.apply_fit,
        #                         da,
        #                         input_core_dims=[['time']],
        #                         vectorize=True,
        #                         kwargs=kwargs)
        #
        # # modify to highlight inc, dec and recovery trends
        # da_inc = (da_slp < 0) * np.abs(da_slp)  # neg values are increasing trend
        # da_dec = (da_slp > 0) * np.abs(da_slp)  # pos values are decreasing trend
        # da_rec = (da_crv < 0) * np.abs(da_crv)  # recent recovery has neg curve
        #
        # # standardise to 0-255
        # da_inc = (da_inc / da_inc.max() * 255)
        # da_dec = (da_dec / da_dec.max() * 255)
        # da_rec = (da_rec / da_rec.max() * 255)

        # get mankendall tau per pixel vector
        da_tau = xr.apply_ufunc(shared.apply_mk_tau,
                                da,
                                input_core_dims=[['time']],
                                vectorize=True)

        # get mankendall pvalue per pixel vector
        da_pvl = xr.apply_ufunc(shared.apply_mk_pvalue,
                                da,
                                input_core_dims=[['time']],
                                vectorize=True)

        # set output folder # TODO: add this to capture
        trends_folder = r"C:\Users\Lewis\Desktop\trends"

        # iter each dataset and export each band
        native_tifs, weed_tifs, other_tifs = [], [], []
        #names, das = ['dec', 'rec', 'inc'], [da_dec, da_rec, da_inc]
        names, das = ['tau', 'pvalue'], [da_tau, da_pvl]
        for name, da in zip(names, das):

            # append attributes back on
            da.attrs = ds_attrs
            da['spatial_ref'].attrs = ds_spatial_ref_attrs
            for var in da:
                da[var].attrs = ds_band_attrs

            for var in da:
                out_nc = os.path.join(trends_folder, f'trend_{name}_{var}_month_{month}.nc')
                out_tif = os.path.join(trends_folder, f'trend_{name}_{var}_month_{month}.tif')

                da[var].to_netcdf(out_nc)

                dataset = gdal.Open(out_nc, gdal.GA_ReadOnly)
                dataset = gdal.Translate(out_tif, dataset)
                dataset = None

                arcpy.management.Delete(out_nc)

                #if var == 'native':
                    #native_tifs.append(out_tif)
                #elif var == 'weed':
                    #weed_tifs.append(out_tif)
                #elif var == 'other':
                    #other_tifs.append(out_tif)

        # composite bands in order of dec, rec and inc for natives
        # tifs = [native_tifs, weed_tifs, other_tifs]
        # for name, bands in zip(['native', 'weed', 'other'], tifs):
        #     tmp_comp = os.path.join(trends_folder, f'trend_comp_{name}_month_{month}.tif')
        #     arcpy.management.CompositeBands(in_rasters=bands,
        #                                     out_raster=tmp_comp)
        #
        #     # delete original bands
        #     for band in bands:
        #         arcpy.management.Delete(band)











        1




    1


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region APPEND NEW CAPTURE TO METADATA

    # arcpy.SetProgressor('default', 'Updating metadata...')
    #
    # meta['data'].append(
    #     {
    #         'capture_folder': 'uav' + '_' + flight_date,
    #         'capture_date': in_flight_datetime.strftime('%Y-%m-%d %H:%M:%S'),
    #         'type': 'additional',
    #         'ingested': True,
    #         'classified': False
    #     }
    # )
    #
    # with open(in_project_file, 'w') as fp:
    #     json.dump(meta, fp)

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    # TODO: enable if move to non-memory temp files
    # try:
    #     # drop temp files (free up space)
    #     arcpy.management.Delete(tmp_comp)
    #
    # except Exception as e:
    #     arcpy.AddWarning('Could not drop temporary files. See messages.')
    #     arcpy.AddMessage(str(e))

    arcpy.CheckInExtension('Spatial')

    arcpy.env.overwriteOutput = False
    arcpy.env.addOutputsToMap = True

    # endregion

    return

# testing
execute(None)