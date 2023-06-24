
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

    # inputs from arcgis pro ui
    in_project_file = parameters[0].valueAsText

    # inputs for testing only
    #in_project_file = r'D:\Work\Curtin\Water Corp Project - General\Testing\Demo\Projects\city beach demo full\meta.json'

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE ENVIRONMENT

    arcpy.SetProgressor('default', 'Preparing environment...')

    # check if user has spatial analyst, error if not
    if arcpy.CheckExtension('Spatial') != 'Available':
        arcpy.AddError('Spatial Analyst license is unavailable.')
        raise  # return
    # TODO: remove below if wc has no ia
    elif arcpy.CheckExtension('ImageAnalyst') != 'Available':
        arcpy.AddError('Image Analyst license is unavailable.')
        raise  # return
    else:
        arcpy.CheckOutExtension('Spatial')
        arcpy.CheckOutExtension('ImageAnalyst')  # TODO: remove if wc has no ia

    # set data overwrites and mapping
    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = False

    # set current workspace to scratch folder
    arcpy.env.workspace = arcpy.env.scratchFolder

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK PROJECT FOLDER STRUCTURE AND FILES

    arcpy.SetProgressor('default', 'Checking project folders and files...')

    # check if input project file exists
    if not os.path.exists(in_project_file):
        arcpy.AddError('Project file does not exist.')
        raise  # return

    # get top-level project folder from project file
    in_project_folder = os.path.dirname(in_project_file)

    # check if required project folders already exist, error if so
    sub_folders = ['grid', 'uav_captures', 'sat_captures', 'visualise']
    for sub_folder in sub_folders:
        sub_folder = os.path.join(in_project_folder, sub_folder)
        if not os.path.exists(sub_folder):
            arcpy.AddError('Project is missing required folders.')
            raise  # return

    # build visualise folder
    visualise_folder = os.path.join(in_project_folder, 'visualise')

    # check if uav grid file exists
    #grid_tif = os.path.join(in_project_folder, 'grid', 'grid_uav.tif')
    #if not os.path.exists(grid_tif):
        #arcpy.AddError('Project grid file does not exist.')
        #raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ AND CHECK METADATA

    arcpy.SetProgressor('default', 'Reading and checking metadata...')

    try:
        # read project json file
        with open(in_project_file, 'r') as fp:
            meta = json.load(fp)

    except Exception as e:
        arcpy.AddError('Could not read metadata. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # check if any captures exist (will be >= 4)
    if len(meta) < 4:
        arcpy.AddError('Project has no UAV capture data.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ AND CHECK SENTINEL 2 DATA

    arcpy.SetProgressor('default', 'Reading and checking Sentinel 2 data...')

    # build sat captures folder and combine netcdfs folder
    sat_folder = os.path.join(in_project_folder, 'sat_captures')
    combine_ncs_folder = os.path.join(sat_folder, 'cmb_ncs')

    # build monthly median netcdf folder
    monthly_meds_nc = os.path.join(combine_ncs_folder, 'raw_monthly_meds.nc')

    # check netcdf exists (thus folder)
    if not os.path.exists(monthly_meds_nc):
        arcpy.AddError('Project monthly median NetCDF does not exist.')
        raise  # return

    try:
        # load combined netcdf
        with xr.open_dataset(monthly_meds_nc) as ds:
            ds.load()

        # check dims exist
        if 'time' not in ds or 'x' not in ds or 'y' not in ds:
            arcpy.AddError('Sentinel 2 NetCDF is not compatible.')
            raise  # return

        # check we have times
        if len(ds['time']) == 0:
            arcpy.AddError('No time dimension detected in Sentinel 2 NetCDF.')
            raise  # return

        # extract netcdf attributes
        ds_attrs = ds.attrs
        ds_band_attrs = ds[list(ds)[0]].attrs
        ds_spatial_ref_attrs = ds['spatial_ref'].attrs

    except Exception as e:
        arcpy.AddWarning('Could not read Sentinel 2 NetCDF. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE NDVI

    arcpy.SetProgressor('default', 'Calculating NDVI from Sentinel 2 data...')

    try:
        # calculate ndvi for each slice
        ds['ndvi'] = (ds['nbart_nir_1'] - ds['nbart_red']) / (ds['nbart_nir_1'] + ds['nbart_red'])
        ds = ds[['ndvi']]

        # append attributes back on
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds:
            ds[var].attrs = ds_band_attrs

        # export combined monthly median as new netcdf
        #out_nc = os.path.join(sat_folder, 'monthly_ndvi.nc')
        #ndvi.to_netcdf(out_nc)
        #ndvi.close()

        # create a crf version of ndvi netcdf
        #out_crf = os.path.join(visual_folder, 'monthly_ndvi.crf')
        #arcpy.management.CopyRaster(in_raster=out_nc,
                                    #out_rasterdataset=out_crf,
                                    #format='CRF')

        # delete ndvi netcdf
        #arcpy.management.Delete(out_nc)

    except Exception as e:
        arcpy.AddError('Could not calculate NDVI. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region SUBSET NDVI DATASET TO START OF REHAB

    arcpy.SetProgressor('default', 'Subsetting Sentinel 2 dates...')

    try:
        # check and get start of rehab date
        rehab_start_date = meta.get('date_rehab')
        if rehab_start_date is None:
            arcpy.AddError('Project has no start of rehab date.')
            raise  # return

        # remove time if exists
        rehab_start_date = rehab_start_date.split(' ')[0].strip()

        # if less than 2017-01-01, use 2017
        if rehab_start_date < '2017-01-01':
            arcpy.AddWarning('Rehabilitation start date < 2017-01-01, using 2017-01-01.')
            rehab_start_date = '2017-01-01'

        # strip day and replace with is 01 to conform with cube
        rehab_start_date = rehab_start_date[:-3] + '-' + '01'

        # subset dataset
        ds = ds.sel(time=slice(rehab_start_date, None))

        # check if we have 3 or more years
        num_years = len(np.unique(ds['time.year']))
        if num_years < 3:
            arcpy.AddError('Need >= 3 years since rehabilitation start to calculate trends.')
            raise # return

    except Exception as e:
        arcpy.AddError('Could not subset Sentinel 2 data. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE MONTHLY SLOPE, CURVATURE VIA OPC

    arcpy.SetProgressor('default', 'Calculating monthly trends via OPCs...')

    try:
        # get opcs for linear and curvature fits
        lin_coeffs, lin_ss, lin_c, crv_coeffs, crv_ss, crv_c = shared.get_opcs(num_years)

        # iter each month in netcdf...
        for month in np.arange(1, 13):
            # extract current month only
            da = ds.where(ds['time.month'] == month, drop=True)

            # TODO REMOVE THIS
            da['x'] = da['x'] - 10
            da['y'] = da['y'] - 10

            # check we have 3 full years (last year might not be full)
            slice_num_years = len(np.unique(da['time.year']))
            if slice_num_years < 3 or slice_num_years != num_years:
                arcpy.AddWarning(f'Month {month} does not have 3 full years or missing a year, skipping.')
                continue
                # TODO: create null da?

            # get linear slope per pixel vector
            kwargs = {'coeffs': lin_coeffs, 'ss': lin_ss, 'c': lin_c}
            da_slp = xr.apply_ufunc(shared.apply_fit,
                                    da,
                                    input_core_dims=[['time']],
                                    vectorize=True,
                                    kwargs=kwargs)

            # get curvature per pixel vector
            kwargs = {'coeffs': crv_coeffs, 'ss': crv_ss, 'c': crv_c}
            da_crv = xr.apply_ufunc(shared.apply_fit,
                                    da,
                                    input_core_dims=[['time']],
                                    vectorize=True,
                                    kwargs=kwargs)

            # modify to highlight inc, dec and recovery trends
            da_inc = (da_slp < 0) * np.abs(da_slp)  # neg values are increasing trend
            da_dec = (da_slp > 0) * np.abs(da_slp)  # pos values are decreasing trend
            da_rec = (da_crv < 0) * np.abs(da_crv)  # recent recovery has neg curve

            # standardise to 0-255
            da_inc = (da_inc / da_inc.max() * 255)
            da_dec = (da_dec / da_dec.max() * 255)
            da_rec = (da_rec / da_rec.max() * 255)

            # get mankendall tau per pixel vector
            # TODO: might not use this
            da_tau = xr.apply_ufunc(shared.apply_mk_tau,
                                    da,
                                    input_core_dims=[['time']],
                                    vectorize=True)

            # get mankendall pvalue per pixel vector
            # TODO: might not use this
            da_pvl = xr.apply_ufunc(shared.apply_mk_pvalue,
                                    da,
                                    input_core_dims=[['time']],
                                    vectorize=True)

            # rename array vars
            da_inc = da_inc.rename({'ndvi': 'opc_inc'})
            da_dec = da_dec.rename({'ndvi': 'opc_dec'})
            da_rec = da_rec.rename({'ndvi': 'opc_rec'})
            da_tau = da_tau.rename({'ndvi': 'mk_tau'})
            da_pvl = da_pvl.rename({'ndvi': 'mk_pvl'})

            # combine into one
            da = xr.merge([da_inc, da_dec, da_rec, da_tau, da_pvl])

            # append attributes back on
            da.attrs = ds_attrs
            da['spatial_ref'].attrs = ds_spatial_ref_attrs
            for var in da:
                da[var].attrs = ds_band_attrs

            # set up scratch folder
            scratch = arcpy.env.scratchFolder

            # init raster map
            ras_map = {
                'opc_inc': None,
                'opc_dec': None,
                'opc_rec': None,
                'mk_tau': None,
                'mk_pvl': None,
            }

            # iter each array...
            for var in da.data_vars:
                # extract var as new array
                da_var = da[var]

                # export var as new netcdf
                out_nc = os.path.join(scratch, f'{var}_month_{month}.nc')
                da_var.to_netcdf(out_nc)

                # convert to tif
                out_tif = os.path.join(scratch, f'{var}_month_{month}.tif')
                dataset = gdal.Open(out_nc, gdal.GA_ReadOnly)
                dataset = gdal.Translate(out_tif, dataset)
                dataset = None

                # update raster map
                ras_map[var] = out_tif

            # TODO: check all are there
            # ...

            # read in as rasters
            ras_inc = arcpy.Raster(ras_map['opc_inc'])
            ras_dec = arcpy.Raster(ras_map['opc_dec'])
            ras_rec = arcpy.Raster(ras_map['opc_rec'])
            ras_tau = arcpy.Raster(ras_map['mk_tau'])
            ras_pvl = arcpy.Raster(ras_map['mk_pvl'])

            # TODO: project all to wgs 84 utm zone 50 s
            ras_inc_prj = arcpy.ia.Reproject(ras_inc, {'wkid': 32750})
            ras_dec_prj = arcpy.ia.Reproject(ras_dec, {'wkid': 32750})
            ras_rec_prj = arcpy.ia.Reproject(ras_rec, {'wkid': 32750})
            ras_tau_prj = arcpy.ia.Reproject(ras_tau, {'wkid': 32750})
            ras_pvl_prj = arcpy.ia.Reproject(ras_pvl, {'wkid': 32750})

            # combine inc, dec, rec to rgb in order dec, rec, inc
            ras_cmb = arcpy.ia.CompositeBand([ras_dec_prj, ras_rec_prj, ras_inc_prj])

            # save all rasters to visual folder
            # TODO: elsewhere?
            ras_cmb.save(os.path.join(visualise_folder, f'trend_rgb_month_{month}.tif'))
            ras_tau_prj.save(os.path.join(visualise_folder, f'trend_tau_month_{month}.tif'))
            ras_pvl_prj.save(os.path.join(visualise_folder, f'trend_pvl_month_{month}.tif'))

    except Exception as e:
        arcpy.AddError('Could not calculate slope and curvature. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region Prepare fractional data

    #arcpy.SetProgressor('default', 'Preparing fractional data...')

    # TODO: select specific capture date for fractions?
    # captures_folder = os.path.join(in_project_folder, 'uav_captures')
    # capture_folder = os.path.join(captures_folder, meta_item['capture_folder'])
    # fractions_folder = os.path.join(capture_folder, 'fractions')

    # if not os.path.exists(fractions_folder):
    #     arcpy.AddError('Fractions folder does not exist.')
    #     raise  # return
    #
    # if len(os.listdir(fractions_folder)) == 0:
    #     arcpy.AddError('No fractional layers available.')
    #     raise  # return
    #
    # fraction_ncs = []
    # fraction_dates = sorted(os.listdir(fractions_folder))
    # for fraction_date in fraction_dates:
    #     fraction_folder = os.path.join(fractions_folder, fraction_date)
    #
    #     fraction_map = {
    #         'native': None,
    #         'weed': None,
    #         'other': None
    #     }
    #
    #     files = os.listdir(fraction_folder)
    #     for file in files:
    #         if file.endswith('.tif'):
    #             if 'native' in file:
    #                 fraction_map['native'] = os.path.join(fraction_folder, file)
    #             elif 'weed' in file:
    #                 fraction_map['weed'] = os.path.join(fraction_folder, file)
    #             elif 'other' in file:
    #                 fraction_map['other'] = os.path.join(fraction_folder, file)
    #
    #     if None in list(fraction_map.values()):
    #         arcpy.AddError('Not all fractional layers exist.')
    #         raise  # return
    #
    #     ncs = []
    #     for k, v in fraction_map.items():
    #
    #         # convert tif to nc
    #         tmp_nc = os.path.join(fraction_folder, f'tmp_{k}.nc')
    #         arcpy.management.CopyRaster(in_raster=fraction_map[k],
    #                                     out_rasterdataset=tmp_nc)
    #
    #         # read netcdf as xr
    #         with xr.open_dataset(tmp_nc) as ds_tmp:
    #             ds_tmp.load()
    #
    #         # set up crs info
    #         for band in ds_tmp:
    #             if len(ds_tmp[band].shape) == 0:
    #                 crs_name = band
    #                 crs_wkt = str(ds_tmp[band].attrs.get('spatial_ref'))
    #                 ds_tmp = ds_tmp.drop_vars(crs_name)
    #                 break
    #
    #         ds_tmp = ds_tmp.assign_coords({'spatial_ref': 32750})
    #         ds_tmp['spatial_ref'].attrs = {
    #             'spatial_ref': crs_wkt,
    #             'grid_mapping_name': crs_name
    #         }
    #
    #         if 'time' not in ds_tmp:
    #             dt = pd.to_datetime(fraction_date + '-01', format='%Y-%m-%d')
    #             ds_tmp = ds_tmp.assign_coords({'time': dt.to_numpy()})
    #             ds_tmp = ds_tmp.expand_dims('time')
    #
    #         for dim in ds_tmp.dims:
    #             if dim in ['x', 'y', 'lat', 'lon']:
    #                 ds_tmp[dim].attrs = {
    #                     'resolution': np.mean(np.diff(ds_tmp[dim])),
    #                     'crs': f'EPSG:{32750}'
    #                 }
    #
    #         for i, band in enumerate(ds_tmp):
    #             ds_tmp[band].attrs = {
    #                 'units': '1',
    #                 'crs': f'EPSG:{32570}',
    #                 'grid_mapping': 'spatial_ref',
    #             }
    #
    #             ds_tmp = ds_tmp.rename({band: k})
    #
    #         ds_tmp.attrs = {
    #             'crs': f'EPSG:{32570}',
    #             'grid_mapping': 'spatial_ref'
    #         }
    #
    #         # append
    #         ncs.append(ds_tmp)
    #
    #         # delete nc
    #         arcpy.management.Delete(tmp_nc)
    #
    #     # combine datasets into one
    #     ds = xr.merge(ncs)
    #     ds.close()
    #
    #     # append it
    #     fraction_ncs.append(ds)
    #
    # # combien fraction ncs into one
    # ds = xr.concat(fraction_ncs, 'time').sortby('time')
    #
    # # extract netcdf attributes
    # ds_attrs = ds.attrs
    # ds_band_attrs = ds[list(ds)[0]].attrs
    # ds_spatial_ref_attrs = ds['spatial_ref'].attrs

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

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region Restrict fractional data dates

    #arcpy.SetProgressor('default', 'Restricting fractional data dates...')

    # # TODO: get start rehab from json
    # start_rehab_date = '2019-01-17'
    #
    # # restrict to 2017 or user rehab date
    # if start_rehab_date < '2017-01-01':
    #     start_rehab_date = '2017-01-01'
    #
    # # strip day and add 01
    # start_rehab_date = start_rehab_date[:7] + '-' + '01'
    #
    # # restrict dataset to start rehab or 2017-01-01
    # ds = ds.sel(time=slice(start_rehab_date, None))
    #
    # # check if we have 3 or more years
    # num_years = len(np.unique(ds['time.year']))
    # if num_years < 3:
    #     arcpy.AddError('Need more than three years since capture date to calculate trend.')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region Calculate slope and curvature via ortho. polynomial coefficients

    #arcpy.SetProgressor('default', 'Calculating trends per month over years...')

    # get opcs for linear and curvature fits
    #lin_coeffs, lin_ss, lin_c, crv_coeffs, crv_ss, crv_c = shared.get_opcs(num_years)

    # ...

    #for month in np.arange(1, 13):
        #da = ds.where(ds['time.month'] == month, drop=True)

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

        # # get mankendall tau per pixel vector
        # da_tau = xr.apply_ufunc(shared.apply_mk_tau,
        #                         da,
        #                         input_core_dims=[['time']],
        #                         vectorize=True)
        #
        # # get mankendall pvalue per pixel vector
        # da_pvl = xr.apply_ufunc(shared.apply_mk_pvalue,
        #                         da,
        #                         input_core_dims=[['time']],
        #                         vectorize=True)
        #
        # # set output folder # TODO: add this to capture
        # trends_folder = r"C:\Users\Lewis\Desktop\trends"

        # iter each dataset and export each band
        # native_tifs, weed_tifs, other_tifs = [], [], []
        # #names, das = ['dec', 'rec', 'inc'], [da_dec, da_rec, da_inc]
        # names, das = ['tau', 'pvalue'], [da_tau, da_pvl]
        # for name, da in zip(names, das):
        #
        #     # append attributes back on
        #     da.attrs = ds_attrs
        #     da['spatial_ref'].attrs = ds_spatial_ref_attrs
        #     for var in da:
        #         da[var].attrs = ds_band_attrs
        #
        #     for var in da:
        #         out_nc = os.path.join(trends_folder, f'trend_{name}_{var}_month_{month}.nc')
        #         out_tif = os.path.join(trends_folder, f'trend_{name}_{var}_month_{month}.tif')
        #
        #         da[var].to_netcdf(out_nc)
        #
        #         dataset = gdal.Open(out_nc, gdal.GA_ReadOnly)
        #         dataset = gdal.Translate(out_tif, dataset)
        #         dataset = None
        #
        #         arcpy.management.Delete(out_nc)
        #
        #         #if var == 'native':
        #             #native_tifs.append(out_tif)
        #         #elif var == 'weed':
        #             #weed_tifs.append(out_tif)
        #         #elif var == 'other':
        #             #other_tifs.append(out_tif)

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

    # free up spatial analyst
    arcpy.CheckInExtension('Spatial')
    arcpy.CheckInExtension('ImageAnalyst')  # TODO: remove if wc has no ia

    # set changed env variables back to default
    arcpy.env.overwriteOutput = False
    arcpy.env.addOutputsToMap = True

    # endregion

    return

# testing
#execute(None)