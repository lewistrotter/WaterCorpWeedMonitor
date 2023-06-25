
import os
import json
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp
import arcpy

from osgeo import gdal
from scipy.stats import zscore
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from scripts import web, shared


def execute(
        parameters
        # messages # TODO: implement
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    in_project_file = parameters[0].valueAsText
    in_flight_datetime = parameters[1].value

    # inputs for testing only
    #in_project_file = r'C:\Users\Lewis\Desktop\testing\full\meta.json'
    #in_flight_datetime = '2023-06-08 16:35:07'

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
            arcpy.AddError('Project folder is missing expected folders.')
            raise  # return

    # check if s2 grid file exists
    grid_tif = os.path.join(in_project_folder, 'grid', 'grid_s2.tif')
    if not os.path.exists(grid_tif):
        arcpy.AddError('Sentinel 2 grid raster does not exist.')
        raise  # return

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

    # check and get start of rehab date
    rehab_start_date = meta.get('date_rehab')
    if rehab_start_date is None:
        arcpy.AddError('Project has no start of rehab date.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT SELECTED UAV CAPTURE METADATA

    arcpy.SetProgressor('default', 'Extracting selected UAV capture metadata...')

    # exclude top-level metadata items
    exclude_keys = ['project_name', 'date_created', 'date_rehab']

    # extract selected metadata item based on capture date
    meta_item = None
    for k, v in meta.items():
        if k not in exclude_keys:
            if v['capture_date'] == in_flight_datetime:
                meta_item = v

    # check if meta item exists, else error
    if meta_item is None:
        arcpy.AddError('Could not find selected UAV capture in metadata file.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region FETCHING CLEAN DEA STAC SENTINEL 2 DOWNLOADS

    arcpy.SetProgressor('default', 'Fetching clean DEA STAC downloads...')

    # set sat folder for nc outputs
    sat_folder = os.path.join(in_project_folder, 'sat_captures')

    try:
        # query and prepare downloads
        downloads = web.get_s2_wc_downloads(grid_tif=grid_tif,
                                            out_folder=sat_folder)

    except Exception as e:
        arcpy.AddError('Unable to download Sentinel 2 data from DEA. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # if nothing left, leave
    if len(downloads) == 0:
        arcpy.AddWarning('No valid satellite downloads were found.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region DOWNLOAD WCS DATA

    arcpy.SetProgressor('step', 'Downloading Sentinel 2 data...', 0, len(downloads), 1)

    # set relevant download parameters
    num_cpu = int(np.ceil(mp.cpu_count() / 2))

    try:
        i = 0
        results = []
        with ThreadPoolExecutor(max_workers=num_cpu) as pool:
            futures = []
            for download in downloads:
                task = pool.submit(web.validate_and_download,
                                   download,
                                   [1],   # quality_flags,
                                   1,     # max_out_of_bounds,
                                   0,     # max_invalid_pixels,
                                   -999)  # nodata_value
                futures.append(task)

            for future in as_completed(futures):
                arcpy.AddMessage(future.result())
                results.append(future.result())

                i += 1
                if i % 1 == 0:
                    arcpy.SetProgressorPosition(i)

    except Exception as e:
        arcpy.AddError('Unable to download Sentinel 2 data from DEA. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # check if any valid downloads (non-cloud or new)
    num_valid_downlaods = len([dl for dl in results if 'success' in dl])
    if num_valid_downlaods == 0:
        arcpy.AddMessage('No new valid satellite downloads were found.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region COMBINE NETCDFS INTO MONTHLY MEDIANS

    arcpy.SetProgressor('default', 'Combining Sentinel 2 data...')

    # set raw output nc folder (one nc per date)
    raw_ncs_folder = os.path.join(sat_folder, 'raw_ncs')
    if not os.path.exists(raw_ncs_folder):
        os.mkdir(raw_ncs_folder)

    # get all raw ncs in raw netcdf folder
    nc_files = []
    for file in os.listdir(raw_ncs_folder):
        if file.startswith('R') and file.endswith('.nc'):
            nc_files.append(os.path.join(raw_ncs_folder, file))

    # check if anything came back, error if not
    if len(nc_files) == 0:
        arcpy.AddError('No NetCDF files were found.')
        raise  # return

    try:
        # load each netcdf and append to list
        ds_list = []
        for nc_file in nc_files:
            with xr.open_dataset(nc_file) as nc:
                nc.load()
            ds_list.append(nc)

        # combine all netcdfs into one and sort by date
        ds = xr.concat(ds_list, 'time').sortby('time')

        # extract netcdf attributes
        ds_attrs = ds.attrs
        ds_band_attrs = ds[list(ds)[0]].attrs
        ds_spatial_ref_attrs = ds['spatial_ref'].attrs

        # set nodata (-999) to nan
        ds = ds.where(ds != -999)

        # detect outliers via z-score, make mask, set pixel to nan when any band is outlier nan
        z_mask = xr.apply_ufunc(zscore, ds, 0)    # 0 is time dim
        z_mask = np.abs(z_mask) > 3.29            # p-value 0.001
        z_mask = z_mask.to_array().max('variable')

        # set pixels to nan where outlier detected
        ds = ds.where(~z_mask)

        # resample to monthly means and interpolate
        ds = ds.resample(time='1MS').median('time')
        ds = ds.interpolate_na('time')

        # append attributes back on
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds:
            ds[var].attrs = ds_band_attrs

        # set combined output nc folder (one nc per month)
        cmb_ncs_folder = os.path.join(sat_folder, 'cmb_ncs')
        if not os.path.exists(cmb_ncs_folder):
            os.mkdir(cmb_ncs_folder)

        # export combined monthly median as new netcdf
        out_montly_meds_nc = os.path.join(cmb_ncs_folder, 'raw_monthly_meds.nc')
        ds.to_netcdf(out_montly_meds_nc)
        ds.close()

    except Exception as e:
        arcpy.AddError('Unable to combine Sentinel 2 NetCDFs. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region LOAD HIGH RES CLASSIFIED DRONE IMAGE AS XR DATASET

    arcpy.SetProgressor('default', 'Preparing high-resolution classified UAV data...')

    # set up capture and classify folders
    capture_folders = os.path.join(in_project_folder, 'uav_captures')
    capture_folder = os.path.join(capture_folders, meta_item['capture_folder'])
    classify_folder = os.path.join(capture_folder, 'classify')

    # get optimal classified rf
    optimal_rf_tif = None
    for file in os.listdir(classify_folder):
        if 'rf_optimal' in file and file.endswith('.tif'):
            optimal_rf_tif = os.path.join(classify_folder, file)

    # check if optimal rf exists, error if not
    if optimal_rf_tif is None:
        arcpy.AddError('No optimal classified UAV image exists.')
        raise  # return

    try:
        # read classified uav image (take tif of best classified model) into scratch
        tmp_class_nc = os.path.join(arcpy.env.workspace, 'rf_optimal.nc')
        arcpy.management.CopyRaster(in_raster=optimal_rf_tif,
                                    out_rasterdataset=tmp_class_nc)

        # read it in as netcdf
        with xr.open_dataset(tmp_class_nc) as ds_hr:
            ds_hr.load()

        # prepare it for use (get 2d array, set nan to -999, int16 for speed)
        da_hr = ds_hr[['Band1']].to_array().squeeze(drop=True)
        da_hr = xr.where(~np.isnan(da_hr), da_hr, -999).astype('int16')

    except Exception as e:
        arcpy.AddError('Could not load classified UAV image. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region LOAD LOW RES SENTINEL 2 MEDIAN XR DATASET

    arcpy.SetProgressor('default', 'Preparing low-resolution Sentinel 2 data...')

    try:
        # load monthly median low-res satellite data
        with xr.open_dataset(out_montly_meds_nc) as ds_lr:
            ds_lr.load()

        # slice from 2017-01-01 up to now
        ds_lr = ds_lr.sel(time=slice('2017-01-01', None))

    except Exception as e:
        arcpy.AddError('Could not load Sentinel 2 images. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # check if any time slices exist
    if len(ds_lr['time']) == 0:
        arcpy.AddError('No time slices in Sentinel 2 data.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region GENERATE FRACTIONAL DATA

    arcpy.SetProgressor('default', 'Generating fractional maps...')

    # create fractions folder if not exists
    fractions_folder = os.path.join(capture_folder, 'fractions')
    if not os.path.exists(fractions_folder):
        os.mkdir(fractions_folder)

    # get list of prior processed year-month fractional folders
    exist_fractions = meta_item.get('fractions')
    if exist_fractions is None:
        arcpy.AddError('No fraction list detected.')
        raise  # return

    # set up step-wise progressor
    arcpy.SetProgressor('step', None, 0, len(ds_lr['time']))

    # TODO: break these up into seperate try excepts
    try:
        # iterate each date...
        for i in range(0, len(ds_lr['time'])):
            # extract time slice array and get year-month
            da = ds_lr.isel(time=i)
            dt = str(da['time'].dt.strftime('%Y-%m').values)

            # skip if fraction date already exists
            if dt in exist_fractions:
                arcpy.AddMessage(f'Skipping fractions for date: {dt}, already done.')
                continue

            # notify
            arcpy.AddMessage(f'Generating fractions for date: {dt}.')

            # set new folder named year-month
            fraction_folder = os.path.join(fractions_folder, dt)
            if not os.path.exists(fraction_folder):
                os.mkdir(fraction_folder)

            # convert current data array vars into band tifs, return list of paths
            out_band_tifs = shared.export_xr_vars_into_tifs(da=da,
                                                            out_folder=arcpy.env.workspace)

            # read band tifs into composite and reproject to be safe
            tmp_cmp = arcpy.ia.CompositeBand(out_band_tifs)
            tmp_prj = arcpy.ia.Reproject(tmp_cmp, {'wkid': 32750})

            # build roi polygons from low res pixels and freqs from high-res raster
            tmp_env = shared.build_lr_freqs_rois_from_hr_xr(ras_lr=tmp_prj,
                                                            da_hr=da_hr)

            # save training and validation rois for current date
            tmp_rois = os.path.join(fraction_folder, 'frac_rois.shp')
            arcpy.analysis.Select(in_features=tmp_env,
                                  out_feature_class=tmp_rois,
                                  where_clause='inc = 1')

            # iter each class for fractional mapping...
            for classvalue in ["c_0", "c_1", "c_2"]:

                # set explanotory vars as bands
                exp_vars = out_band_tifs

                # set current readable name based on current class code
                class_desc = None
                if classvalue == 'c_0':
                    class_desc = 'other'
                elif classvalue == 'c_1':
                    class_desc = 'native'
                elif classvalue == 'c_2':
                    class_desc = 'weed'

                # create output regression tif and cmatrix
                out_fn = f'{dt}_{classvalue}_{class_desc}'.replace('-', '_')
                out_frc_tif = os.path.join(fraction_folder, 'frac_' + out_fn + '.tif')
                out_cmx_csv = os.path.join(fraction_folder, 'cm_' + out_fn + '.csv')

                # perform regression and export files
                shared.regress(in_rois=tmp_rois,
                               in_classvalue=classvalue,
                               in_class_desc=class_desc,
                               in_exp_vars=exp_vars,
                               out_regress_tif=out_frc_tif,
                               out_cmatrix_csv=out_cmx_csv)

            # delete temp comp projected comp
            del tmp_cmp
            del tmp_prj

            # delete temporary rois
            arcpy.management.Delete(tmp_rois)

            # delete remaining bands
            for band in exp_vars:
                arcpy.management.Delete(band)

            # add successful fractional date to metadata
            meta_item['fractions'].append(dt)

    except Exception as e:
        arcpy.AddError('Could not generate fractional map. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region UPDATE NEW FRACTIONAL INFO IN METADATA

    arcpy.SetProgressor('default', 'Updating metadata...')

    try:
        # write json metadata file to project folder top-level
        with open(in_project_file, 'w') as fp:
            json.dump(meta, fp)

    except Exception as e:
        arcpy.AddError('Could not write metadata. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

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