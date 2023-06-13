
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

from scripts import web  #, shared


def execute(
        parameters
        # messages # TODO: implement
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    in_project_file = parameters[0].valueAsText

    # TODO: uncomment these when testing
    #in_project_file = r'C:\Users\Lewis\Desktop\testing\project_1\meta.json'

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

    grid_tif = os.path.join(in_project_folder, 'grid', 'grid_s2.tif')

    if not os.path.exists(grid_tif):
        arcpy.AddError('Sentinel 2 grid raster does not exist.')
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

    # TODO: extract rehab start
    rehab_start_date = '2021-03-17'

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE STAC QUERY

    arcpy.SetProgressor('default', 'Querying DEA STAC endpoint...')

    start_date, end_date = '2017-01-01', '2039-12-31'

    collections = [
        'ga_s2am_ard_3',
        'ga_s2bm_ard_3'
    ]

    # reproject grid to wgs84 bounding box
    tmp_prj = r'memory\tmp_prj'  # os.path.join(tmp_folder, 'tmp_prj.tif')
    arcpy.management.ProjectRaster(in_raster=grid_tif,
                                   out_raster=tmp_prj,
                                   out_coor_system=arcpy.SpatialReference(4326))

    # get bounding box in wgs84 for stac query
    extent = arcpy.Describe(tmp_prj).extent
    x_min, y_min = float(extent.XMin), float(extent.YMin)
    x_max, y_max = float(extent.XMax), float(extent.YMax)
    stac_bbox = x_min, y_min, x_max, y_max

    # get stac features from 2018-01-01 to now
    stac_features = web.fetch_all_stac_features(collections=collections,
                                                start_date=start_date,
                                                end_date=end_date,
                                                bbox=stac_bbox,
                                                limit=100)

    if len(stac_features) == 0:
        arcpy.AddWarning('No STAC features were found.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE STAC FEATURES

    arcpy.SetProgressor('default', 'Preparing DEA STAC features...')

    sat_folder = os.path.join(in_project_folder, 'sat_captures')

    assets = [
        'nbart_blue',
        'nbart_green',
        'nbart_red',
        'nbart_red_edge_1',
        'nbart_red_edge_2',
        'nbart_red_edge_3',
        'nbart_nir_1',
        'nbart_nir_2',
        'nbart_swir_2',
        'nbart_swir_3'
    ]

    quality_flags = [1]  # s2cloudless valid is 1  # [1, 3, 5]  # valid, shadow, water

    # reproject grid to albers bounding box
    tmp_prj = r'memory\tmp_prj'  # os.path.join(tmp_folder, 'tmp_prj.tif')
    arcpy.management.ProjectRaster(in_raster=grid_tif,
                                   out_raster=tmp_prj,
                                   out_coor_system=arcpy.SpatialReference(3577))

    # get bounding box in albers for download query
    extent = arcpy.Describe(tmp_prj).extent
    x_min, y_min = float(extent.XMin), float(extent.YMin)
    x_max, y_max = float(extent.XMax), float(extent.YMax)
    out_bbox = x_min, y_min, x_max, y_max

    # set raw output nc folder (one nc per date)
    raw_ncs_folder = os.path.join(sat_folder, 'raw_ncs')
    if not os.path.exists(raw_ncs_folder):
        os.mkdir(raw_ncs_folder)

    # prepare downloads from raw stac features
    downloads = web.convert_stac_features_to_downloads(features=stac_features,
                                                       assets=assets,
                                                       out_bbox=out_bbox,
                                                       out_epsg=3577,
                                                       out_res=10,
                                                       out_path=raw_ncs_folder,
                                                       out_extension='.nc')

    # group downloads captured on same solar day
    downloads = web.group_downloads_by_solar_day(downloads=downloads)
    if len(downloads) == 0:
        arcpy.AddWarning('No valid downloads were found.')
        return

    # remove downloads if current month (we want complete months)
    downloads = web.remove_downloads_for_current_month(downloads)
    if len(downloads) == 0:
        arcpy.AddWarning('Not enough downloads in current month exist yet.')
        return

    exist_dates = []
    for file in os.listdir(raw_ncs_folder):
        if file != 'monthly_meds.nc' and file.endswith('.nc'):
            file = file.replace('R', '').replace('.nc', '')
            exist_dates.append(file)

    # remove downloads that already exist in sat folder
    if len(exist_dates) > 0:
        downloads = web.remove_existing_downloads(downloads, exist_dates)
        if len(downloads) == 0:
            arcpy.AddWarning('No new satellite downloads were found.')
            return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region DOWNLOAD WCS DATA

    arcpy.SetProgressor('step', 'Downloading Sentinel 2 data...', 0, len(downloads), 1)

    # set relevant download parameters
    num_cpu = int(np.ceil(mp.cpu_count() / 2))
    max_out_of_bounds = 5
    max_invalid_pixels = 0
    nodata_value = -999

    i = 0
    results = []
    with ThreadPoolExecutor(max_workers=num_cpu) as pool:
        futures = []
        for download in downloads:
            task = pool.submit(web.validate_and_download,
                               download,
                               quality_flags,
                               max_out_of_bounds,
                               max_invalid_pixels,
                               nodata_value)

            futures.append(task)

        for future in as_completed(futures):
            arcpy.AddMessage(future.result())
            results.append(future.result())

            i += 1
            if i % 1 == 0:
                arcpy.SetProgressorPosition(i)

    # check if any valid downloads (non-cloud or new)
    num_valid_downlaods = len([dl for dl in results if 'success' in dl])
    if num_valid_downlaods == 0:
        arcpy.AddMessage('No new satellite downloads were found.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region COMBINE NETCDFS INTO MONTHLY MEDIANS

    arcpy.SetProgressor('default', 'Combining Sentinel 2 data...')

    nc_files = []
    for file in os.listdir(raw_ncs_folder):
        if file.startswith('R') and file.endswith('.nc'):
            nc_files.append(os.path.join(raw_ncs_folder, file))

    if len(nc_files) == 0:
        arcpy.AddError('No NetCDF files were found.')
        raise  # return

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

    # set nodata to nan
    ds = ds.where(ds != nodata_value)

    # detect outliers via z-score, make mask, set pixel to nan when any band is outlier nan
    z_mask = xr.apply_ufunc(zscore, ds, 0)  # 0 is time dim
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

    # export combined monthly median as new netcdf
    out_nc = os.path.join(sat_folder, 'monthly_meds.nc')
    ds.to_netcdf(out_nc)
    ds.close()

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region LOAD HIGH RES CLASSIFIED DRONE IMAGE AS XR DATASET

    arcpy.SetProgressor('default', 'Preparing classified UAV data...')

    capture_folders = os.path.join(in_project_folder, 'uav_captures')
    capture_folder = os.path.join(capture_folders, meta_item['capture_folder'])
    classify_folder = os.path.join(capture_folder, 'classify')

    optimal_rf_tif = None
    for file in os.listdir(classify_folder):
        if 'rf_optimal' in file and file.endswith('.tif'):
            optimal_rf_tif = os.path.join(classify_folder, file)

    if optimal_rf_tif is None:
        arcpy.AddError('No optimal classified UAV image exists.')
        raise  # return

    # read classified uav image (take tif of best classified model)
    tmp_class_nc = os.path.join(classify_folder, 'rf_optimal.nc')
    arcpy.management.CopyRaster(in_raster=optimal_rf_tif,
                                out_rasterdataset=tmp_class_nc)

    # read it in as netcdf
    with xr.open_dataset(tmp_class_nc) as ds_hr:
        ds_hr.load()

    # prepare it for use (get 2d array, set nan to -999, int16 for speed)
    da_hr = ds_hr[['Band1']].to_array().squeeze(drop=True)
    da_hr = xr.where(~np.isnan(da_hr), da_hr, -999).astype('int16')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region GENERATE FRACTIONAL DATA

    arcpy.SetProgressor('step', 'Generating fractional data...', 0, len(ds['time']), 1)

    fractions_folder = os.path.join(capture_folder, 'fractions')
    if not os.path.exists(fractions_folder):
        os.mkdir(fractions_folder)

    # get latest capture date and minus a year for fractionals
    #y, m, d =
    #y, m, d = meta_item['capture_date'].split(' ')[0].split('-')
    #first_date = f'{str(int(y) - 1)}-{m}-01'
    first_date = '2017-01-01'
    #

    # load monthly median low-res satellite data
    with xr.open_dataset(out_nc) as ds_lr:
        ds_lr.load()

    # slice to first date up to now
    ds_lr = ds_lr.sel(time=slice(first_date, None))

    # get list of prior processed year-month folders
    exist_fractions = []
    for file in os.listdir(fractions_folder):
        fp = os.path.join(fractions_folder, file)
        if os.path.isdir(fp):
            exist_fractions.append(file)

    # iterate each date...
    for i in range(0, len(ds_lr['time'])):
        da = ds_lr.isel(time=i)
        dt = str(da['time'].dt.strftime('%Y-%m').values)

        if dt in exist_fractions:
            arcpy.AddMessage(f'Skipping fractions for date: {dt}, already exists.')
            continue

        arcpy.AddMessage(f'Generating fractions for date: {dt}.')

        fraction_folder = os.path.join(fractions_folder, dt)
        if not os.path.exists(fraction_folder):
            os.mkdir(fraction_folder)

        out_band_tifs = []
        for var in list(da.data_vars):
            out_nc = os.path.join(fraction_folder, f'{var}.nc')
            out_tif = os.path.join(fraction_folder, f'{var}.tif')

            da[var].to_netcdf(out_nc)

            dataset = gdal.Open(out_nc, gdal.GA_ReadOnly)
            dataset = gdal.Translate(out_tif, dataset)
            dataset = None

            os.remove(out_nc)
            out_band_tifs.append(out_tif)

        # create composite of bands  # memory seems to fail...
        tmp_comp = os.path.join(fraction_folder, 'tmp_comp.tif')  # r'memory\tmp_comp'
        arcpy.management.CompositeBands(in_rasters=out_band_tifs,
                                        out_raster=tmp_comp)

        # reproject it to wgs 84 zone 50s
        tmp_prj = os.path.join(fraction_folder, 'comp.tif')
        arcpy.management.ProjectRaster(in_raster=tmp_comp,
                                       out_raster=tmp_prj,
                                       out_coor_system=arcpy.SpatialReference(32750),
                                       resampling_type='CUBIC',
                                       cell_size='10 10')

        # delete original raw bands
        for band_tif in out_band_tifs:
            arcpy.management.Delete(band_tif)

        # split bands from projected raster
        desc = arcpy.Describe(tmp_prj)
        for band in [band.name for band in desc.children]:
            out_band_tif = os.path.join(fraction_folder, f'{band.lower()}.tif')
            arcpy.management.CopyRaster(in_raster=os.path.join(tmp_prj, band),
                                        out_rasterdataset=out_band_tif)

        # get a centroid point for each grid cell on sentinel raster
        tmp_pnts = r'memory\tmp_points'
        arcpy.conversion.RasterToPoint(in_raster=tmp_prj,
                                       out_point_features=tmp_pnts,
                                       raster_field='Value')

        # convert points into 10m buffer circles (5 m half cell res)
        tmp_buff = r'memory\tmp_circle_buff'
        arcpy.analysis.PairwiseBuffer(in_features=tmp_pnts,
                                      out_feature_class=tmp_buff,
                                      buffer_distance_or_field='5 Meters')

        # convert circle buffers to square rectangles
        tmp_env =r'memory\tmp_square_buff'
        arcpy.management.FeatureEnvelopeToPolygon(in_features=tmp_buff,
                                                  out_feature_class=tmp_env)

        # add required fields to envelope shapefile
        arcpy.management.AddFields(in_table=tmp_env,
                                   field_description="c_0 FLOAT;c_1 FLOAT;c_2 FLOAT;inc SHORT")

        # delete composite and projected composite
        arcpy.management.Delete(tmp_comp)
        arcpy.management.Delete(tmp_prj)

        # extract all high-res class values within each low res pixel
        with arcpy.da.UpdateCursor(tmp_env, ['c_0', 'c_1', 'c_2', 'inc', 'SHAPE@']) as cursor:
            for row in cursor:
                x_slice = slice(row[-1].extent.XMin, row[-1].extent.XMax)
                y_slice = slice(row[-1].extent.YMin, row[-1].extent.YMax)

                arr = da_hr.sel(x=x_slice, y=y_slice).values

                if arr.size != 0 and ~np.any(arr == -999):
                    classes, counts = np.unique(arr, return_counts=True)

                    classes = [f'c_{c}' for c in classes]
                    freqs = (counts / np.sum(counts)).astype('float16')

                    class_map = {
                        'c_0': 0.0,
                        'c_1': 0.0,
                        'c_2': 0.0,
                        'inc': 1
                    }

                    class_map.update(dict(zip(classes, freqs)))

                    row[0:4] = list(class_map.values())
                else:
                    row[3] = 0

                cursor.updateRow(row)

        # save training and validation rois
        tmp_rois = os.path.join(fraction_folder, 'frac_rois.shp')
        arcpy.analysis.Select(in_features=tmp_env,
                              out_feature_class=tmp_rois,
                              where_clause='inc = 1')

        # iter each class for fractional mapping...
        for classvalue in ["c_0", "c_1", "c_2"]:

            exp_vars = []
            for file in os.listdir(fraction_folder):
                if file.endswith('.tif'):
                    if 'band_' in file:
                        exp_vars.append(os.path.join(fraction_folder, file))

            class_desc = None
            if classvalue == 'c_0':
                class_desc = 'other'
            elif classvalue == 'c_1':
                class_desc = 'native'
            elif classvalue == 'c_2':
                class_desc = 'weed'

            out_tif_fn = f'frac_{dt}_{classvalue}_{class_desc}.tif'.replace('-', '_')
            out_frac_tif = os.path.join(fraction_folder, out_tif_fn)

            out_cmx_fn = f'cm_{dt}_{classvalue}_{class_desc}.dbf'.replace('-', '_')
            out_cmx_dbf = os.path.join(fraction_folder, out_cmx_fn)

            # perform regression  # FIXME: this fails if we run via PyCharm - works ok via toolbox... threading issue?
            rf_model = arcpy.stats.Forest(prediction_type='PREDICT_RASTER',
                                          in_features=tmp_rois,
                                          variable_predict=classvalue,
                                          explanatory_rasters=exp_vars,
                                          output_raster=out_frac_tif,
                                          explanatory_rasters_matching=exp_vars,
                                          number_of_trees=100,
                                          percentage_for_training=25,
                                          number_validation_runs=5,
                                          output_validation_table=out_cmx_dbf)

            out_cmx_fn = f'cm_{dt}_{classvalue}_{class_desc}.csv'.replace('-', '_')
            out_cmx_csv = os.path.join(fraction_folder, out_cmx_fn)

            # convert dbf to csv
            arcpy.conversion.ExportTable(in_table=out_cmx_dbf,
                                         out_table=out_cmx_csv)

            # drop dbf
            arcpy.management.Delete(out_cmx_dbf)

            # read csv with pandas and get average r-squares
            avg_r2 = pd.read_csv(out_cmx_csv)['R2'].mean().round(3)
            arcpy.AddMessage(f'Average R2 for {classvalue} ({class_desc}): {str(avg_r2)}')

        # drop temporary rois and projected composite raster
        arcpy.management.Delete(tmp_rois)

        # delete remaining bands
        for band in exp_vars:
            arcpy.management.Delete(band)

    # TODO: check folders and if any missing files, delete it?

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
#execute(None)