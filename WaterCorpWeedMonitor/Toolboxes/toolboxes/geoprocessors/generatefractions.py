
def execute(
        parameters
        # messages # TODO: implement
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region IMPORTS

    import os
    import json
    import numpy as np
    import xarray as xr
    import multiprocessing as mp
    import arcpy

    from scipy.stats import zscore
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import as_completed

    from scripts import web, uav_fractions, shared

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    # in_project_file = parameters[0].valueAsText
    # in_flight_datetime = parameters[1].value

    # inputs for testing only
    in_project_file = r'C:\Users\Lewis\Desktop\testing\city beach dev\meta.json'
    in_flight_datetime = '2023-06-09 13:03:39'

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE ENVIRONMENT

    arcpy.SetProgressor('default', 'Preparing environment...')

    # check if user has spatial analyst, error if not
    if arcpy.CheckExtension('Spatial') != 'Available':
        arcpy.AddError('Spatial Analyst license is unavailable.')
        return
    elif arcpy.CheckExtension('ImageAnalyst') != 'Available':
        arcpy.AddError('Image Analyst license is unavailable.')
        return
    else:
        arcpy.CheckOutExtension('Spatial')
        arcpy.CheckOutExtension('ImageAnalyst')

    # set data overwrites and mapping
    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = False

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK PROJECT FOLDER STRUCTURE AND FILES

    arcpy.SetProgressor('default', 'Checking project folders and files...')

    # check if input project file exists
    if not os.path.exists(in_project_file):
        arcpy.AddError('Project file does not exist.')
        return

    # get top-level project folder from project file
    in_project_folder = os.path.dirname(in_project_file)

    # check if required project folders already exist, error if so
    sub_folders = ['grid', 'uav_captures', 'sat_captures', 'visualise']
    for sub_folder in sub_folders:
        sub_folder = os.path.join(in_project_folder, sub_folder)
        if not os.path.exists(sub_folder):
            arcpy.AddError('Project folder is missing expected folders.')
            return

    # check if uav grid file exists
    grid_tif = os.path.join(in_project_folder, 'grid', 'grid.tif')
    if not os.path.exists(grid_tif):
        arcpy.AddError('UAV grid raster does not exist.')
        return

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
        return

    # check if any captures exist (will be >= 4)
    if len(meta) < 4:
        arcpy.AddError('Project has no UAV capture data.')
        return

    # check and get start of rehab date
    rehab_start_date = meta.get('date_rehab')
    if rehab_start_date is None:
        arcpy.AddError('Project has no start of rehab date.')
        return

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
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region FETCH CLEAN DEA STAC SENTINEL 2 DOWNLOADS

    arcpy.SetProgressor('default', 'Fetching clean DEA STAC downloads...')

    # set sat folder for nc outputs
    sat_folder = os.path.join(in_project_folder, 'sat_captures')

    try:
        # set query date range, collections and assets
        start_date, end_date = '2017-01-01', '2039-12-31'

        # read grid as raster
        tmp_grd = arcpy.Raster(grid_tif)

        # get stac and output coordinate bbox based on grid exent
        stac_bbox = shared.get_raster_bbox(in_ras=tmp_grd, out_epsg=4326)

        # get output netcdf bbox in albers and expand
        out_bbox = shared.get_raster_bbox(in_ras=tmp_grd, out_epsg=3577)
        out_bbox = shared.expand_bbox(bbox=out_bbox, by_metres=30.0)

        # set output folder for raw sentinel 2 cubes and check
        out_raw_folder = os.path.join(sat_folder, 'raw_ncs')
        if not os.path.exists(out_raw_folder):
            os.mkdir(out_raw_folder)

        # query and prepare downloads
        downloads = web.quick_fetch(start_date=start_date,
                                    end_date=end_date,
                                    stac_bbox=stac_bbox,
                                    out_bbox=out_bbox,
                                    out_folder=out_raw_folder)

    except Exception as e:
        arcpy.AddError('Unable to download Sentinel 2 data from DEA. See messages.')
        arcpy.AddMessage(str(e))
        return

    # check if downloads returned, else leave
    if len(downloads) == 0:
        arcpy.AddWarning('No valid satellite downloads were found.')
        return  # TODO: carry on in case fractionals remain unprocessed?

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region DOWNLOAD WCS DATA

    arcpy.SetProgressor('step', 'Downloading Sentinel 2 data...', 0, len(downloads), 1)

    # set relevant download parameters
    num_cpu = int(np.ceil(mp.cpu_count() / 2))

    # TODO: move to quick fetch func and remove this block if progressor works

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
        return

    # check if any valid downloads (non-cloud or new)
    num_valid_downlaods = len([dl for dl in results if 'success' in dl])
    if num_valid_downlaods == 0:
        arcpy.AddMessage('No new valid satellite downloads were found.')
        return

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
        return

    # TODO: move this to isolated function

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

        # TODO: back and forward fill

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
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region LOAD HIGH RES CLASSIFIED DRONE IMAGE AS XR DATASET

    arcpy.SetProgressor('default', 'Preparing high-resolution classified UAV data...')

    # set up capture and classify folders
    capture_folders = os.path.join(in_project_folder, 'uav_captures')
    capture_folder = os.path.join(capture_folders, meta_item['capture_folder'])
    classify_folder = os.path.join(capture_folder, 'classify')

    # create optimal classified rf path
    optimal_rf_tif = os.path.join(classify_folder, 'rf_optimal.tif')

    # check if optimal rf exists, error if not
    if not os.path.exists(optimal_rf_tif):
        arcpy.AddError('No optimal classified UAV image exists.')
        return

    try:
        # read classified uav image (take tif of best classified model) into scratch
        tmp_class_nc = os.path.join(arcpy.env.scratchFolder, 'rf_optimal.nc')
        arcpy.md.RasterToNetCDF(in_raster=optimal_rf_tif,
                                out_netCDF_file=tmp_class_nc,
                                variable='Band1',
                                x_dimension='x',
                                y_dimension='y')

        # read it in as netcdf
        with xr.open_dataset(tmp_class_nc) as ds_hr:
            ds_hr.load()

        # prepare it for use (get 2d array, set nan to -999, int16 for speed)
        da_hr = ds_hr[['Band1']].to_array().squeeze(drop=True)
        da_hr = xr.where(~np.isnan(da_hr), da_hr, -999).astype('int16')

    except Exception as e:
        arcpy.AddError('Could not load classified UAV image. See messages.')
        arcpy.AddMessage(str(e))
        return

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
        return

    # check if any time slices exist
    if len(ds_lr['time']) == 0:
        arcpy.AddError('No time slices in Sentinel 2 data.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region BUILD ROI WINDOWS FROM SENTINEL 2 GRID PIXELS

    arcpy.SetProgressor('default', 'Creating training areas...')

    try:
        # get an all-time max of sentinel 2 cube to remove nulls
        tmp_da = ds_lr.max('time', keep_attrs=True)

        # export temp max netcdf to scratch
        tmp_max_nc = os.path.join(arcpy.env.scratchFolder, 'tmp_max.nc')
        tmp_da.to_netcdf(tmp_max_nc)

        # convert temporary netcdf to a crf
        tmp_max_crf = os.path.join(arcpy.env.scratchFolder, 'tmp_max.crf')
        arcpy.management.CopyRaster(in_raster=tmp_max_nc,
                                    out_rasterdataset=tmp_max_crf)

        # read temp crf in as reproject to 32750
        tmp_max_cmp = arcpy.Raster(tmp_max_crf)
        tmp_max_prj = arcpy.ia.Reproject(tmp_max_cmp, {'wkid': 32750})

        # create grid of 10 m rois from crf pixels in scratch
        tmp_rois = os.path.join(arcpy.env.scratchFolder, 'tmp_roi.shp')
        tmp_rois = uav_fractions.build_rois_from_raster(in_ras=tmp_max_prj,
                                                        out_rois=tmp_rois)

    except Exception as e:
        arcpy.AddError('Could not create training areas. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE CLASS FREQUENCIES PER ROI

    arcpy.SetProgressor('default', 'Calculating class fractions in training areas...')

    try:
        # calculate freq of high-res class pixels per sentinel 2 roi window
        tmp_rois = uav_fractions.calculate_roi_freqs(rois=tmp_rois,
                                                     da_hr=da_hr)

        # subset to valid rois (i.e., not all nans) only and save shapefile
        rois = os.path.join(arcpy.env.scratchFolder, 'rois.shp')
        arcpy.analysis.Select(in_features=tmp_rois,
                              out_feature_class=rois,
                              where_clause='inc = 1')

    except Exception as e:
        arcpy.AddError('Could not calculate class fractions. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region GENERATE RANDOM SAMPLES IN CLASS FRACTION ROIS

    arcpy.SetProgressor('default', 'Generating random samples...')

    # create random samples in roi polygons and save to scratch
    tmp_rnd_pnt = r'memory\rnd_pnt'
    arcpy.management.CreateRandomPoints(out_path=r'memory',
                                        out_name='rnd_pnt',
                                        constraining_feature_class=rois,
                                        number_of_points_or_field=2)

    # extract class values from rois per point
    tmp_smp = r'memory\tmp_smp'
    arcpy.analysis.PairwiseIntersect(in_features=[tmp_rnd_pnt, rois],
                                     out_feature_class=tmp_smp,
                                     output_type='POINT')

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
        return

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

            # notify of date currently processing
            arcpy.AddMessage(f'Generating fractions for date: {dt}.')

            # set new folder named year-month
            fraction_folder = os.path.join(fractions_folder, dt)
            if not os.path.exists(fraction_folder):
                os.mkdir(fraction_folder)

            # convert dataset to multiband composite raster and reproject it
            tmp_s2_cmp = shared.convert_xr_vars_to_raster(da=da)
            tmp_s2_prj = arcpy.ia.Reproject(tmp_s2_cmp, {'wkid': 32750})

            # save it to scratch (regress will fail otherwise) and read it
            tmp_exp_vars = os.path.join(arcpy.env.scratchFolder, 'tmp_exp_vars.tif')
            tmp_s2_prj.save(tmp_exp_vars)

            # iter each class for fractional mapping...
            vals = ['c_0', 'c_1', 'c_2']
            descs = ['other', 'native', 'weed']
            for classvalue, classdesc in zip(vals, descs):

                # notify of class currently processing
                arcpy.AddMessage(f'> Working on class: {classdesc}.')

                # create output regression tif and cmatrix
                out_fn = f'{dt}_{classvalue}_{classdesc}'.replace('-', '_')
                out_frc_tif = os.path.join(fraction_folder, 'frac_' + out_fn + '.tif')

                # perform regression modelling and prediction
                ras_reg = uav_fractions.regress(exp_vars=tmp_exp_vars,
                                                sample_points=tmp_smp,
                                                classvalue=classvalue,
                                                classdesc=classdesc)

                # apply cubic resampling to smooth pixels out
                ras_rsp = arcpy.sa.Resample(raster=ras_reg,
                                            resampling_type='Cubic',
                                            input_cellsize=10,
                                            output_cellsize=2.5)

                # save regression prediction
                ras_rsp.save(out_frc_tif)

            # TODO: combine 3 bands into one netcdf for easier use later
            # TODO:

            # delete temp comp projected comp
            del tmp_s2_cmp
            del tmp_s2_prj
            del tmp_exp_vars
            del ras_reg

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
execute(None)