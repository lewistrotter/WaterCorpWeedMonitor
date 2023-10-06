
def execute(
        parameters
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region IMPORTS

    import os
    import json
    import warnings
    import numpy as np
    import xarray as xr
    import arcpy

    from scripts import uav_fractions, web, shared

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region WARNINGS

    # disable warnings
    warnings.filterwarnings('ignore')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    in_project_file = parameters[0].valueAsText
    in_flight_datetime = parameters[1].value

    # inputs for testing only
    #in_project_file = r'C:\Users\Lewis\Desktop\testing\lancelin\meta.json'
    #in_flight_datetime = '2023-09-07 11:00:00'

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE ENVIRONMENT

    arcpy.SetProgressor('default', 'Preparing environment...')

    # check if advanced license is available
    if arcpy.CheckProduct('ArcInfo') not in ['AlreadyInitialized', 'Available']:
        arcpy.AddError('Advanced ArcGIS Pro license is unavailable.')
        return

    # check if user has spatial/image analyst, error if not
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
    sub_folders = ['boundary', 'grid', 'uav_captures', 'sat_captures', 'visualise']
    for sub_folder in sub_folders:
        sub_folder = os.path.join(in_project_folder, sub_folder)
        if not os.path.exists(sub_folder):
            arcpy.AddError('Project is missing required folders.')
            return

    # check if uav grid file exists
    grid_tif = os.path.join(in_project_folder, 'grid', 'grid.tif')
    if not os.path.exists(grid_tif):
        arcpy.AddError('Project grid file does not exist.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CREATE AND SET WORKSPACE TO TEMPORARY FOLDER

    arcpy.SetProgressor('default', 'Preparing workspace...')

    # create temp folder if does not already exist
    tmp = os.path.join(in_project_folder, 'tmp')
    if not os.path.exists(tmp):
        os.mkdir(tmp)

    # clear temp folder (errors skipped)
    shared.clear_tmp_folder(tmp_folder=tmp)

    # set temp folder to arcpy workspace
    arcpy.env.workspace = tmp

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
        arcpy.AddError('Project has no rehab start date.')
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
        start_date, end_date = '2016-01-01', '2039-12-31'

        # read grid as raster
        tmp_grd = arcpy.Raster(grid_tif)

        # get stac and output coordinate bbox based on grid exent
        stac_bbox = shared.get_raster_bbox(in_ras=tmp_grd, out_epsg=4326)

        # get output netcdf bbox in albers and expand
        out_bbox = shared.get_raster_bbox(in_ras=tmp_grd, out_epsg=3577)
        out_bbox = shared.expand_bbox(bbox=out_bbox, by_metres=50.0)

        # set output folder for raw sentinel 2 cubes and check
        raw_ncs_folder = os.path.join(sat_folder, 'raw_ncs')
        if not os.path.exists(raw_ncs_folder):
            os.mkdir(raw_ncs_folder)

        # query and prepare downloads
        downloads = web.quick_fetch(start_date=start_date,
                                    end_date=end_date,
                                    stac_bbox=stac_bbox,
                                    out_bbox=out_bbox,
                                    out_folder=raw_ncs_folder)

    except Exception as e:
        arcpy.AddError('Unable to download Sentinel 2 data from DEA. See messages.')
        arcpy.AddMessage(str(e))
        return

    # check if downloads returned (should always find something), else leave
    if len(downloads) == 0:
        arcpy.AddError('No valid satellite downloads found. Check your firewall.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region DOWNLOAD WCS DATA

    arcpy.SetProgressor('default', 'Downloading Sentinel 2 data...')

    try:
        # download everything and return success or fail statuses
        results = web.quick_download(downloads=downloads,
                                     quality_flags=[1],
                                     max_out_of_bounds=30,
                                     max_invalid_pixels=30,
                                     nodata_value=-999)

    except Exception as e:
        arcpy.AddError('Unable to download Sentinel 2 data from DEA. See messages.')
        arcpy.AddMessage(str(e))
        return

    # count number of valid downloads returned
    num_valid_downloads = len([dl for dl in results if 'success' in dl])

    # check if any valid downloads (non-cloud or new)
    new_downloads = True
    if num_valid_downloads == 0:
        arcpy.AddMessage('No new valid satellite downloads were found. Checking fractions.')
        new_downloads = False

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region VALIDATE SENTINEL 2 NETCDFS

    arcpy.SetProgressor('default', 'Validating Sentinel 2 data...')

    try:
        # check results for errors and delete errorneous nc files
        if new_downloads:
            web.delete_error_downloads(results=results,
                                       nc_folder=raw_ncs_folder)

    except Exception as e:
        arcpy.AddError('Unable to delete errorneous Sentinel 2 data. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region COMBINE SENTINEL 2 NETCDFS

    arcpy.SetProgressor('default', 'Combining Sentinel 2 data...')

    # get all raw nc dates in raw netcdf folder
    nc_files = []
    for file in os.listdir(raw_ncs_folder):
        if file.startswith('R') and file.endswith('.nc'):
            nc_files.append(os.path.join(raw_ncs_folder, file))

    # check if anything came back, error if not
    if len(nc_files) == 0:
        arcpy.AddError('No NetCDF files were found.')
        return

    try:
        # read all netcdfs into single dataset
        ds = shared.concat_netcdf_files(nc_files=nc_files)

    except Exception as e:
        arcpy.AddError('Unable to combine Sentinel 2 NetCDFs. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CLEAN SENTINEL 2 NETCDFS

    arcpy.SetProgressor('default', 'Cleaning Sentinel 2 data...')

    try:
        # extract netcdf attributes
        ds_attrs = ds.attrs
        ds_band_attrs = ds[list(ds)[0]].attrs
        ds_spatial_ref_attrs = ds['spatial_ref'].attrs

        # set nodata (-999) to nan
        ds = ds.where(ds != -999)

        # set pixel to nan when any band has outlier within pval 0.0001 (z = 4) per date
        ds = uav_fractions.remove_xr_outliers(ds=ds,
                                              max_z_value=4.0)

        # resample to monthly medians and interpolate nan
        ds = uav_fractions.resample_xr_monthly_medians(ds=ds)

        # fill in nan values
        ds = ds.interpolate_na(dim='time',
                               method='linear',
                               fill_value='extrapolate')

        # append attributes back on
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds:
            ds[var].attrs = ds_band_attrs

    except Exception as e:
        arcpy.AddError('Unable to clean Sentinel 2 NetCDFs. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXPORT CLEAN SENTINEL 2 NETCDF

    arcpy.SetProgressor('default', 'Exporting clean Sentinel 2 data...')

    # set combined output nc folder
    cmb_ncs_folder = os.path.join(sat_folder, 'cmb_ncs')
    if not os.path.exists(cmb_ncs_folder):
        os.mkdir(cmb_ncs_folder)

    try:
        # export combined monthly median as new netcdf
        out_montly_meds_nc = os.path.join(cmb_ncs_folder, 'raw_monthly_meds.nc')
        ds.to_netcdf(out_montly_meds_nc)
        ds.close()

    except Exception as e:
        arcpy.AddError('Unable to export clean Sentinel 2 NetCDF. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ HIGH RES CLASSIFIED DRONE IMAGE AS XR DATASET

    arcpy.SetProgressor('default', 'Reading high-resolution classified UAV data...')

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
        # read classified uav image as xr (save netcdf to tmp)
        tmp_cls_nc = os.path.join(tmp, 'rf_optimal.nc')
        ds_hr = shared.raster_to_xr(in_ras=optimal_rf_tif,
                                    out_nc=tmp_cls_nc,
                                    epsg=3577,
                                    datetime=None,
                                    var_names=None,
                                    dtype='float32')

        # convert to array and clean it up for efficiency
        da_hr = ds_hr[['Band1']].to_array().squeeze(drop=True)
        da_hr = xr.where(~np.isnan(da_hr), da_hr, -999).astype('int16')

    except Exception as e:
        arcpy.AddError('Could not read classified UAV image. See messages.')
        arcpy.AddMessage(str(e))
        return

    # check if classified uav has the four classes
    for cls in np.unique(da_hr):
        if cls not in [-999, 0, 1, 2]:
            raise ValueError('High-resolution xr can only have classes 0, 1, 2.')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ LOW RES SENTINEL 2 MEDIAN XR DATASET

    arcpy.SetProgressor('default', 'Reading low-resolution Sentinel 2 data...')

    try:
        # load monthly median low-res satellite data
        with xr.open_dataset(out_montly_meds_nc) as ds_lr:
            ds_lr.load()

        # slice from 2016-01-01 up to now
        ds_lr = ds_lr.sel(time=slice('2016-01-01', None))

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
        # generate roi point shapefile with frequencies
        tmp_roi = os.path.join(tmp, 'tmp_roi.shp')
        uav_fractions.create_roi_freq_points(da_hr=da_hr,
                                             ds_lr=ds_lr,
                                             out_shp=tmp_roi)

    except Exception as e:
        arcpy.AddError('Could not create training areas. See messages.')
        arcpy.AddMessage(str(e))
        return

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

            # notify user of date
            arcpy.AddMessage(f'Working on date: {dt}')

            # set new folder named year-month
            fraction_folder = os.path.join(fractions_folder, dt)
            if not os.path.exists(fraction_folder):
                os.mkdir(fraction_folder)

            # delete previous band fields, extract new ones
            uav_fractions.extract_xr_to_roi_points(roi_shp=tmp_roi,
                                                   in_ds=da)

            # iter each class...
            frac_ncs = []
            classes = {'c_0': 'other', 'c_1': 'native', 'c_2': 'weed'}
            for val, dsc in classes.items():

                # perform gwr via rois shapefile
                tmp_gwr_csv = os.path.join(fraction_folder, f'acc_{dsc}.csv')
                uav_fractions.gwr(in_rois=tmp_roi,
                                  classvalue=val,
                                  classdesc=dsc,
                                  out_prediction_shp='tmp_gwr.shp',
                                  out_accuracy_csv=tmp_gwr_csv)

                # enable this, disable gwr to switch between
                # uav_fractions.old_regress(in_rois=tmp_roi,
                #                           classvalue=val,
                #                           classdesc=dsc,
                #                           out_regress_shp='tmp_gwr.shp',
                #                           out_accuracy_csv=tmp_gwr_csv)

                # force prediction values to 0 - 1
                uav_fractions.force_pred_zero_to_one(in_shp='tmp_gwr.shp')

                # convert roi points to 5x5m pixels and resample to 2.5m
                out_frc_tif = os.path.join(fraction_folder, f'{dt}_{dsc}.tif'.replace('-', '_'))
                uav_fractions.roi_points_to_raster(in_shp='tmp_gwr.shp',
                                                   out_ras=out_frc_tif)

                # create a netcdf version of frac tif
                tmp_frc_nc = os.path.join(tmp, f'tmp_{dsc}.nc')
                shared.raster_to_xr(in_ras=out_frc_tif,
                                    out_nc=tmp_frc_nc,
                                    epsg=3577,
                                    datetime=dt + '-' + '01',
                                    var_names=[dsc],
                                    dtype='float64')

                # append netcdf to list
                frac_ncs.append(tmp_frc_nc)

            # check we have three fraction netcdfs
            if len(frac_ncs) != 3:
                arcpy.AddError('Could not generate all three fraction layers.')
                return

            # merge three fraction datasets
            ds_frc = shared.merge_netcdf_files(nc_files=frac_ncs)

            # export fraction dataset
            ds_frc.to_netcdf(os.path.join(fraction_folder, f'frc_{dt}.nc'))
            ds_frc.close()

            # add successful fractional date to metadata
            meta_item['fractions'].append(dt)

            # increment progressor
            arcpy.SetProgressorPosition()

    except Exception as e:
        arcpy.AddError('Could not generate fractional map. See messages.')
        arcpy.AddMessage(str(e))
        return

    # reset progressor
    arcpy.ResetProgressor()

    # endregion

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
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region ADD FRACTION LAYERS TO ACTIVE MAP

    arcpy.SetProgressor('default', 'Visualising result...')

    # build visualise folder path and
    visualise_folder = os.path.join(in_project_folder, 'visualise')

    try:
        # get full fraction dataset
        ds_list = []
        for root, dirs, files in os.walk(fractions_folder):
            for file in files:
                if 'frc' in file and file.endswith('.nc'):
                    ds_list.append(os.path.join(root, file))

        # concatenate all fractional netcdfs by time
        ds = shared.concat_netcdf_files(nc_files=ds_list)

        # get flight date code
        flight_date = meta_item['capture_folder']

        # convert to a crf for each fractional variable
        for var in ds:
            # export current var
            tmp_frc_nc = os.path.join(tmp, f'frc_{var}.nc')
            ds[[var]].to_netcdf(tmp_frc_nc)

            # convert to crf
            out_crf = os.path.join(visualise_folder, f'frc_{var}_{flight_date}.crf')
            shared.netcdf_to_crf(in_nc=tmp_frc_nc,
                                 out_crf=out_crf)

            # add crf to map
            shared.add_raster_to_map(in_ras=out_crf)

    except Exception as e:
        arcpy.AddWarning('Could not visualise classified image. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    arcpy.SetProgressor('default', 'Cleaning up environment...')

    try:
        # close temp files
        del tmp_grd

        # close netcdfs
        ds.close()

    except Exception as e:
        arcpy.AddWarning('Could not drop temporary files. See messages.')
        arcpy.AddMessage(str(e))

    # clear temp folder (errors skipped)
    shared.clear_tmp_folder(tmp_folder=tmp)

    # free up spatial analyst and image analyst
    arcpy.CheckInExtension('Spatial')
    arcpy.CheckInExtension('ImageAnalyst')

    # set changed env variables back to default
    arcpy.env.overwriteOutput = False
    arcpy.env.addOutputsToMap = True

    # endregion

    return

# testing
#execute(None)