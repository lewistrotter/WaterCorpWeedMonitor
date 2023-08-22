
def execute(
        parameters
        # messages # TODO: implement
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region IMPORTS

    import os
    import json
    import datetime
    import numpy as np
    import arcpy

    from scripts import trends, shared

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    in_project_file = parameters[0].valueAsText
    in_flight_datetime = parameters[1].value
    in_rehab_or_capture_month = parameters[2].value

    # inputs for testing only
    # in_project_file = r'C:\Users\Lewis\Desktop\testing\city beach dev\meta.json'
    # in_flight_datetime = '2023-02-02 16:29:52'
    # in_rehab_or_capture_month = 'Month of Rehabilitation'

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
            arcpy.AddError('Project folder is missing expected folders.')
            return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CREATE AND SET WORKSPACE TO TEMPORARY FOLDER

    arcpy.SetProgressor('default', 'Preparing workspace...')

    # create temp folder, if does not already exist
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
    # region READ AND CHECK FRACTION NETCDFS

    arcpy.SetProgressor('default', 'Reading and checking class fraction NetCDFs...')

    # get captures and fractions folder
    capture_folders = os.path.join(in_project_folder, 'uav_captures')
    capture_folder = os.path.join(capture_folders, meta_item['capture_folder'])
    fraction_folder = os.path.join(capture_folder, 'fractions')

    # check fraction folder exists
    if not os.path.exists(fraction_folder):
        arcpy.AddError('No fraction folder detected. Run fraction tool first.')
        return

    # get list of prior processed year-month fractional folders
    valid_fractions = meta_item.get('fractions')
    if valid_fractions is None:
        arcpy.AddError('No fraction list detected.')
        return

    # get paths of all netcdfs
    ncs = []
    for root, dirs, files in os.walk(fraction_folder):
        for file in files:
            if file.endswith('.nc'):
                ncs.append(os.path.join(root, file))

    # remove any fractions not in valid list
    clean_ncs = []
    for nc in ncs:
        folder = os.path.basename(os.path.dirname(nc))
        if folder in valid_fractions:
            clean_ncs.append(nc)

    # ensure netcdfs were found, else abort
    if len(clean_ncs) == 0:
        arcpy.AddError('No valid fractions NetCDFs returned.')
        return

    try:
        # read and concatnate all netcdfs into one
        ds = shared.concat_netcdf_files(clean_ncs)

    except Exception as e:
        arcpy.AddError('Could not read all fraction NetCDFs. See messages.')
        arcpy.AddMessage(str(e))
        return

    # check at least 3 years exist, else abort
    num_years = len(np.unique(ds['time.year']))
    if num_years < 3:
        arcpy.AddError('Must have >= 3 years worth of data in NetCDF.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region SUBSET NETCDF TO MONTH OF CAPTURE

    arcpy.SetProgressor('default', 'Subsetting NetCDF to year and month...')

    # get rehab year and month
    rehab_year = int(rehab_start_date.split('-')[0])
    rehab_month = int(rehab_start_date.split('-')[1])

    # correct rehab year if < 2016
    if rehab_year < 2016:
        arcpy.AddWarning('Start of rehab < Sentinel 2 data start, using 2016.')
        rehab_year = 2016

    # get capture year and month
    capture_month = int(in_flight_datetime.split('-')[1])

    # set start year and month based on selection
    year, month = None, None
    if in_rehab_or_capture_month == 'Month of Rehabilitation':
        year, month = rehab_year, rehab_month
    elif in_rehab_or_capture_month == 'Month of First UAV Capture':
        year, month = rehab_year, capture_month

    try:
        # extract specific month from dataset
        dts = ds['time.month'].isin(month)
        ds = ds.sel(time=dts)

        # extract specifc years from dataset (up to 2039)
        valid_years = list(range(year, 2039 + 1))
        yrs = ds['time.year'].isin(valid_years)
        ds = ds.sel(time=yrs)

        # ensure it is in order of time
        ds = ds.sortby('time')

    except Exception as e:
        arcpy.AddError('Could not subset NetCDF to month. See messages.')
        arcpy.AddMessage(str(e))
        return

    # check at least 3 years exist again, else abort
    num_years = len(np.unique(ds['time.year']))
    if num_years < 3:
        arcpy.AddError('Must have >= 3 years worth of data in NetCDF.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region GENERATE TREND RGBS

    # create trends folder if not exists
    trends_folder = os.path.join(capture_folder, 'trends')
    if not os.path.exists(trends_folder):
        os.mkdir(trends_folder)

    # set up step-wise progressor
    arcpy.SetProgressor('step', None, 0, len(ds.data_vars))

    try:
        # iter each fraction var
        trend_rgbs = {}
        for var in ds:
            # extract current var
            da = ds[[var]]

            # TODO: enable this when testing to check trends
            #da.to_netcdf(os.path.join(tmp, f'ts_{var}.nc'))

            # create trend rgb xr
            ds_tnd = trends.generate_trend_rgb_xr(ds=da, var_name=var)

            # convert trend vars to rgb rasters
            tmp_rgb = shared.multi_band_xr_to_raster(da=ds_tnd,
                                                     out_folder=tmp)

            # save trend rgb to trend folder
            out_rgb = os.path.join(trends_folder, f'rgb_{var}.tif')
            tmp_rgb.save(out_rgb)

            # # add result to dictionary
            trend_rgbs[var] = out_rgb

            # increment progressor
            arcpy.SetProgressorPosition()

    except Exception as e:
        arcpy.AddError('Could not calculate trend. See messages.')
        arcpy.AddMessage(str(e))
        return

    # reset progressor
    arcpy.ResetProgressor()

    # ensure we have three trend rgb composites
    if len(trend_rgbs) != 3:
        arcpy.AddError('Could not generate three trend composites.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region ADD TREND RGB LAYERS TO ACTIVE MAP

    arcpy.SetProgressor('default', 'Visualising result...')

    # build visualise folder path and
    visualise_folder = os.path.join(in_project_folder, 'visualise')

    # get flight date code
    flight_date = meta_item['capture_folder']

    try:
        # iter trend rgb composites
        for var, path in trend_rgbs.items():

            # save trend rgb to visualise folder
            out_tif = os.path.join(visualise_folder, f'trend_rgb_{var}' + '_' + flight_date + '.tif')
            arcpy.management.CopyRaster(in_raster=path,
                                        out_rasterdataset=out_tif)

            # visualise it on active map and symbolise it to class colors
            shared.add_raster_to_map(in_ras=out_tif)

    except Exception as e:
        arcpy.AddWarning('Could not visualise Trend RGB image. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    arcpy.SetProgressor('default', 'Cleaning up environment...')

    try:
        # drop temp files (free up space)
        del tmp_rgb

    except Exception as e:
        arcpy.AddWarning('Could not drop temporary files. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # clear temp folder (errors skipped)
    shared.clear_tmp_folder(tmp_folder=tmp)  # TODO: disable when testing

    # free up spatial analyst
    arcpy.CheckInExtension('Spatial')
    arcpy.CheckInExtension('ImageAnalyst')

    # set changed env variables back to default
    arcpy.env.overwriteOutput = False
    arcpy.env.addOutputsToMap = True

    # endregion

    return

# testing
#execute(None)