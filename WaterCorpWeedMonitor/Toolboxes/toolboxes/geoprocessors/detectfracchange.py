
def execute(
        parameters
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region IMPORTS

    import os
    import json
    import datetime
    import warnings
    import numpy as np
    import arcpy

    from scripts import change, shared

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
    in_flight_datetime = parameters[1].valueAsText
    in_from_year = parameters[2].valueAsText
    in_manual_from_year = parameters[3].value
    in_to_year = parameters[4].valueAsText
    in_manual_to_year = parameters[5].value
    in_month = parameters[6].valueAsText
    in_manual_month = parameters[7].value
    in_z = parameters[8].value

    # inputs for testing only
    # in_project_file = r'C:\Users\Lewis\Desktop\testing\goegrup\meta.json'
    # in_flight_datetime = '2022-12-15 11:00:00'
    # in_from_year = 'Manual'  # 'Year of Rehabilitation'  # 'Manual'
    # in_manual_from_year = 2016
    # in_to_year = 'Current Year'  # 'Manual'
    # in_manual_to_year = 2023
    # in_month = 'Manual'  #'Month of Rehabilitation'  # 'Manual'
    # in_manual_month = 3 # 6
    # in_z = 2

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

    # check if any captures exist (will be >= 6)
    if len(meta) < 6:
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
    exclude_keys = ['project_name', 'date_created', 'date_rehab', 'sat_shift_x', 'sat_shift_y']

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
    # region PREPARE FRACTIONAL RASTER DATA

    arcpy.SetProgressor('default', 'Reading and checking class fraction folders...')

    # get captures and fractions folder
    capture_folders = os.path.join(in_project_folder, 'uav_captures')
    capture_folder = os.path.join(capture_folders, meta_item['capture_folder'])
    fraction_folder = os.path.join(capture_folder, 'fractions')

    # check fraction folder exists
    if not os.path.exists(fraction_folder):
        arcpy.AddError('No fraction folder detected. Run fraction tool first.')
        return

    # get list of prior processed year-month fractional folders
    valid_frac_dates = meta_item.get('fractions')
    if valid_frac_dates is None:
        arcpy.AddError('No fraction data found. Run fraction tool first.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE FROM AND TO DATES

    arcpy.SetProgressor('default', 'Preparing "from" and "to" dates...')

    # get rehab year
    rehab_year = int(rehab_start_date.split('-')[0])

    # correct rehab year if < 2016
    if rehab_year < 2016:
        rehab_year = 2016
        if in_from_year == 'Year of Rehabilitation':
            arcpy.AddWarning('Start of rehab < Sentinel 2 data start, using 2016.')

    # set "from" year based on selection
    from_year = None
    if in_from_year == 'Year of Rehabilitation':
        from_year = rehab_year
    elif in_from_year == 'Manual':
        from_year = in_manual_from_year

    # set "to" year based on selection
    to_year = None
    if in_to_year == 'Current Year':
        to_year = datetime.datetime.now().year
    elif in_to_year == 'Manual':
        to_year = in_manual_to_year

    # get month of rehab and capture
    rehab_month = int(rehab_start_date.split('-')[1])
    capture_month = int(in_flight_datetime.split('-')[1])

    # set month based on selection
    month = None
    if in_month == 'Month of Rehabilitation':
        month = rehab_month
    elif in_month == 'Month of First UAV Capture':
        month = capture_month
    elif in_month == 'Manual':
        month = in_manual_month

    try:
        # validate if dates exist and attempt fix if bad. error if cant
        result = change.validate_frac_dates(dates=valid_frac_dates,
                                            from_year=from_year,
                                            to_year=to_year,
                                            month=month)

        # unpack clean date values
        from_year, to_year, month = result

    except Exception as e:
        arcpy.AddError('Could not obtain dates from fraction NetCDFs. See messages.')
        arcpy.AddMessage(str(e))
        return

    # check "from" and "to" have a year difference
    if from_year >= to_year:
        arcpy.AddError('Need at least a year between "from" and "to" images.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE FROM AND TO RASTERS

    arcpy.SetProgressor('default', 'Preparing "from" and "to" rasters...')

    # build from and to date strings
    from_date = f'{from_year}-{str(month).zfill(2)}'
    to_date = f'{to_year}-{str(month).zfill(2)}'

    # tell user final dates used
    arcpy.AddMessage(f'Detecting change from {from_date} to {to_date}.')

    try:
        # get nearest dates to left, right of "from" date
        from_dates_l_r = change.get_closest_dates(dates=valid_frac_dates,
                                                  focus_date=from_date)

        # get nearest dates to left, right of "to" date
        to_dates_l_r = change.get_closest_dates(dates=valid_frac_dates,
                                                focus_date=to_date)

    except Exception as e:
        arcpy.AddError('Left, right dates from fraction NetCDFs missing. See messages.')
        arcpy.AddMessage(str(e))
        return

    # create from and to date lists
    from_dates = [from_date] + from_dates_l_r
    to_dates = [to_date] + to_dates_l_r

    # check all returned dates have fraction folders
    for date in from_dates + to_dates:
        folder = os.path.join(fraction_folder, date)
        if not os.path.exists(folder):
            raise ValueError(f'Fraction folder {date} missing.')

    # create fraction "from", "to" maps
    from_map, to_map = {}, {}

    try:
        # construct average "from" rasters per class
        for cls in ['other', 'native', 'weed']:
            # build mean raster from all dates
            tmp_from_avg = f'tmp_from_mean_{cls}.tif'
            change.get_mean_frac_raster(dates=from_dates,
                                        frac_class=cls,
                                        frac_folder=fraction_folder,
                                        out_ras=tmp_from_avg)

            # add raster path to map
            from_map[cls] = tmp_from_avg

            # build mean raster from all dates
            tmp_to_avg = f'tmp_to_mean_{cls}.tif'
            change.get_mean_frac_raster(dates=to_dates,
                                        frac_class=cls,
                                        frac_folder=fraction_folder,
                                        out_ras=tmp_to_avg)

            # add raster path to map
            to_map[cls] = tmp_to_avg

    except Exception as e:
        arcpy.AddError('Could not create mean fraction raster. See messages.')
        arcpy.AddMessage(str(e))
        return

    # check we have three items per map
    if len(from_map) != 3 or len(to_map) != 3:
        arcpy.AddError('Could not find required six fraction rasters.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PERFORM FROM-TO CHANGE CLASSIFICATION

    arcpy.SetProgressor('default', 'Performing change detection...')

    # check z value is appropriate
    z = in_z
    if z < 1 or z > 3:
        raise ValueError('Z-score threshold must be between 1 and 3.')

    try:
        # iter each var...
        pos_chg_map, neg_chg_map = {}, {}
        for var in ['other', 'native', 'weed']:
            # get "from", "to" raster paths
            from_ras, to_ras = from_map[var], to_map[var]

            # perform change detection on frac data
            tmp_chg = f'tmp_frc_chg_{var}.tif'
            change.detect_diff_change(in_from_ras=from_ras,
                                      in_to_ras=to_ras,
                                      out_from_to_ras=tmp_chg)

            # threshold change into zscore where z < -2 or > 2
            tmp_pos, tmp_neg = f'p_{var}.tif', f'n_{var}.tif'  # f'tmp_z_pos_{var}.tif', f'tmp_z_neg_{var}.tif'
            change.threshold_via_zscore(in_ras=tmp_chg,
                                        z=z,
                                        out_z_pos_ras=tmp_pos,
                                        out_z_neg_ras=tmp_neg)

            # create maps of outputs
            pos_chg_map[var] = tmp_pos
            neg_chg_map[var] = tmp_neg

    except Exception as e:
        arcpy.AddError('Could not perform fractional change detection. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region COMBINE CHANGE CLASSIFICATION DATA

    arcpy.SetProgressor('default', 'Combining change detection data...')

    # set output change folder
    change_folder = os.path.join(capture_folder, 'change')
    if not os.path.exists(change_folder):
        os.mkdir(change_folder)

    # create formatted dates for output files
    date_from = from_date.replace('-', '')
    date_to = to_date.replace('-', '')

    try:
        # iter each frac class...
        chg_map = {}
        for direction, item in zip(['gain', 'loss'], [pos_chg_map, neg_chg_map]):
            # unpack map paths
            tmp_other = item['other']
            tmp_native = item['native']
            tmp_weed = item['weed']

            # combine into on raster, three columns and save
            ras_cmb = arcpy.sa.Combine([tmp_other, tmp_native, tmp_weed])

            # set output file name and path and save
            out_fn = f'change_frc_{direction}_{date_from}_to_{date_to}.tif'
            out_cmb = os.path.join(change_folder, out_fn)
            ras_cmb.save(out_cmb)

            # update attribute table classes in-place
            change.update_frac_classes(in_ras=out_cmb)

            # add to final change map
            chg_map[direction] = out_cmb

    except Exception as e:
        arcpy.AddError('Could not combine fractional change data. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE CHANGE AREA

    arcpy.SetProgressor('default', 'Calculating area (ha) of changes...')

    # create path to rehab area polygon
    tmp_bnd = os.path.join(in_project_folder, 'boundary', 'boundary.shp')
    if not os.path.exists(tmp_bnd):
        arcpy.AddError('Could not find rehabilitation area boundary.')
        return

    try:
        # iter each direction...
        for direction in ['gain', 'loss']:

            # set up output csv
            fn_csv = os.path.basename(chg_map[direction])
            fn_csv = fn_csv.split('.')[0]
            fn_csv = f'{fn_csv}_areas.csv'
            out_csv = os.path.join(change_folder, fn_csv)

            # calculate area (ha) per gain class and save to csv
            change.calc_frac_change_areas(in_ras=chg_map[direction],
                                          in_boundary=tmp_bnd,
                                          out_csv=out_csv)

    except Exception as e:
        arcpy.AddWarning('Could not calculate change areas. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region ADD CHANGE RASTER TO ACTIVE MAP

    arcpy.SetProgressor('default', 'Visualising result...')

    # build visualise folder path and
    visualise_folder = os.path.join(in_project_folder, 'visualise')

    try:
        # iter each direction (reverse order for map order)...
        for direction in ['loss', 'gain']:
            # create frac change raster for visualise folder
            tmp_ras = arcpy.Raster(chg_map[direction])

            # prepare path for visualise
            out_fn = os.path.basename(chg_map[direction])
            out_tif = os.path.join(visualise_folder, out_fn)

            # delete previously created visual raster and re-save
            shared.delete_visual_rasters(rasters=[out_tif])
            tmp_ras.save(out_tif)

            # add tif to active map
            shared.add_raster_to_map(in_ras=out_tif)

            # symbolise tif to class colours
            shared.apply_frac_change_symbology(in_ras=out_tif)

    except Exception as e:
        arcpy.AddWarning('Could not visualise fractional change rasters. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    arcpy.SetProgressor('default', 'Cleaning up environment...')

    try:
        # drop temp files (free up space)
        del ras_cmb
        del tmp_ras

    except Exception as e:
        arcpy.AddWarning('Could not drop temporary files. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # clear temp folder (errors skipped)
    shared.clear_tmp_folder(tmp_folder=tmp)

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
