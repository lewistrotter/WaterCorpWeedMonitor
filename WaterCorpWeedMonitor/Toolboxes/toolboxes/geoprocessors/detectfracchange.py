
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

    from scripts import change, shared

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    in_project_file = parameters[0].valueAsText
    in_flight_datetime = parameters[1].valueAsText
    in_s2_from_year = parameters[2].value
    in_s2_to_year = parameters[3].value
    in_s2_month = parameters[4].value

    # inputs for testing only
    # in_project_file = r'C:\Users\Lewis\Desktop\testing\city beach dev\meta.json'
    # in_flight_datetime = '2023-02-02 13:27:34'
    # in_s2_from_year = 2017
    # in_s2_to_year = 2023
    # in_s2_month = 3

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
    # region GET "FROM" AND "TO" FRACTIONAL RASTERS

    arcpy.SetProgressor('default', 'Obtaining fractional "from" and "to" rasters...')

    # check if start year < 2016, fix and warn if so
    if in_s2_from_year < 2016:
        arcpy.AddWarning('Start of rehab < Sentinel 2 data start, using 2016.')
        in_s2_from_year = 2016

    # check if from year less than to year
    if in_s2_from_year >= in_s2_to_year:
        arcpy.AddError('The "from" year must be less than "to" year.')
        return

    # TODO: check if at least 3 years available
    # ...

    # TODO: might want some logic to check if to month exists and roll back 1 if not
    # ...

    # set "from" and "to" dates without days
    date_from = f'{in_s2_from_year}-{str(in_s2_month).zfill(2)}'
    date_to = f'{in_s2_to_year}-{str(in_s2_month).zfill(2)}'

    # get "mid" date
    mid_year = np.floor((in_s2_from_year + in_s2_to_year) / 2)
    mid_date = f'{int(mid_year)}-{str(in_s2_month).zfill(2)}'

    # check if fractions exist
    frac_dates = meta_item['fractions']
    if len(frac_dates) == 0:
        arcpy.AddError('No fractional data found. Run fractional tool.')
        return

    # check if "from", "mid", "to" year-month in fractional list
    for date in [date_from, mid_date, date_to]:
        if date not in frac_dates:
            arcpy.AddError('Could not find requested from and to date fractions.')
            return

    # set fractions folder
    captures_folder = os.path.join(in_project_folder, 'uav_captures')
    capture_folder = os.path.join(captures_folder, meta_item['capture_folder'])
    fractions_folder = os.path.join(capture_folder, 'fractions')

    # build "from" and "to" folders and check they exist
    from_folder = os.path.join(fractions_folder, date_from)
    to_folder = os.path.join(fractions_folder, date_to)
    for folder in [from_folder, to_folder]:
        if not os.path.exists(folder):
            arcpy.AddError('Fraction folders could be found.')
            return

    # create fraction "from", "to" maps
    from_map, to_map = {}, {}

    try:
        folders = [from_folder, to_folder]
        maps = [from_map, to_map]
        for folder, var_map in zip(folders, maps):
            for file in os.listdir(folder):
                if file.endswith('.tif'):
                    var = file.split('.')[0].split('_')[-1]
                    var_map[var] = os.path.join(folder, file)

    except Exception as e:
        arcpy.AddError('Could not obtain fraction rasters. See messages.')
        arcpy.AddMessage(str(e))
        return

    # check if we have three items per map
    if len(from_map) != 3 or len(to_map) != 3:
        arcpy.AddError('Could not obtain find expected six fraction rasters.')
        return

    # create formatted dates for output file later
    date_to = date_to.replace('-', '')
    date_from = date_from.replace('-', '')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PERFORM FROM-TO CHANGE CLASSIFICATION

    # TODO: remove tmp vars

    arcpy.SetProgressor('default', 'Performing change detection...')

    try:
        # iter each var...
        pos_chg_map, neg_chg_map = {}, {}
        for var in ['other', 'native', 'weed']:
            # get "from", "to" raster paths
            from_ras, to_ras = from_map[var], to_map[var]

            # perform change detection on frac data
            tmp_chg_from_to = os.path.join(tmp, f'tmp_chg_from_to_{var}.tif')
            change.detect_diff_change(in_from_ras=from_ras,
                                      in_to_ras=to_ras,
                                      out_from_to_ras=tmp_chg_from_to)

            # threshold "from" to "mid" into zscore where z < -2 or > 2
            tmp_pos = os.path.join(tmp, f'tmp_zsc_pos_{var}.tif')
            tmp_neg = os.path.join(tmp, f'tmp_zsc_neg_{var}.tif')
            change.threshold_via_zscore(in_ras=tmp_chg_from_to,
                                        z=2,
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

    try:
        # for each frac class...
        chg_map = {}
        for dir, item in zip(['gain', 'loss'], [pos_chg_map, neg_chg_map]):
            # unpack map paths
            tmp_other = item['other']
            tmp_native = item['native']
            tmp_weed = item['weed']

            # combine into on raster, three columns and save
            ras_cmb = arcpy.sa.Combine([tmp_other, tmp_native, tmp_weed])

            # set output file name and path and save
            out_fn = f'change_frac_{dir}_{date_from}_to_{date_to}.tif'
            out_cmb = os.path.join(change_folder, out_fn)
            ras_cmb.save(out_cmb)

            # fix field names
            change.fix_field_names(in_ras=out_cmb)

            # update attribute table classes in-place
            change.update_frac_classes(in_ras=out_cmb)

            # add to final change map
            chg_map[dir] = out_cmb

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
        # set up output gain csv
        fn_gain_csv = os.path.basename(chg_map['gain'])
        fn_gain_csv = fn_gain_csv.split('.')[0] + '_areas' + '.csv'
        out_gain_csv = os.path.join(change_folder, fn_gain_csv)

        # calculate area (ha) per gain class and save to csv
        change.calc_frac_change_areas(in_ras=chg_map['gain'],
                                      in_boundary=tmp_bnd,
                                      out_csv=out_gain_csv)

        # set up output gain csv
        fn_loss_csv = os.path.basename(chg_map['gain'])
        fn_loss_csv = fn_loss_csv.split('.')[0] + '_areas' + '.csv'
        out_loss_csv = os.path.join(change_folder, fn_loss_csv)

        # calculate area (ha) per gain class and save to csv
        change.calc_frac_change_areas(in_ras=chg_map['loss'],
                                      in_boundary=tmp_bnd,
                                      out_csv=out_loss_csv)

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
        # create frac change raster for visualise folder
        tmp_gain = arcpy.Raster(chg_map['gain'])
        tmp_loss = arcpy.Raster(chg_map['loss'])

        # save gain to visualise folder
        out_gain_fn = os.path.basename(chg_map['gain'])
        out_gain_tif = os.path.join(visualise_folder, out_gain_fn)
        tmp_gain.save(out_gain_tif)

        # save loss to visualise folder
        out_loss_fn = os.path.basename(chg_map['loss'])
        out_loss_tif = os.path.join(visualise_folder, out_loss_fn)
        tmp_loss.save(out_loss_tif)

        # add gain and loss tifs to active map
        shared.add_raster_to_map(in_ras=out_gain_tif)
        shared.add_raster_to_map(in_ras=out_loss_tif)

        # symbolise gain and loss tifs to class colours
        shared.apply_frac_change_symbology(in_ras=out_gain_tif)
        shared.apply_frac_change_symbology(in_ras=out_loss_tif)

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







