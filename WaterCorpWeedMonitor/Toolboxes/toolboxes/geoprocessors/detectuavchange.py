import pandas as pd


def execute(
        parameters
        # messages # TODO: implement
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region IMPORTS

    import os
    import json
    import datetime
    import arcpy

    from scripts import change, shared

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    in_project_file = parameters[0].valueAsText
    in_uav_from_date = parameters[1].valueAsText
    in_uav_to_date = parameters[2].valueAsText

    # inputs for testing only
    # in_project_file = r'C:\Users\Lewis\Desktop\testing\city beach dev\meta.json'
    # in_uav_from_date = '2023-02-02 13:27:34'
    # in_uav_to_date = '2023-07-06 13:31:28'

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
    # region GET "FROM" AND "TO" UAV CAPTURE RASTERS

    arcpy.SetProgressor('default', 'Obtaining UAV "from" and "to" rasters...')

    # exclude top-level metadata items
    exclude_keys = ['project_name', 'date_created', 'date_rehab']

    # extract "from" raster
    from_meta_item = None
    for k, v in meta.items():
        if k not in exclude_keys:
            if v['capture_date'] == in_uav_from_date:
                from_meta_item = v

    # extract "to" raster
    to_meta_item = None
    for k, v in meta.items():
        if k not in exclude_keys:
            if v['capture_date'] == in_uav_to_date:
                to_meta_item = v

    # check we have both, else error
    if from_meta_item is None or to_meta_item is None:
        arcpy.AddError('Could not obtain requested "from" or "to" metadata.')
        return

    # create formatted dates for output file later
    date_from = from_meta_item['capture_date'].split(' ')[0].replace('-', '')
    date_to = to_meta_item['capture_date'].split(' ')[0].replace('-', '')

    # check "from" is lower than "to"
    if date_from >= date_to:
        arcpy.AddError('UAV capture "from" date must be less than "to" date.')
        return

    # set captures folder
    captures_folder = os.path.join(in_project_folder, 'uav_captures')

    # set "from" uav capture folder and raster
    from_folder = os.path.join(captures_folder, from_meta_item['capture_folder'])
    from_ras = os.path.join(from_folder, 'classify', 'rf_optimal.tif')

    # set "to" uav capture folder and raster
    to_folder = os.path.join(captures_folder, to_meta_item['capture_folder'])
    to_ras = os.path.join(to_folder, 'classify', 'rf_optimal.tif')

    # check both rasters exist
    for ras in [from_ras, to_ras]:
        if not os.path.exists(ras):
            arcpy.AddError('UAV capture raster does not exist.')
            return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PERFORM CHANGE DETECTION

    arcpy.SetProgressor('default', 'Performing change detection...')

    try:
        # TODO: try and remove tmp vars, have done it up to here
        # perform change detection on uav data
        tmp_chg = os.path.join(tmp, 'tmp_chg.tif')
        change.detect_category_change(in_from_ras=from_ras,
                                      in_to_ras=to_ras,
                                      out_change_ras='tmp_chg.tif')

    except Exception as e:
        arcpy.AddError('Could not perform UAV change detection. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CLEAN CHANGE DETECTION RASTER

    arcpy.SetProgressor('default', 'Cleaning change detection output...')

    # set output change folder
    change_folder = os.path.join(captures_folder, to_folder, 'change')
    if not os.path.exists(change_folder):
        os.mkdir(change_folder)

    try:
        # extract raster attributes depending on user selection
        chg_attrs = change.extract_uav_change_attrs(in_ras=tmp_chg)

        # set output file name and path
        out_fn = f'change_uav_{date_from}_to_{date_to}.tif'
        ras_cln = os.path.join(change_folder, out_fn)

        # apply majority filter x3 and create new raster
        change.remove_uav_noise(in_ras=tmp_chg,
                                out_ras=ras_cln)

        # append original raster attributes back on
        change.append_uav_attrs(in_ras=ras_cln,
                                in_attrs=chg_attrs)

    except Exception as e:
        arcpy.AddError('Could not clean UAV change detection raster. See messages.')
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
        # calculate area (ha) per class and save to csv
        tmp_csv = os.path.join(change_folder, 'areas.csv')
        change.calc_uav_change_areas(in_ras=ras_cln,
                                     in_boundary=tmp_bnd,
                                     out_csv=tmp_csv)

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
        # create uav change raster for visualise folder
        tmp_cln = arcpy.Raster(ras_cln)

        # save uav rgb raster to visualise folder
        out_tif = os.path.join(visualise_folder, out_fn)
        tmp_cln.save(out_tif)

        # visualise it on active map and symbolise it to class colors
        shared.add_raster_to_map(in_ras=out_tif)
        shared.apply_uav_change_symbology(in_ras=out_tif)

    except Exception as e:
        arcpy.AddWarning('Could not visualise UAV change raster. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    arcpy.SetProgressor('default', 'Cleaning up environment...')

    try:
        # drop temp files (free up space)
        del tmp_cln

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







