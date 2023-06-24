
import os
import json
import datetime
import arcpy

# from scripts import ...


def execute(
        parameters
        # messages # TODO: implement
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    in_project_file = parameters[0].valueAsText

    # inputs for testing only
    #in_project_file = r'C:\Users\Lewis\Desktop\testing\city beach demo\meta.json'

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

    # check if uav grid file exists
    grid_tif = os.path.join(in_project_folder, 'grid', 'grid_uav.tif')
    if not os.path.exists(grid_tif):
        arcpy.AddError('Project grid file does not exist.')
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
    # region EXTRACT FIRST AND LAST UAV CAPTURE METADATA

    arcpy.SetProgressor('default', 'Extracting first and last UAV capture metadata...')

    # exclude top-level metadata items
    exclude_keys = ['project_name', 'date_created', 'date_rehab']

    # extract selected metadata item based on capture date
    base_meta_item = None
    other_meta_items = []
    for k, v in meta.items():
        if k not in exclude_keys:
            if v['capture_type'] == 'baseline':
                base_meta_item = v
            elif v['capture_type'] == 'revisit':
                other_meta_items.append(v)

    # check if we got a second uav capture
    if base_meta_item is None or len(other_meta_items) == 0:
        arcpy.AddError('Project only has one UAV capture, need at least two.')
        raise  # return

    # get last meta item
    # TODO: sort this by date first
    last_meta_item = other_meta_items[-1]

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # region PREPARE UAV CAPTURE RASTERS

    arcpy.SetProgressor('default', 'Preparing UAV capture rasters...')

    # create capture and uav folders folders
    captures_folder = os.path.join(in_project_folder, 'uav_captures')
    base_uav_folder = os.path.join(captures_folder, base_meta_item['capture_folder'])
    last_uav_folder = os.path.join(captures_folder, last_meta_item['capture_folder'])

    # crerate classify folders
    base_classify_folder = os.path.join(base_uav_folder, 'classify')
    last_classify_folder = os.path.join(last_uav_folder, 'classify')

    # check if both folders exist
    for folder in [base_classify_folder, last_classify_folder]:
        if not os.path.exists(folder):
            arcpy.AddError('Baseline capture folder missing.')
            raise  # return

    try:
        # read baseline uav raster in
        tmp_bse = os.path.join(base_classify_folder, 'rf_optimal.tif')
        tmp_bse = arcpy.Raster(tmp_bse)

        # read last uav raster in
        tmp_lst = os.path.join(last_classify_folder, 'rf_optimal.tif')
        tmp_lst = arcpy.Raster(tmp_lst)

    except Exception as e:
        arcpy.AddWarning('Could not read expected class rasters. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # region PERFORM FROM-TO CHANGE CLASSIFICATION

    arcpy.SetProgressor('default', 'Detecting change between UAV captures...')

    # create new change folder to store clean bands
    change_folder = os.path.join(last_uav_folder, 'change')

    # check if change folder already exists, error if so
    if not os.path.exists(change_folder):
        os.mkdir(change_folder)

    try:
        # calculate change from baseline to latest uav capture
        out_ras = arcpy.ia.ComputeChangeRaster(from_raster=tmp_bse,
                                               to_raster=tmp_lst,
                                               compute_change_method='CATEGORICAL_DIFFERENCE',
                                               filter_method='ALL',
                                               define_transition_colors='AVERAGE')

        # save out raster to last capture folder
        tmp_out = os.path.join(change_folder, 'change_from_baseline.tif')
        out_ras.save(tmp_out)

    except Exception as e:
        arcpy.AddWarning('Could not perform change detection. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # region END ENVIRONMENT

    try:
        # TODO: uncomment below if wc have ia
        # drop temp files (free up space)
        #arcpy.management.Delete(tmp_comp)

        # TODO: remove below if wc has no ia
        # close temp files
        del tmp_bse
        del tmp_lst

    except Exception as e:
        arcpy.AddWarning('Could not drop temporary files. See messages.')
        arcpy.AddMessage(str(e))

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







