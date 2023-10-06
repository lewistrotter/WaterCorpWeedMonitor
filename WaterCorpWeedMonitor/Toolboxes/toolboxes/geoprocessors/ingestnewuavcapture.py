
def execute(
        parameters
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region IMPORTS

    import os
    import json
    import warnings
    import datetime
    import arcpy

    from scripts import shared

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
    in_blue_band = parameters[2].value
    in_green_band = parameters[3].value
    in_red_band = parameters[4].value
    in_redge_band = parameters[5].value
    in_nir_band = parameters[6].value
    in_dsm_band = parameters[7].value
    in_dtm_band = parameters[8].value

    # inputs for testing only
    # in_project_file = r'C:\Users\Lewis\Desktop\testing\city beach dev\meta.json'
    # in_flight_datetime = datetime.datetime.now()
    # in_blue_band = r'D:\Work\Curtin\Water Corp Project - General\Processed\City Beach\Final Data\ms\ms_ref_blue.tif'
    # in_green_band = r'D:\Work\Curtin\Water Corp Project - General\Processed\City Beach\Final Data\ms\ms_ref_green.tif'
    # in_red_band = r'D:\Work\Curtin\Water Corp Project - General\Processed\City Beach\Final Data\ms\ms_ref_red.tif'
    # in_redge_band = r'D:\Work\Curtin\Water Corp Project - General\Processed\City Beach\Final Data\ms\ms_ref_redge.tif'
    # in_nir_band = r'D:\Work\Curtin\Water Corp Project - General\Processed\City Beach\Final Data\ms\ms_ref_nir.tif'
    # in_dsm_band = r'D:\Work\Curtin\Water Corp Project - General\Processed\City Beach\Final Data\ms\ms_dsm.tif'
    # in_dtm_band = r'D:\Work\Curtin\Water Corp Project - General\Processed\City Beach\Final Data\ms\ms_dtm.tif'

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

    # check if any captures exist (will be >= 4), else error
    if len(meta) < 4:
        arcpy.AddError('Project has no UAV capture data.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE UAV BANDS

    arcpy.SetProgressor('default', 'Preparing UAV bands...')

    try:
        # create band map to maintain band order
        new_band_map = {
            'blue': in_blue_band,
            'green': in_green_band,
            'red': in_red_band,
            'redge': in_redge_band,
            'nir': in_nir_band,
            'dsm': in_dsm_band,
            'dtm': in_dtm_band
        }

        # for each input band, store in map
        for k, v in new_band_map.items():
            new_band_map[k] = arcpy.Describe(v).catalogPath

    except Exception as e:
        arcpy.AddError('Not all input bands are valid. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE CLEAN COMPOSITE WITH REPROJECT AND RESAMPLE

    arcpy.SetProgressor('default', 'Preparing clean band composite...')

    # temporarily disable pyramids to speed things up
    arcpy.env.pyramid = 'NONE'

    try:
        # read raster as a composite
        arcpy.management.CompositeBands(in_rasters=list(new_band_map.values()),
                                        out_raster='tmp_cmp.tif')

        # reproject it to gda 1994 albers using geoprocessor (ia has shift issue)
        arcpy.management.ProjectRaster(in_raster='tmp_cmp.tif',
                                       out_raster='tmp_rsp.tif',
                                       out_coor_system=arcpy.SpatialReference(3577),
                                       resampling_type='BILINEAR',
                                       cell_size='0.05 0.05')

    except Exception as e:
        arcpy.AddError('Could not prepare bands. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region REGISTER NEW COMPOSITE TO FIRST UAV CAPTURE

    arcpy.SetProgressor('default', 'Registering new capture to baseline...')

    # get top-level captures folder
    captures_folder = os.path.join(in_project_folder, 'uav_captures')

    # get baseline capture (element 3 always first capture added)
    base_capture_folder = list(meta.keys())[3]

    # prepend project folder on to base capture folder
    base_capture_folder = os.path.join(captures_folder, base_capture_folder)
    base_capture_bands = os.path.join(base_capture_folder, 'bands')

    # check if baseline folder has bands sub-folder
    if not os.path.exists(base_capture_bands):
        arcpy.AddError('Baseline capture band folder missing.')
        return

    # build baseline band map
    base_bands = []
    for band in list(new_band_map.keys()):
        # build baseline band tif filepath
        base_band_file = os.path.join(base_capture_bands, band + '.tif')
        base_bands.append(base_band_file)

        # check if band exists, else error
        if not os.path.exists(base_band_file):
            arcpy.AddError('Not all input baseline bands exist.')
            return

    try:
        # create composite of baseline bands in order of map
        arcpy.management.CompositeBands(in_rasters=base_bands,
                                        out_raster='tmp_cmp.tif')

        # register new raster to baseline raster
        arcpy.management.RegisterRaster(in_raster='tmp_rsp.tif',
                                        register_mode='REGISTER',
                                        reference_raster='tmp_cmp.tif',
                                        transformation_type='POLYORDER0')

    except Exception as e:
        arcpy.AddError('Could not register new capture data to baseline. See messages.')
        arcpy.AddMessage(str(e))
        return

    # set pyramids back to default
    arcpy.env.pyramid = None

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CREATE NEW UAV CAPTURE FOLDER

    arcpy.SetProgressor('default', 'Creating new UAV flight capture folder...')

    # generate flight date from user input and set as new folder name
    new_flight_date = in_flight_datetime.strftime('%Y%m%d%H%M%S')
    new_capture_folder = os.path.join(in_project_folder, 'uav_captures', new_flight_date)

    # check if flight capture already exists, error if so
    if os.path.exists(new_capture_folder):
        arcpy.AddError('Current UAV capture already exists.')
        return
    else:
        os.mkdir(new_capture_folder)

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT AND EXPORT CLEAN BANDS

    arcpy.SetProgressor('default', 'Conforming and exporting clean UAV bands...')

    # create new bands folder to store clean bands
    bands_folder = os.path.join(new_capture_folder, 'bands')

    # check if bands folder already exists, error if so
    if os.path.exists(bands_folder):
        arcpy.AddError('UAV capture bands folder already exists.')
        return
    else:
        os.mkdir(bands_folder)

    try:
        # read uav grid in as raster
        tmp_grd = arcpy.Raster(grid_tif)

        # read resample raster
        tmp_rsp = arcpy.Raster('tmp_rsp.tif')

        # set up step-wise progressor
        arcpy.SetProgressor('step', None, 0, len(new_band_map))

        # iter over each band in new raster with name...
        for name, band in zip(list(new_band_map.keys()), tmp_rsp.getRasterBands()):
            # set output band tif path
            out_band_path = os.path.join(bands_folder, name + '.tif')

            # conform band to grid pixels via multiply and save
            out_band = band * tmp_grd
            out_band.save(out_band_path)

            # increment progressor
            arcpy.SetProgressorPosition()

    except Exception as e:
        arcpy.AddError('Could not extract clean bands. See messages.')
        arcpy.AddMessage(str(e))
        return

    # reset progressor
    arcpy.ResetProgressor()

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region APPEND NEW CAPTURE TO METADATA

    arcpy.SetProgressor('default', 'Adding new capture to metadata...')

    # build metadata json file
    data = {
        'capture_folder': new_flight_date,
        'capture_date': in_flight_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'capture_type': 'revisit',
        'classified': False,
        'fractions': []
    }

    # add to metadata (dict will always put at end)
    meta[new_flight_date] = data

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
    # region ADD RGB COMPOSITE TO ACTIVE MAP

    arcpy.SetProgressor('default', 'Visualising result...')

    # build visualise folder path and
    visualise_folder = os.path.join(in_project_folder, 'visualise')

    # prepare rgb band paths
    r_tif = os.path.join(bands_folder, 'red.tif')
    g_tif = os.path.join(bands_folder, 'green.tif')
    b_tif = os.path.join(bands_folder, 'blue.tif')

    try:
        # create uav raster rgb compoisite for visualise
        tmp_rgb = arcpy.sa.CompositeBand([r_tif, g_tif, b_tif])

        # save uav rgb raster to visualise folder
        out_tif = os.path.join(visualise_folder, 'uav_rgb' + '_' + new_flight_date + '.tif')
        tmp_rgb.save(out_tif)

        # visualise it on active map
        shared.add_raster_to_map(in_ras=out_tif)

    except Exception as e:
        arcpy.AddWarning('Could not visualise UAV RGB composite. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    arcpy.SetProgressor('default', 'Cleaning up environment...')

    try:
        # close temp files
        del tmp_rsp
        del tmp_rgb

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
