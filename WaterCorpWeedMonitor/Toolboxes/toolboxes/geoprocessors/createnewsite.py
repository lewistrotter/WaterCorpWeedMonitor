
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
    in_out_folder = parameters[0].valueAsText
    in_boundary_feat = parameters[1].valueAsText
    in_rehab_datetime = parameters[2].value
    in_flight_datetime = parameters[3].value
    in_blue_band = parameters[4].value
    in_green_band = parameters[5].value
    in_red_band = parameters[6].value
    in_redge_band = parameters[7].value
    in_nir_band = parameters[8].value
    in_dsm_band = parameters[9].value
    in_dtm_band = parameters[10].value

    # inputs for testing only
    # in_out_folder = r'C:\Users\Lewis\Desktop\testing\city beach dev'
    # in_boundary_feat = r'D:\Work\Curtin\Water Corp Project - General\Processed\City Beach\Boundary\CityBeach.shp'
    # in_rehab_datetime = datetime.datetime.now()
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
    # region CREATE PROJECT FOLDER AND STRUCTURE

    arcpy.SetProgressor('default', 'Creating project folder structure...')

    # check if output folder exists
    if not os.path.exists(in_out_folder):
        arcpy.AddError('Project folder does not exist.')
        return

    # check if a project meta file exists, error if so
    in_out_meta_file = os.path.join(in_out_folder, 'meta.json')
    if os.path.exists(in_out_meta_file):
        arcpy.AddError('Project metadata file already exists.')
        return

    # check if required project folders already exist, error if so
    sub_folders = ['boundary', 'grid', 'uav_captures', 'sat_captures', 'visualise']
    for sub_folder in sub_folders:
        sub_folder = os.path.join(in_out_folder, sub_folder)
        if os.path.exists(sub_folder):
            arcpy.AddError('Project folders already exist.')
            return
        else:
            os.mkdir(sub_folder)

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CREATE AND SET WORKSPACE TO TEMPORARY FOLDER

    arcpy.SetProgressor('default', 'Preparing workspace...')

    # create temp folder if does not already exist
    tmp = os.path.join(in_out_folder, 'tmp')
    if not os.path.exists(tmp):
        os.mkdir(tmp)

    # clear temp folder (errors skipped)
    shared.clear_tmp_folder(tmp_folder=tmp)

    # set temp folder to arcpy workspace
    arcpy.env.workspace = tmp

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE BOUNDARY

    arcpy.SetProgressor('default', 'Preparing boundary...')

    try:
        # create boundary shapefile
        arcpy.management.Dissolve(in_features=in_boundary_feat,
                                  out_feature_class='tmp_diss.shp')

        # add sid field to boundary shapefile
        arcpy.management.AddField(in_table='tmp_diss.shp',
                                  field_name='SID',
                                  field_type='SHORT')

        # make sure sid row has a value of 1
        arcpy.management.CalculateField(in_table='tmp_diss.shp',
                                        field='SID',
                                        expression='1')

        # reproject boundary to gda 1994 albers and output to project
        boundary_shp = os.path.join(in_out_folder, 'boundary', 'boundary.shp')
        arcpy.management.Project(in_dataset='tmp_diss.shp',
                                 out_dataset=boundary_shp,
                                 out_coor_system=arcpy.SpatialReference(3577))

    except Exception as e:
        arcpy.AddError('Could not prepare boundary. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE UAV BANDS

    arcpy.SetProgressor('default', 'Preparing UAV bands...')

    try:
        # create band map to maintain band order
        band_map = {
            'blue': in_blue_band,
            'green': in_green_band,
            'red': in_red_band,
            'redge': in_redge_band,
            'nir': in_nir_band,
            'dsm': in_dsm_band,
            'dtm': in_dtm_band
        }

        # for each input band, store in map
        for k, v in band_map.items():
            band_map[k] = arcpy.Describe(v).catalogPath

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
        arcpy.management.CompositeBands(in_rasters=list(band_map.values()),
                                        out_raster='tmp_cmp.tif')

        # reproject it to gda94 albers using geoprocessor (ia has shift issue)
        arcpy.management.ProjectRaster(in_raster='tmp_cmp.tif',
                                       out_raster='tmp_rsp.tif',
                                       out_coor_system=arcpy.SpatialReference(3577),
                                       resampling_type='BILINEAR',
                                       cell_size='0.05 0.05')

    except Exception as e:
        arcpy.AddError('Could not prepare bands. See messages.')
        arcpy.AddMessage(str(e))
        return

    # set pyramids back to default
    arcpy.env.pyramid = None

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CREATE BASELINE UAV GRID

    arcpy.SetProgressor('default', 'Creating baseline UAV grid...')

    # set project grid folder
    grid_folder = os.path.join(in_out_folder, 'grid')

    try:
        # build uniform grid for future uav resampling
        arcpy.management.CreateRandomRaster(out_path=grid_folder,
                                            out_name='grid.tif',
                                            distribution='INTEGER 1 1',
                                            raster_extent='tmp_rsp.tif',
                                            cellsize=0.05)

    except Exception as e:
        arcpy.AddError('Could not create baseline UAV grid. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CREATE NEW UAV CAPTURE FOLDER

    arcpy.SetProgressor('default', 'Creating new UAV capture folder...')

    # generate flight date from user input and set as new folder name
    new_flight_date = in_flight_datetime.strftime('%Y%m%d%H%M%S')
    capture_folder = os.path.join(in_out_folder, 'uav_captures', new_flight_date)

    # check if flight capture already exists, error if so
    if os.path.exists(capture_folder):
        arcpy.AddError('Current UAV capture already exists.')
        return
    else:
        os.mkdir(capture_folder)

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT AND EXPORT CLEAN BANDS

    arcpy.SetProgressor('default', 'Conforming and exporting clean UAV bands...')

    # create new bands folder to store clean bands
    bands_folder = os.path.join(capture_folder, 'bands')

    # check if bands folder already exists, error if so
    if os.path.exists(bands_folder):
        arcpy.AddError('UAV capture bands folder already exists.')
        return
    else:
        os.mkdir(bands_folder)

    try:
        # set uav grid tif path and read it in as raster
        grid_tif = os.path.join(grid_folder, 'grid.tif')
        tmp_grd = arcpy.Raster(grid_tif)

        # read resample raster
        tmp_rsp = arcpy.Raster('tmp_rsp.tif')

        # set up step-wise progressor
        arcpy.SetProgressor('step', None, 0, len(band_map))

        # iter over each band in raster with name...
        for name, band in zip(list(band_map.keys()), tmp_rsp.getRasterBands()):
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
    # region CREATE AND BUILD METADATA

    arcpy.SetProgressor('default', 'Creating metadata...')

    # set top-level metadata variables
    meta_project_name = os.path.basename(in_out_folder)
    meta_date_created = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    meta_date_rehab = in_rehab_datetime.strftime('%Y-%m-%d %H:%M:%S')

    # set first capture metadata
    data = {
        'capture_folder': new_flight_date,
        'capture_date': in_flight_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'capture_type': 'baseline',
        'classified': False,
        'fractions': []
    }

    # build metadata json file
    meta = {
        'project_name': meta_project_name,
        'date_created': meta_date_created,
        'date_rehab': meta_date_rehab,
        'sat_shift_x': 0.0,
        'sat_shift_y': 0.0,
        new_flight_date: data
    }

    try:
        # write json metadata file to project folder top-level
        with open(in_out_meta_file, 'w') as fp:
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
    visualise_folder = os.path.join(in_out_folder, 'visualise')

    # prepare rgb band paths
    r_tif = os.path.join(bands_folder, 'red.tif')
    g_tif = os.path.join(bands_folder, 'green.tif')
    b_tif = os.path.join(bands_folder, 'blue.tif')

    try:
        # create uav raster rgb compoisite for visualise
        tmp_rgb = arcpy.sa.CompositeBand([r_tif, g_tif, b_tif])

        # create uav rgb raster path to visualise folder
        out_tif = os.path.join(visualise_folder, 'uav_rgb' + '_' + new_flight_date + '.tif')

        # delete previously created visual raster and re-save
        shared.delete_visual_rasters(rasters=[out_tif])
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
        del tmp_grd
        del tmp_rgb

    except Exception as e:
        arcpy.AddWarning('Could not drop temporary files. See messages.')
        arcpy.AddMessage(str(e))
        pass

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
