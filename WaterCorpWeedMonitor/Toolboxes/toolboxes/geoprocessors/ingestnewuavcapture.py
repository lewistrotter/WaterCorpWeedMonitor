
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
    in_flight_datetime = parameters[1].value
    in_blue_band = parameters[2].value
    in_green_band = parameters[3].value
    in_red_band = parameters[4].value
    in_redge_band = parameters[5].value
    in_nir_band = parameters[6].value
    in_dsm_band = parameters[7].value
    in_dtm_band = parameters[8].value

    # inputs for testing only
    # in_project_file = r'C:\Users\Lewis\Desktop\testing\tmp2\meta.json'
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
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE CLEAN COMPOSITE WITH REPROJECT AND RESAMPLE

    arcpy.SetProgressor('default', 'Preparing clean band composite...')

    try:
        # TODO: uncomment below if wc has no ia
        # # composite bands in order of map and output to scratch
        # tmp_comp = 'tmp_comp.crf'  # r'memory\tmp_comp'
        # arcpy.management.CompositeBands(in_rasters=list(new_band_map.values()),
        #                                 out_raster=tmp_comp)
        #
        # # reproject to wgs84 utm zone 50s to be safe
        # tmp_prj = 'tmp_prj.crf'  # r'memory\tmp_prj'
        # arcpy.management.ProjectRaster(in_raster=tmp_comp,
        #                                out_raster=tmp_prj,
        #                                out_coor_system=arcpy.SpatialReference(32750))
        #
        # # resample (bilinear) new bands to project grid (0.05 is grid cell size)
        # tmp_rsp = 'tmp_rsp.crf'  # r'memory\tmp_rsp'
        # arcpy.management.Resample(in_raster=tmp_prj,
        #                           out_raster=tmp_rsp,
        #                           cell_size='0.05',
        #                           resampling_type='BILINEAR')

        # TODO: remove below if wc has no ia
        # read raster, composite it, reproject to wgs84 utm, resample to standard grid
        tmp_cmp = arcpy.ia.CompositeBand(list(new_band_map.values()))
        tmp_prj = arcpy.ia.Reproject(tmp_cmp, {'wkid': 32750})
        tmp_rsp = arcpy.ia.Resample(tmp_prj, 'Bilinear', None, 0.05)

    except Exception as e:
        arcpy.AddError('Could not prepare bands. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

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
        raise  # return

    # build baseline band map
    base_bands = []
    for band in list(new_band_map.keys()):
        # build baseline band tif filepath
        base_band_file = os.path.join(base_capture_bands, band + '.tif')
        base_bands.append(base_band_file)

        # check if band exists, else error
        if not os.path.exists(base_band_file):
            arcpy.AddError('Not all input baseline bands exist.')
            raise  # return

    try:
        # TODO: uncomment below if wc has no ia
        # # composite baseline bands in order of map and output to scratch
        # tmp_base_comp = 'tmp_base_comp.crf'  # r'memory\tmp_base_comp'
        # arcpy.management.CompositeBands(in_rasters=base_bands,
        #                                 out_raster=tmp_base_comp)

        # # register (shift) new resampled capture composite to baseline composite
        # arcpy.management.RegisterRaster(in_raster=tmp_rsp,
        #                                 register_mode='REGISTER',
        #                                 reference_raster=tmp_base_comp,
        #                                 transformation_type='POLYORDER0')

        # TODO: remove below if wc has no ia
        # create composite of baseline bands in order of map
        tmp_base_cmp = arcpy.ia.CompositeBand(base_bands)

        # register new raster to baseline raster, should update in memory
        arcpy.management.RegisterRaster(in_raster=tmp_rsp,
                                        register_mode='REGISTER',
                                        reference_raster=tmp_base_cmp,
                                        transformation_type='POLYORDER0')

    except Exception as e:
        arcpy.AddError('Could not register new capture data to baseline. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

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
        raise  # return
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
        raise  # return
    else:
        os.mkdir(bands_folder)

    try:
        # TODO: uncomment below if wc has no ia
        # # iter over each band in raster...
        # desc = arcpy.Describe(tmp_rsp)
        # for i, b in enumerate(desc.children):
        #     # set current crf band name, output tif band names
        #     in_band_path = os.path.join(tmp_rsp, b.name + '.crf')
        #     out_band_path = os.path.join(bands_folder, list(new_band_map.keys())[i] + '.tif')
        #
        #     # extract clean band pixels to project grid via times
        #     out_raster = arcpy.sa.Times(in_raster_or_constant1=grid_tif,
        #                                 in_raster_or_constant2=in_band_path)
        #     out_raster.save(out_band_path)

        # TODO: remove below wc has no ia
        # read uav grid in as raster
        tmp_grd = arcpy.Raster(grid_tif)

        # set up step-wise progressor
        arcpy.SetProgressor('step', 'Conforming and exporting clean UAV bands...', 0, len(new_band_map))

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
        raise  # return

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
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    try:
        # TODO: uncomment below if wc have ia
        # drop temp files (free up space)
        #arcpy.management.Delete(tmp_comp)
        #arcpy.management.Delete(tmp_prj)
        #arcpy.management.Delete(tmp_rsp)
        #arcpy.management.Delete(tmp_base_comp)

        # TODO: remove below if wc has no ia
        # close temp files
        del tmp_cmp
        del tmp_prj
        del tmp_rsp
        del tmp_base_cmp

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
