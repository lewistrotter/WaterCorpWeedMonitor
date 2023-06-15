
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
    in_out_folder = parameters[0].valueAsText
    in_rehab_datetime = parameters[1].value
    in_flight_datetime = parameters[2].value
    in_blue_band = parameters[3].value
    in_green_band = parameters[4].value
    in_red_band = parameters[5].value
    in_redge_band = parameters[6].value
    in_nir_band = parameters[7].value
    in_dsm_band = parameters[8].value
    in_dtm_band = parameters[9].value

    # inputs for testing only
    # in_out_folder = r'C:\Users\Lewis\Desktop\testing\project_2'
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
    # region CREATE PROJECT FOLDER AND STRUCTURE

    arcpy.SetProgressor('default', 'Creating project folder structure...')

    # check if output folder exists
    if not os.path.exists(in_out_folder):
        arcpy.AddError('Project folder does not exist.')
        raise  # return

    # check if a project meta file exists, error if it does
    in_out_meta_file = os.path.join(in_out_folder, 'meta.json')
    if os.path.exists(in_out_meta_file):
        arcpy.AddError('Project metadata file already exists.')
        raise  # return

    # check if required project folders already exist, error if so
    sub_folders = ['grid', 'uav_captures', 'sat_captures', 'visualise']
    for sub_folder in sub_folders:
        sub_folder = os.path.join(in_out_folder, sub_folder)
        if os.path.exists(sub_folder):
            arcpy.AddError('Project folders already exist.')
            raise  # return
        else:
            os.mkdir(sub_folder)

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
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE CLEAN COMPOSITE WITH REPROJECT AND RESAMPLE

    arcpy.SetProgressor('default', 'Preparing clean band composite...')

    try:
        # TODO: uncomment below if wc has no ia
        # # composite bands in order of map and output to scratch
        # tmp_comp = 'tmp_comp.crf'  # r'memory\tmp_comp'
        # arcpy.management.CompositeBands(in_rasters=list(band_map.values()),
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
        tmp_cmp = arcpy.ia.CompositeBand(list(band_map.values()))
        tmp_prj = arcpy.ia.Reproject(tmp_cmp, {'wkid': 32750})
        tmp_rsp = arcpy.ia.Resample(tmp_prj, 'Bilinear', None, 0.05)

    except Exception as e:
        arcpy.AddError('Could not prepare bands. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CREATE BASELINE GRID

    arcpy.SetProgressor('default', 'Creating baseline grids...')

    # set project grid folder
    grid_folder = os.path.join(in_out_folder, 'grid')

    try:
        # build uniform grid for future uav resampling
        arcpy.management.CreateRandomRaster(out_path=grid_folder,
                                            out_name='grid_uav.tif',
                                            distribution='INTEGER 1 1',
                                            raster_extent=tmp_rsp,
                                            cellsize=0.05)

        # build uniform grid for future sentinel 2 resampling
        arcpy.management.CreateRandomRaster(out_path=grid_folder,
                                            out_name='grid_s2.tif',
                                            distribution='INTEGER 1 1',
                                            raster_extent=tmp_rsp,
                                            cellsize=10.0)

    except Exception as e:
        arcpy.AddError('Could not create baseline grids. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

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
        raise  # return
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
        raise  # return
    else:
        os.mkdir(bands_folder)

    try:
        # set uav grid tif path
        grid_tif = os.path.join(grid_folder, 'grid_uav.tif')

        # TODO: uncomment below if wc has no ia
        # # iter over each band in raster...
        # desc = arcpy.Describe(tmp_rsp)
        # for i, b in enumerate(desc.children):
        #     # set current crf band name, output tif band names
        #     in_band_path = os.path.join(tmp_rsp, b.name + '.crf')
        #     out_band_path = os.path.join(bands_folder, list(band_map.keys())[i] + '.tif')
        #
        #     # extract clean band from crf to project grid via times, save as tif
        #     out_raster = arcpy.sa.Times(in_raster_or_constant1=grid_tif,
        #                                 in_raster_or_constant2=in_band_path)
        #     out_raster.save(out_band_path)

        # TODO: remove below wc has no ia
        # read uav grid in as raster
        tmp_grd = arcpy.Raster(grid_tif)

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
        raise  # return

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
        'classified': False
    }

    # build metadata json file
    meta = {
        'project_name': meta_project_name,
        'date_created': meta_date_created,
        'date_rehab': meta_date_rehab,
        new_flight_date: data
    }

    try:
        # write json metadata file to project folder top-level
        with open(in_out_meta_file, 'w') as fp:
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

        # TODO: remove below if wc has no ia
        # close temp files
        del tmp_cmp
        del tmp_prj
        del tmp_rsp
        del tmp_grd

    except Exception as e:
        arcpy.AddWarning('Could not drop temporary files. See messages.')
        arcpy.AddMessage(str(e))

    # free up spatial analyst and image analyst
    arcpy.CheckInExtension('Spatial')
    arcpy.CheckInExtension('ImageAnalyst')  # TODO: remove if wc has no ia

    # set changed env variables back to default
    arcpy.env.overwriteOutput = False
    arcpy.env.addOutputsToMap = True

    # endregion

    return

# testing
#execute(None)
