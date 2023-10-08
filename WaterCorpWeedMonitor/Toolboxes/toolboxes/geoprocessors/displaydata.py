
def execute(
        parameters
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region IMPORTS

    import os
    import json
    import warnings
    import xarray as xr
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
    in_layers_to_visualise = parameters[2].valueAsText

    # inputs for testing only
    #in_project_file = r'C:\Users\Lewis\Desktop\testing\citybeach\meta.json'
    #in_flight_datetime = '2024-02-05 11:00:00'
    #in_layers_to_visualise = "'RGB (UAV)';'NDVI (UAV)';'Classified (UAV)';'Change (UAV)';'Fractions (SAT)'"

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

    # build capture and visulise folders
    capture_folder = os.path.join(in_project_folder, 'uav_captures', meta_item['capture_folder'])
    visualise_folder = os.path.join(in_project_folder, 'visualise')

    # get flight date code
    flight_date = meta_item['capture_folder']

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARING PARAMETERS

    arcpy.SetProgressor('default', 'Preparing input parameters...')

    # parse layers from parameter
    lyrs = []
    for lyr in in_layers_to_visualise.split(';'):
        lyr = lyr.replace("'", '').strip()
        lyrs.append(lyr)

    # check we got something back
    if len(lyrs) == 0:
        arcpy.AddError('No layers requested for visualisation.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region VISUALISE UAV RGB RASTER IF REQUESTED

    if 'RGB (UAV)' in lyrs:

        arcpy.SetProgressor('default', 'Visualising UAV RGB data...')

        # create bands folder
        bands_folder = os.path.join(capture_folder, 'bands')

        # create raw band map to maintain band order
        band_map = {
            'blue': os.path.join(bands_folder, 'blue.tif'),
            'green': os.path.join(bands_folder, 'green.tif'),
            'red': os.path.join(bands_folder, 'red.tif')
        }

        try:
            # open bands as seperate rasters
            blue = arcpy.Raster(band_map['blue'])
            green = arcpy.Raster(band_map['green'])
            red = arcpy.Raster(band_map['red'])

            # create uav raster rgb compoisite for visualise
            tmp_rgb = arcpy.sa.CompositeBand([red, green, blue])

            # create uav rgb raster path to visualise folder
            out_tif = os.path.join(visualise_folder, 'uav_rgb' + '_' + flight_date + '.tif')

            # delete previously created visual raster and re-save
            shared.delete_visual_rasters(rasters=[out_tif])
            tmp_rgb.save(out_tif)

            # visualise it on active map
            shared.add_raster_to_map(in_ras=out_tif)

        except Exception as e:
            arcpy.AddWarning('Could not visualise UAV RGB raster. See messages.')
            arcpy.AddMessage(str(e))
            pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region VISUALISE UAV NDVI RASTER IF REQUESTED

    if 'NDVI (UAV)' in lyrs:

        arcpy.SetProgressor('default', 'Visualising UAV NDVI data...')

        # create bands folder
        bands_folder = os.path.join(capture_folder, 'bands')

        # create raw band map to maintain band order
        band_map = {
            'red': os.path.join(bands_folder, 'red.tif'),
            'nir': os.path.join(bands_folder, 'nir.tif')
        }

        try:
            # open bands as seperate rasters
            red = arcpy.Raster(band_map['red'])
            nir = arcpy.Raster(band_map['nir'])

            # calculate ndvi
            ndvi = (nir - red) / (nir + red)

            # create uav rgb raster path to visualise folder
            out_tif = os.path.join(visualise_folder, 'uav_ndvi' + '_' + flight_date + '.tif')

            # delete previously created visual raster and re-save
            shared.delete_visual_rasters(rasters=[out_tif])
            ndvi.save(out_tif)

            # visualise it on active map
            shared.add_raster_to_map(in_ras=out_tif)
            shared.apply_ndvi_layer_symbology(in_ras=out_tif)

        except Exception as e:
            arcpy.AddWarning('Could not visualise UAV NDVI raster. See messages.')
            arcpy.AddMessage(str(e))
            pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region VISUALISE UAV CLASSIFIED RASTER IF REQUESTED

    if 'Classified (UAV)' in lyrs:

        arcpy.SetProgressor('default', 'Visualising UAV Classified data...')

        # get classify folder
        classify_folder = os.path.join(capture_folder, 'classify')

        try:
            # get optimal classification raster
            cls_tif = os.path.join(classify_folder, 'rf_optimal.tif')

            # open classification raster
            tmp_cls = arcpy.Raster(cls_tif)

            # create uav classified raster path to visualise folder
            out_tif = os.path.join(visualise_folder, 'uav_classified' + '_' + flight_date + '.tif')

            # delete previously created visual raster and re-save
            shared.delete_visual_rasters(rasters=[out_tif])
            tmp_cls.save(out_tif)

            # visualise it on active map
            shared.add_raster_to_map(in_ras=out_tif)
            shared.apply_classified_symbology(in_ras=out_tif)

        except Exception as e:
            arcpy.AddWarning('Could not visualise UAV Classified raster. See messages.')
            arcpy.AddMessage(str(e))
            pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region VISUALISE UAV CHANGE RASTER IF REQUESTED

    if 'Change (UAV)' in lyrs:

        arcpy.SetProgressor('default', 'Visualising UAV Change data...')

        # get classify folder
        change_folder = os.path.join(capture_folder, 'change')

        try:
            # check if uav change raster in folder
            tif = None
            for file in os.listdir(change_folder):
                if 'change_uav_' in file and file.endswith('.tif'):
                    tif = file
                    break

            if tif is not None:
                # create uav classified raster path to visualise folder
                in_tif = os.path.join(change_folder, tif)
                out_tif = os.path.join(visualise_folder, tif)

                # delete previously created visual raster and re-save
                shared.delete_visual_rasters(rasters=[out_tif])

                # copy raster over to visualise folder
                arcpy.management.CopyRaster(in_raster=in_tif,
                                            out_rasterdataset=out_tif)

                # visualise it on active map
                shared.add_raster_to_map(in_ras=out_tif)
                shared.apply_uav_change_symbology(in_ras=out_tif)
            else:
                arcpy.AddWarning('Could not find UAV Change raster.')

        except Exception as e:
            arcpy.AddWarning('Could not visualise UAV Change raster. See messages.')
            arcpy.AddMessage(str(e))
            pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region VISUALISE SAT FRACTIONS RASTERs IF REQUESTED

    if 'Fractions (SAT)' in lyrs:

        arcpy.SetProgressor('default', 'Visualising Satellite Fractions data...')

        # get fraction folder
        fraction_folder = os.path.join(capture_folder, 'fractions')

        try:
            # get full fraction dataset
            ds_list = []
            for root, dirs, files in os.walk(fraction_folder):
                for file in files:
                    if 'frc' in file and file.endswith('.nc'):
                        ds_list.append(os.path.join(root, file))

            # read all netcdfs into single dataset
            ds = shared.concat_netcdf_files(nc_files=ds_list)

            # convert to a crf for each fractional variable
            for var in ds:
                # export current var
                tmp_frc_nc = os.path.join(tmp, f'frc_{var}.nc')
                ds[[var]].to_netcdf(tmp_frc_nc)

                # create crf path to visualise folder
                out_crf = os.path.join(visualise_folder, f'frc_{var}_{flight_date}.crf')

                # delete previously created visual crf
                shared.delete_visual_rasters(rasters=[out_crf])

                # re-save it
                shared.netcdf_to_crf(in_nc=tmp_frc_nc,
                                     out_crf=out_crf)

                # add crf to map
                shared.add_raster_to_map(in_ras=out_crf)

        except Exception as e:
            arcpy.AddError('Could not visualise Satellite Fraction rasters. See messages.')
            arcpy.AddMessage(str(e))
            return

        # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    arcpy.SetProgressor('default', 'Cleaning up environment...')

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
