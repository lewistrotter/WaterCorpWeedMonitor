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
    in_freq = parameters[1].valueAsText

    # inputs for testing only
    #in_project_file = r'C:\Users\Lewis\Desktop\testing\lancelin\meta.json'
    #in_freq = 'Quarterly'

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
    # region READ AND CHECK SENTINEL 2 NETCDFS

    arcpy.SetProgressor('default', 'Reading and checking Sentinel 2 NetCDFs...')

    # get sat captures folder
    capture_folders = os.path.join(in_project_folder, 'sat_captures')
    raw_sat_folder = os.path.join(capture_folders, 'raw_ncs')

    # check raw sat folder exists
    if not os.path.exists(raw_sat_folder):
        arcpy.AddError('No satellite folder detected. Run fraction tool first.')
        return

    # get paths of all netcdfs
    ncs = []
    for root, dirs, files in os.walk(raw_sat_folder):
        for file in files:
            if file.endswith('.nc'):
                ncs.append(os.path.join(root, file))

    # ensure netcdfs were found, else abort
    if len(ncs) == 0:
        arcpy.AddError('No satellite NetCDFs returned. Run fraction tool first')
        return

    try:
        # read and concatnate all netcdfs into one
        ds = shared.concat_netcdf_files(ncs)

    except Exception as e:
        arcpy.AddError('Could not read all satellite NetCDFs. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT ATTRIBUTES FROM NETCDF

    try:
        # extract netcdf attributes
        ds_attrs = ds.attrs
        ds_band_attrs = ds[list(ds)[0]].attrs
        ds_spatial_ref_attrs = ds['spatial_ref'].attrs

    except Exception as e:
        arcpy.AddError('Unable to extract NetCDF attributes. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE NDVI

    try:
        # calculate ndvi
        ds['ndvi'] = ((ds['nbart_nir_1'] - ds['nbart_red']) /
                      (ds['nbart_nir_1'] + ds['nbart_red']))

        # isolate variable
        ds = ds[['ndvi']]

    except Exception as e:
        arcpy.AddError('Could not calculate NDVI. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region RESAMPLE NETCDFS BASED ON USER FREQUENCY

    arcpy.SetProgressor('default', 'Resampling NetCDF based on user frequency...')

    # set nodata (-999) to nan
    ds = ds.where(ds != -999)

    # perform resample
    if in_freq == 'Fortnightly':
        ds = ds.resample(time='SMS').median('time')
    elif in_freq == 'Monthly':
        ds = ds.resample(time='1MS').median('time')
    elif in_freq == 'Quarterly':
        ds = ds.resample(time='1QS').median('time')
    elif in_freq == 'Yearly':
        ds = ds.resample(time='1YS').median('time')
    else:
        raise ValueError('Frequency not supported.')

    # fill in nan values
    ds = ds.interpolate_na(dim='time')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region APPEND ATTRIBUTES TO NETCDF

    try:
        # append attributes back on
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds:
            ds[var].attrs = ds_band_attrs

    except Exception as e:
        arcpy.AddError('Unable to append NetCDF attributes. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region ADD NDVI LAYER TO ACTIVE MAP

    arcpy.SetProgressor('default', 'Visualising result...')

    # build visualise folder path and
    visualise_folder = os.path.join(in_project_folder, 'visualise')

    try:
        # export netcdf to tmp
        tmp_nc = os.path.join(tmp, 'tmp_ndvi.nc')
        ds.to_netcdf(tmp_nc)

        # create crf path to visualise folder
        out_crf = os.path.join(visualise_folder, f'ndvi.crf')

        # delete previously created visual crf
        shared.delete_visual_rasters(rasters=[out_crf])

        # re-save it
        shared.netcdf_to_crf(in_nc=tmp_nc,
                             out_crf=out_crf)

        # add crf to map
        shared.add_raster_to_map(in_ras=out_crf)

    except Exception as e:
        arcpy.AddWarning('Could not visualise NDVI image. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    arcpy.SetProgressor('default', 'Cleaning up environment...')

    try:
        # close netcdfs
        ds.close()

    except Exception as e:
        arcpy.AddWarning('Could not drop temporary files. See messages.')
        arcpy.AddMessage(str(e))

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