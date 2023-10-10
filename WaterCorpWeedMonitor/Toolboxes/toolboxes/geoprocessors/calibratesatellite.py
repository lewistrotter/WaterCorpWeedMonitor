
def execute(
        parameters
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region IMPORTS

    import os
    import json
    import warnings
    import datetime
    import xarray as xr
    import arcpy

    from scripts import web, shared

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
    in_operation = parameters[1].valueAsText
    in_x_shift = parameters[2].value
    in_y_shift = parameters[3].value

    # inputs for testing only
    # in_project_file = r'C:\Users\Lewis\Desktop\testing\citybeach\meta.json'
    # in_operation = 'Shift'  # 'Reset'
    # in_x_shift = -2.5
    # in_y_shift = 0.0

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

    # check if any captures exist (will be >= 6), else error
    if len(meta) < 6:
        arcpy.AddError('Project has no UAV capture data.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region ENSURE FRACTIONS HAVE NOT BEEN GENERATED ALREADY

    arcpy.SetProgressor('default', 'Checking if fractions already exist...')

    # exclude top-level metadata items
    exclude_keys = ['project_name', 'date_created', 'date_rehab', 'sat_shift_x', 'sat_shift_y']

    # extract selected metadata item based on capture date
    meta_item = None
    for k, v in meta.items():
        if k not in exclude_keys:
            if len(v['fractions']) > 0:
                arcpy.AddError('Cannot shift satellite after fractions have been generated.')
                return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region RESET SHIFT IF REQUESTED

    arcpy.SetProgressor('default', 'Removing existing shift settings...')

    if in_operation == 'Reset':
        # reset shift values
        meta['sat_shift_x'] = 0.0
        meta['sat_shift_y'] = 0.0

        try:
            # write json metadata file to project folder top-level
            with open(in_project_file, 'w') as fp:
                json.dump(meta, fp)

        except Exception as e:
            arcpy.AddError('Could not write metadata. See messages.')
            arcpy.AddMessage(str(e))
            return

        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region FETCH CLEAN DEA STAC SENTINEL 2 DOWNLOADS

    arcpy.SetProgressor('default', 'Fetching clean DEA STAC downloads...')

    try:
        # set query date range, collections and assets
        start_date, end_date = '2020-01-01', '2020-12-31'

        # read grid as raster
        tmp_grd = arcpy.Raster(grid_tif)

        # get stac and output coordinate bbox based on grid exent
        stac_bbox = shared.get_raster_bbox(in_ras=tmp_grd, out_epsg=4326)

        # get output netcdf bbox in albers and expand
        out_bbox = shared.get_raster_bbox(in_ras=tmp_grd, out_epsg=3577)
        out_bbox = shared.expand_bbox(bbox=out_bbox, by_metres=50.0)

        # set output folder for raw sentinel 2 cubes and check
        raw_ncs_folder = os.path.join(tmp, 'raw_ncs')
        if not os.path.exists(raw_ncs_folder):
            os.mkdir(raw_ncs_folder)

        # query and prepare downloads
        downloads = web.quick_fetch(start_date=start_date,
                                    end_date=end_date,
                                    stac_bbox=stac_bbox,
                                    out_bbox=out_bbox,
                                    out_folder=raw_ncs_folder)

    except Exception as e:
        arcpy.AddError('Unable to download Sentinel 2 data from DEA. See messages.')
        arcpy.AddMessage(str(e))
        return

    # check if downloads returned (should always find something), else leave
    if len(downloads) == 0:
        arcpy.AddError('No valid satellite downloads found. Check your firewall.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region DOWNLOAD WCS DATA

    arcpy.SetProgressor('default', 'Downloading Sentinel 2 data...')

    try:
        # download everything and return success or fail statuses
        results = web.quick_download(downloads=downloads,
                                     quality_flags=[1],
                                     max_out_of_bounds=30,
                                     max_invalid_pixels=30,
                                     nodata_value=-999)

    except Exception as e:
        arcpy.AddError('Unable to download Sentinel 2 data from DEA. See messages.')
        arcpy.AddMessage(str(e))
        return

    # count number of valid downloads returned
    num_valid_downloads = len([dl for dl in results if 'success' in dl])

    # check if any valid downloads (non-cloud or new)
    if num_valid_downloads == 0:
        arcpy.AddError('No valid satellite downloads were found.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region VALIDATE SENTINEL 2 NETCDFS

    arcpy.SetProgressor('default', 'Validating Sentinel 2 data...')

    try:
        # check results for errors and delete errorneous nc files
        web.delete_error_downloads(results=results,
                                   nc_folder=raw_ncs_folder)

    except Exception as e:
        arcpy.AddError('Unable to delete errorneous Sentinel 2 data. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region COMBINE SENTINEL 2 NETCDFS

    arcpy.SetProgressor('default', 'Combining Sentinel 2 data...')

    # get all raw nc dates in raw netcdf folder
    nc_files = []
    for file in os.listdir(raw_ncs_folder):
        if file.startswith('R') and file.endswith('.nc'):
            nc_files.append(os.path.join(raw_ncs_folder, file))

    # check if anything came back, error if not
    if len(nc_files) == 0:
        arcpy.AddError('No NetCDF files were found.')
        return

    try:
        # validate ncs to ensure all conform to expected
        nc_files = shared.validate_ncs(nc_list=nc_files)

        # read all netcdfs into single dataset
        ds = shared.concat_netcdf_files(nc_files=nc_files)

    except Exception as e:
        arcpy.AddError('Unable to combine Sentinel 2 NetCDFs. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CLEAN SENTINEL 2 NETCDFS

    arcpy.SetProgressor('default', 'Cleaning Sentinel 2 data...')

    try:
        # extract netcdf attributes
        ds_attrs = ds.attrs
        ds_band_attrs = ds[list(ds)[0]].attrs
        ds_spatial_ref_attrs = ds['spatial_ref'].attrs

        # set nodata (-999) to nan
        ds = ds.where(ds != -999)

        # subset to just rgb
        ds = ds[['nbart_red', 'nbart_green', 'nbart_blue']]

        # resample to all-time median
        ds = ds.resample(time='1YS').median()

        # append attributes back on
        ds.attrs = ds_attrs
        ds['spatial_ref'].attrs = ds_spatial_ref_attrs
        for var in ds:
            ds[var].attrs = ds_band_attrs

    except Exception as e:
        arcpy.AddError('Unable to clean Sentinel 2 NetCDFs. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region SHIFT SENTINEL 2 NETCDF

    arcpy.SetProgressor('default', 'Shifting Sentinel 2 data...')

    try:
        # shift x, y coordinates via input
        ds['x'] = ds['x'] + in_x_shift
        ds['y'] = ds['y'] + in_y_shift

    except Exception as e:
        arcpy.AddError('Unable to shift Sentinel 2 NetCDF. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXPORT CLEAN SENTINEL 2 NETCDF

    arcpy.SetProgressor('default', 'Exporting clean Sentinel 2 data...')

    # set combined output nc folder
    cmb_ncs_folder = os.path.join(tmp, 'cmb_ncs')
    if not os.path.exists(cmb_ncs_folder):
        os.mkdir(cmb_ncs_folder)

    try:
        # export combined monthly median as new netcdf
        out_annual_med_nc = os.path.join(cmb_ncs_folder, 'raw_annual_med.nc')
        ds.to_netcdf(out_annual_med_nc)
        ds.close()

    except Exception as e:
        arcpy.AddError('Unable to export clean Sentinel 2 NetCDF. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region UPDATE NEW SHIFT INFO IN METADATA

    arcpy.SetProgressor('default', 'Updating metadata...')

    # update metdata variables
    meta['sat_shift_x'] = in_x_shift
    meta['sat_shift_y'] = in_y_shift

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
    # region ADD RGB SHIFT DATA TO ACTIVE MAP

    arcpy.SetProgressor('default', 'Visualising result...')

    # build visualise folder path and
    visualise_folder = os.path.join(in_project_folder, 'visualise')

    try:
        # re-load the netcdf
        ds = xr.open_dataset(out_annual_med_nc)

        # squeeze time off
        ds = ds.squeeze(drop=True)

        # convert netcdf to rgb geotiff
        tmp_ras = shared.multi_band_xr_to_raster(da=ds,
                                                 out_folder=tmp)

        # create tif path to visualise folder
        out_tif = os.path.join(visualise_folder, f'sft_rgb.tif')

        # delete previously created visual tif
        shared.delete_visual_rasters(rasters=[out_tif])

        # re-save it
        tmp_ras.save(out_tif)

        # add tif to map
        shared.add_raster_to_map(in_ras=out_tif)

    except Exception as e:
        arcpy.AddWarning('Could not visualise shifted RGB image. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    arcpy.SetProgressor('default', 'Cleaning up environment...')

    try:
        # close temp files
        del tmp_grd
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
