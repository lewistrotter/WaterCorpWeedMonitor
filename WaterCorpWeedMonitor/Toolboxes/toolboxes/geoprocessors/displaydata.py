
import os
import json
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import arcpy

from scripts import shared


def execute(
        parameters
        # messages # TODO: implement
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    in_project_file = parameters[0].valueAsText
    in_flight_datetime = parameters[1].value
    in_layers_to_visualise = parameters[2].valueAsText

    # inputs for testing only
    # in_project_file = r'C:\Users\Lewis\Desktop\testing\city beach demo\meta.json'
    # in_flight_datetime = '2023-02-08 10:22:09'
    # in_layers_to_visualise = "'UAV RGB';'UAV NDVI';'UAV Classified';'S2 NDVI';'S2 Fractions'"

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
            arcpy.AddError('Project folder is missing expected folders.')
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
    #rehab_start_date = meta.get('date_rehab')
    #if rehab_start_date is None:
        #arcpy.AddError('Project has no start of rehab date.')
        #raise  # return

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
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARING INPUT PARAMETERS

    arcpy.SetProgressor('default', 'Preparing input parameters...')

    # parse layers from parameter
    requested_layers = [e for e in in_layers_to_visualise.split(';')]
    requested_layers = [e.replace("'", '').strip() for e in requested_layers]
    if len(requested_layers) == 0:
        arcpy.AddError('No layers requested..')
        raise  # return

    # build visualise folder
    visualise_folder = os.path.join(in_project_folder, 'visualise')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region VISUALISE UAV RASTER IF REQUESTED

    arcpy.SetProgressor('default', 'Visualising requested UAV capture data...')

    # init visualise uav
    visualise_uav_data = False

    # check if any requested layers are for uav data
    for lyr in requested_layers:
        if 'UAV' in lyr:
            visualise_uav_data = True
            break

    # read uav raster if requested
    if visualise_uav_data:

        # build capture folder and band folder
        capture_folder = os.path.join(in_project_folder, 'uav_captures', meta_item['capture_folder'])
        bands_folder = os.path.join(capture_folder, 'bands')
        classify_folder = os.path.join(capture_folder, 'classify')

        # create raw band map to maintain band order
        band_map = {
            'blue': os.path.join(bands_folder, 'blue.tif'),
            'green': os.path.join(bands_folder, 'green.tif'),
            'red': os.path.join(bands_folder, 'red.tif'),
            'redge': os.path.join(bands_folder, 'redge.tif'),
            'nir': os.path.join(bands_folder, 'nir.tif'),
        }

        # check if bands exist
        for v in band_map.values():
            if not os.path.exists(v):
                arcpy.AddError('Some required UAV bands missing from capture folder.')
                raise  # return

        # visualise uav capture rgb if requested
        if 'UAV RGB' in requested_layers:
            try:
                # open bands as seperate rasters
                red = arcpy.Raster(band_map['red'])
                green = arcpy.Raster(band_map['green'])
                blue = arcpy.Raster(band_map['blue'])

                # combine bands into composite
                tmp = arcpy.ia.CompositeBand([red, green, blue])

                # save to visualise folder
                out_fn = os.path.join(visualise_folder, 'uav_rgb.tif')
                tmp.save(out_fn)

                # add to map
                shared.add_raster_to_map(in_ras=out_fn)

            except Exception as e:
                arcpy.AddWarning('Could not read raster bands. See messages.')
                arcpy.AddMessage(str(e))
                #raise  # return

        # visualise uav capture ndvi if requested
        if 'UAV NDVI' in requested_layers:
            try:
                # open bands as seperate rasters
                red = arcpy.Raster(band_map['red'])
                nir = arcpy.Raster(band_map['nir'])

                # calculate ndvi
                tmp = (nir - red) / (nir + red)

                # save to visualise folder
                out_fn = os.path.join(visualise_folder, 'uav_ndvi.tif')
                tmp.save(out_fn)

                # add to map and update symbology
                shared.add_raster_to_map(in_ras=out_fn)
                shared.apply_ndvi_layer_symbology(in_ras=out_fn)

            except Exception as e:
                arcpy.AddWarning('Could not read raster bands. See messages.')
                arcpy.AddMessage(str(e))
                #raise  # return

        # visualise uav capture classified if requested
        if 'UAV Classified' in requested_layers:
            try:
                # create and check optimal rf tif exists
                class_tif = os.path.join(classify_folder, 'rf_optimal.tif')
                if not os.path.exists(class_tif):
                    arcpy.AddError('Classified UAV capture image does not exist.')
                    raise  # return

                # open bands as seperate rasters
                tmp = arcpy.Raster(class_tif)

                # save to visualise folder
                out_fn = os.path.join(visualise_folder, 'uav_classified.tif')
                tmp.save(out_fn)

                # add to map and update symbology
                shared.add_raster_to_map(in_ras=out_fn)
                shared.apply_uav_classified_layer_symbology(in_ras=out_fn)

            except Exception as e:
                arcpy.AddWarning('Could not read classified raster. See messages.')
                arcpy.AddMessage(str(e))
                #raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region VISUALISE SENTINEL 2 IF REQUESTED

    arcpy.SetProgressor('default', 'Visualising requested Sentinel 2 data...')

    # init visualise uav
    visualise_s2_data = False

    # check if any requested layers are for uav data
    for lyr in requested_layers:
        if 'S2' in lyr:
            visualise_s2_data = True
            break

    # read sentinel 2 netcdf if requested
    if visualise_s2_data:

        # build sat captures folder and combine netcdfs folder
        sat_folder = os.path.join(in_project_folder, 'sat_captures')
        combine_ncs_folder = os.path.join(sat_folder, 'cmb_ncs')

        # build capture folder and fractions folder
        capture_folder = os.path.join(in_project_folder, 'uav_captures', meta_item['capture_folder'])
        fractions_folder = os.path.join(capture_folder, 'fractions')

        # TODO: check exists

        # visualise sentinel 2 ndvi time series if requested
        if 'S2 NDVI' in requested_layers:
            try:
                # load combined netcdf
                cmb_ncs = os.path.join(combine_ncs_folder, 'raw_monthly_meds.nc')
                with xr.open_dataset(cmb_ncs) as ds:
                    ds.load()

                if 'time' not in ds or 'x' not in ds or 'y' not in ds:
                    arcpy.AddError('Sentinel 2 NetCDF is not compatible.')
                    raise  # return

                if len(ds['time']) == 0:
                    arcpy.AddError('No time dimension detected in Sentinel 2 NetCDF.')
                    raise  # return

                # extract netcdf attributes
                ds_attrs = ds.attrs
                ds_band_attrs = ds[list(ds)[0]].attrs
                ds_spatial_ref_attrs = ds['spatial_ref'].attrs

                # calculate ndvi for each slice
                ds_ndvi = (ds['nbart_nir_1'] - ds['nbart_red']) / (ds['nbart_nir_1'] + ds['nbart_red'])
                ds_ndvi = ds_ndvi.to_dataset(name='ndvi')

                # append attributes back on
                ds_ndvi.attrs = ds_attrs
                ds_ndvi['spatial_ref'].attrs = ds_spatial_ref_attrs
                for var in ds_ndvi:
                    ds_ndvi[var].attrs = ds_band_attrs

                # export combined monthly median to scratch
                out_nc = os.path.join(arcpy.env.scratchFolder, 'ndvi_monthly_meds.nc')
                ds_ndvi.to_netcdf(out_nc)
                ds_ndvi.close()

                # convert netcdf to crf and export to visualise folder
                out_fn = os.path.join(visualise_folder, 's2_ndvi.crf')
                arcpy.management.CopyRaster(in_raster=out_nc,
                                            out_rasterdataset=out_fn)

                # add to map
                shared.add_raster_to_map(in_ras=out_fn)
                shared.apply_ndvi_layer_symbology(in_ras=out_fn)

            except Exception as e:
                arcpy.AddWarning('Could not read Sentinel 2 NetCDF. See messages.')
                arcpy.AddMessage(str(e))
                #raise  # return

        # visualise uav capture rgb if requested
        if 'S2 Fractions' in requested_layers:
            try:
                # check fractional folder exists
                if not os.path.exists(fractions_folder):
                    arcpy.AddError('Fractions folder does not exist.')
                    raise  # return

                # check we have something in it
                if len(os.listdir(fractions_folder)) == 0:
                    arcpy.AddError('No fractional layers available.')
                    raise  # return

                # get list of folders (dates) in fractional folder
                frac_dates = sorted(os.listdir(fractions_folder))

                # build list of fractional map dates and paths
                frac_data = []
                for frac_date in frac_dates:
                    # build current folder path
                    frac_folder = os.path.join(fractions_folder, frac_date)

                    # init new fraction map
                    frac_map = {
                        'native': None,
                        'weed': None,
                        'other': None
                    }

                    # get all tif in current frac folder and build map
                    files = os.listdir(frac_folder)
                    for file in files:
                        if file.endswith('.tif'):
                            if 'native' in file:
                                frac_map['native'] = os.path.join(frac_folder, file)
                            elif 'weed' in file:
                                frac_map['weed'] = os.path.join(frac_folder, file)
                            elif 'other' in file:
                                frac_map['other'] = os.path.join(frac_folder, file)

                    # append to list if all tifs found, else warm
                    if None not in list(frac_map.values()):
                        frac_data.append({frac_date: frac_map})
                    else:
                        arcpy.AddWarning(f'Fractional month {frac_date} missing layers.')

                # check we got something, error otherwise
                if len(frac_data) == 0:
                    arcpy.AddError('No fractional layers available.')
                    raise  # return

                # iter each fractional month date...
                all_ncs = []
                for item in frac_data:
                    # unpack values
                    frac_date = list(item.keys())[0]
                    frac_map = list(item.values())[0]

                    # init current ncs
                    new_ncs = []

                    # iter native, weed, other...
                    for tif_name, tif_path in frac_map.items():
                        # convert tif to nc and store in scratch
                        tmp_nc = os.path.join(arcpy.env.scratchFolder, f'tmp_{tif_name}.nc')
                        arcpy.management.CopyRaster(in_raster=tif_path,
                                                    out_rasterdataset=tmp_nc)

                        # read netcdf as xr
                        with xr.open_dataset(tmp_nc) as ds_tmp:
                            ds_tmp.load()

                        # set up crs info
                        for band in ds_tmp:
                            if len(ds_tmp[band].shape) == 0:
                                crs_name = band
                                crs_wkt = str(ds_tmp[band].attrs.get('spatial_ref'))
                                ds_tmp = ds_tmp.drop_vars(crs_name)
                                break

                        # append crs info back on
                        ds_tmp = ds_tmp.assign_coords({'spatial_ref': 32750})
                        ds_tmp['spatial_ref'].attrs = {
                            'spatial_ref': crs_wkt,
                            'grid_mapping_name': crs_name
                        }

                        # append time dimension on
                        if 'time' not in ds_tmp:
                            dt = pd.to_datetime(frac_date + '-01', format='%Y-%m-%d')
                            ds_tmp = ds_tmp.assign_coords({'time': dt.to_numpy()})
                            ds_tmp = ds_tmp.expand_dims('time')

                        # append
                        for dim in ds_tmp.dims:
                            if dim in ['x', 'y', 'lat', 'lon']:
                                ds_tmp[dim].attrs = {
                                    'resolution': np.mean(np.diff(ds_tmp[dim])),
                                    'crs': f'EPSG:{32750}'
                                }

                        # append
                        for i, band in enumerate(ds_tmp):
                            ds_tmp[band].attrs = {
                                'units': '1',
                                'crs': f'EPSG:{32570}',
                                'grid_mapping': 'spatial_ref',
                            }

                            # rename band
                            ds_tmp = ds_tmp.rename({band: tif_name})

                        # append
                        ds_tmp.attrs = {
                            'crs': f'EPSG:{32570}',
                            'grid_mapping': 'spatial_ref'
                        }

                        # append
                        new_ncs.append(ds_tmp)

                        # delete nc
                        arcpy.management.Delete(tmp_nc)

                    # combine datasets into one
                    ds = xr.merge(new_ncs)
                    ds.close()

                    # append it
                    all_ncs.append(ds)

                # combine fraction ncs into one
                ds = xr.concat(all_ncs, 'time').sortby('time')

                # export combined fractionals to scratch
                out_nc = os.path.join(arcpy.env.scratchFolder, 's2_fracs.nc')
                ds.to_netcdf(out_nc)
                ds.close()

                # create a crf version of fractional netcdf
                out_fn = os.path.join(visualise_folder, 's2_fracs.crf')
                arcpy.management.CopyRaster(in_raster=out_nc,
                                            out_rasterdataset=out_fn)

                # add to map
                shared.add_raster_to_map(in_ras=out_fn)
                shared.apply_fraction_layer_symbology(in_ras=out_fn)

            except Exception as e:
                arcpy.AddWarning('Could not read Sentinel 2 NetCDF. See messages.')
                arcpy.AddMessage(str(e))
                #raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    # TODO: enable if move to non-memory temp files
    # try:
    #     # drop temp files (free up space)
    #     arcpy.management.Delete(tmp_comp)
    #
    # except Exception as e:
    #     arcpy.AddWarning('Could not drop temporary files. See messages.')
    #     arcpy.AddMessage(str(e))

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