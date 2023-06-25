def execute(
        parameters
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region IMPORTS

    import os
    import numpy as np
    import multiprocessing as mp
    import arcpy

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import as_completed

    from scripts import web, shared

    # set data overwrites and mapping
    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = False

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    in_out_folder = parameters[0].valueAsText

    # inputs for testing only
    #in_out_folder = r'C:\Users\Lewis\Desktop\working'

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK SPATIAL ANALYST WORKS

    arcpy.SetProgressor('default', 'Checking spatial analyst extension...')

    # check if user has spatial/image analyst, error if not
    if arcpy.CheckExtension('Spatial') != 'Available':
        arcpy.AddError('Spatial Analyst license is unavailable.')
        pass

    try:
        # try check out
        arcpy.CheckOutExtension('Spatial')

    except Exception as e:
        arcpy.AddError('Spatial Analyst could not be checked out.')
        arcpy.AddMessage(str(e))
        pass

    try:
        # create extent (city beach)
        srs = arcpy.SpatialReference(32750)
        ext = arcpy.Extent(XMin=382371.76,
                           YMin=6466469.15,
                           XMax=382983.01,
                           YMax=6467053.82,
                           spatial_reference=srs)

        # create raster of 1s
        tmp_raster = arcpy.sa.CreateConstantRaster(constant_value=1,
                                                   data_type='INTEGER',
                                                   cell_size=5,
                                                   extent=ext)

        # save it
        tmp_raster.save(os.path.join(in_out_folder, 'rand_raster.tif'))

    except Exception as e:
        arcpy.AddError('Spatial Analyst Resample tool could not run. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK IMAGE ANALYST WORKS

    arcpy.SetProgressor('default', 'Checking image analyst extension...')

    # check if user has spatial/image analyst, error if not
    if arcpy.CheckExtension('ImageAnalyst') != 'Available':
        arcpy.AddError('Image Analyst license is unavailable.')
        return

    try:
        # try check out
        arcpy.CheckOutExtension('ImageAnalyst')

    except Exception as e:
        arcpy.AddError('Image Analyst could not be checked out.')
        arcpy.AddMessage(str(e))
        pass

    try:
        # create extent (city beach)
        srs = arcpy.SpatialReference(32750)
        ext = arcpy.Extent(XMin=382371.76,
                           YMin=6466469.15,
                           XMax=382983.01,
                           YMax=6467053.82,
                           spatial_reference=srs)

        # create raster of 1s
        tmp_raster = arcpy.sa.CreateConstantRaster(constant_value=1,
                                                   data_type='INTEGER',
                                                   cell_size=5,
                                                   extent=ext)

        # resample it raster of 1s
        tmp_resample = arcpy.ia.Resample(raster=tmp_raster,
                                         resampling_type='Cubic',
                                         output_cellsize=10)

        # save it
        tmp_resample.save(os.path.join(in_out_folder, 'rand_raster_rs.tif'))

    except Exception as e:
        arcpy.AddError('Spatial Analyst Resample tool could not run. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region TEST DEA STAC SENTINEL 2 DOWNLOADS

    arcpy.SetProgressor('default', 'Testing Digital Earth Australia servers...')

    # set sentinel 2 data date range
    start_date, end_date = '2020-01-01', '2021-12-31'

    # set dea sentinel 2 collection 3 names
    collections = [
        'ga_s2am_ard_3',
        'ga_s2bm_ard_3'
    ]

    # get bounding box in wgs84 for stac query
    stac_bbox = (115.756098001898, -31.9317218134778, 115.761672256462, -31.9263899178077)
    if len(stac_bbox) != 4:
        arcpy.AddError('Could not generate STAC bounding box.')
        return

    try:
        # get all stac features from 2017 to now
        stac_features = web.fetch_all_stac_features(collections=collections,
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    bbox=stac_bbox,
                                                    limit=100)

    except Exception as e:
        arcpy.AddMessage(str(e))
        arcpy.AddError('Unable to fetch STAC features. See messages.')
        return

    # check if anything came back, warning if not
    if len(stac_features) == 0:
        arcpy.AddWarning('No STAC Sentinel 2 scenes were found.')
        return []

    # set desired sentinel 2 bands
    assets = [
        'nbart_blue',
        'nbart_green',
        'nbart_red',
        'nbart_red_edge_1',
        'nbart_red_edge_2',
        'nbart_red_edge_3',
        'nbart_nir_1',
        'nbart_nir_2',
        'nbart_swir_2',
        'nbart_swir_3'
    ]

    # get bounding box in wgs84 for stac query
    out_bbox = (-1518267.7245241, -3576309.07183537, -1517647.20134446, -3575715.52792441)
    if len(out_bbox) != 4:
        arcpy.AddError('Could not generate output bounding box.')
        return

    # add 30 metres on every side to prevent gaps
    out_bbox = shared.expand_box_by_metres(bbox=out_bbox, metres=30)

    # set raw output nc folder (one nc per date)
    raw_ncs_folder = os.path.join(in_out_folder, 'raw_ncs')
    if not os.path.exists(raw_ncs_folder):
        os.mkdir(raw_ncs_folder)

    try:
        # prepare downloads from raw stac features
        downloads = web.convert_stac_features_to_downloads(features=stac_features,
                                                           assets=assets,
                                                           out_bbox=out_bbox,
                                                           out_epsg=3577,
                                                           out_res=10,
                                                           out_path=raw_ncs_folder,
                                                           out_extension='.nc')

    except Exception as e:
        arcpy.AddError('Unable to convert STAC features to downloads. See messages.')
        arcpy.AddMessage(str(e))
        return

    # group downloads captured on same solar day
    downloads = web.group_downloads_by_solar_day(downloads=downloads)
    if len(downloads) == 0:
        arcpy.AddWarning('No valid downloads were found.')
        return []

    # remove downloads if current month (we want complete months)
    downloads = web.remove_downloads_for_current_month(downloads)
    if len(downloads) == 0:
        arcpy.AddWarning('Not enough downloads in current month exist yet.')
        return []

    # get existing netcdfs and convert to dates
    exist_dates = []
    for file in os.listdir(raw_ncs_folder):
        if file != 'monthly_meds.nc' and file.endswith('.nc'):
            file = file.replace('R', '').replace('.nc', '')
            exist_dates.append(file)

    # remove downloads that already exist in sat folder
    if len(exist_dates) > 0:
        downloads = web.remove_existing_downloads(downloads, exist_dates)

        # if nothing left, leave
        if len(downloads) == 0:
            arcpy.AddWarning('No new satellite downloads were found.')
            return []


    arcpy.SetProgressor('step', 'Downloading Sentinel 2 data...', 0, len(downloads), 1)

    # set relevant download parameters
    num_cpu = int(np.ceil(mp.cpu_count() / 2))

    try:
        i = 0
        results = []
        with ThreadPoolExecutor(max_workers=num_cpu) as pool:
            futures = []
            for download in downloads:
                task = pool.submit(web.validate_and_download,
                                   download,
                                   [1],   # quality_flags,
                                   1,     # max_out_of_bounds,
                                   0,     # max_invalid_pixels,
                                   -999)  # nodata_value
                futures.append(task)

            for future in as_completed(futures):
                arcpy.AddMessage(future.result())
                results.append(future.result())

                i += 1
                if i % 1 == 0:
                    arcpy.SetProgressorPosition(i)

    except Exception as e:
        arcpy.AddError('Unable to download Sentinel 2 data from DEA. See messages.')
        arcpy.AddMessage(str(e))
        return

    # check if any valid downloads (non-cloud or new)
    num_valid_downlaods = len([dl for dl in results if 'success' in dl])
    if num_valid_downlaods == 0:
        arcpy.AddMessage('No new valid satellite downloads were found.')
        return

    # endregion

    return

# testing
#execute(None)
