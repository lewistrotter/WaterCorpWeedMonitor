
import os
import shutil
import datetime
import arcpy
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from scripts import web


def test_license():

    try:
        product = arcpy.CheckProduct('ArcInfo')
        if product not in ['AlreadyInitialized', 'Available']:
            raise ValueError('Advanced license not available.')

    except Exception as e:
        raise e

    return


def test_extensions():

    try:
        if arcpy.CheckExtension('Spatial') != 'Available':
            raise ValueError('Spatial Analyst extension not available.')

        if arcpy.CheckExtension('ImageAnalyst') != 'Available':
            raise ValueError('Image Analyst extension not available.')

    except Exception as e:
        raise e

    return


def test_web(tmp_folder):

    # set sentinel 2 data date range
    start_date, end_date = '2020-01-01', '2021-12-31'

    # set dea sentinel 2 collection 3 names
    collections = [
        'ga_s2am_ard_3',
        'ga_s2bm_ard_3'
    ]

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
    stac_bbox = [
        115.756098001898,
        -31.9317218134778,
        115.761672256462,
        -31.9263899178077
    ]

    try:
        # get all stac features from 2017 to now
        stac_features = web.fetch_all_stac_features(collections=collections,
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    bbox=stac_bbox,
                                                    limit=100)

    except Exception as e:
        raise e

    # get bounding box in wgs84 for stac query
    out_bbox = [
        -1518267.7245241,
        -3576309.07183537,
        -1517647.20134446,
        -3575715.52792441
    ]

    # set raw output nc folder (one nc per date)
    raw_ncs_folder = os.path.join(tmp_folder, 'raw_ncs')
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

    # set relevant download parameters
    num_cpu = int(np.ceil(os.cpu_count() / 2))

    try:
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
                #arcpy.AddMessage(future.result())
                results.append(future.result())

    except Exception as e:
        raise e

    # check if any valid downloads (non-cloud or new)
    num_valid_downlaods = len([dl for dl in results if 'success' in dl])
    if num_valid_downlaods == 0:
        raise 'No downloads were successful. Check firewall.'

    # try delete folder
    try:
        shutil.rmtree(raw_ncs_folder)
    except:
        pass

    return


def test_createnewsite(in_folder, project_folder):

    # init params
    params = []

    p00 = arcpy.Parameter(name='in_output_folder')
    p00.value = project_folder
    params.append(p00)

    p01 = arcpy.Parameter(name='in_boundary_feat')
    p01.value = os.path.join(in_folder, 'studyarea.shp')
    params.append(p01)

    p02 = arcpy.Parameter(name='in_rehab_datetime')
    p02.value = datetime.datetime(2017, 3, 13, 11, 22, 15, 552082)
    params.append(p02)

    p03 = arcpy.Parameter(name='in_flight_datetime')
    p03.value = datetime.datetime(2022, 2, 2, 10, 30, 15, 652082)
    params.append(p03)

    p04 = arcpy.Parameter(name='in_blue_band')
    p04.value = os.path.join(in_folder, 'ms_ref_blue.tif')
    params.append(p04)

    p05 = arcpy.Parameter(name='in_green_band')
    p05.value = os.path.join(in_folder, 'ms_ref_green.tif')
    params.append(p05)

    p06 = arcpy.Parameter(name='in_red_band')
    p06.value = os.path.join(in_folder, 'ms_ref_red.tif')
    params.append(p06)

    p07 = arcpy.Parameter(name='in_redge_band')
    p07.value = os.path.join(in_folder, 'ms_ref_redge.tif')
    params.append(p07)

    p08 = arcpy.Parameter(name='in_nir_band')
    p08.value = os.path.join(in_folder, 'ms_ref_nir.tif')
    params.append(p08)

    p09 = arcpy.Parameter(name='in_dsm_band')
    p09.value = os.path.join(in_folder, 'ms_dsm.tif')
    params.append(p09)

    p10 = arcpy.Parameter(name='in_dtm_band')
    p10.value = os.path.join(in_folder, 'ms_dtm.tif')
    params.append(p10)

    try:
        # import geoprocessor
        from geoprocessors import createnewsite
        createnewsite.execute(params)

    except Exception as e:
        raise e


def test_ingestnewuavcapture(in_folder, project_folder):

    # init params
    params = []

    p00 = arcpy.Parameter(name='in_project_file')
    p00.value = os.path.join(project_folder, 'meta.json')
    params.append(p00)

    p01 = arcpy.Parameter(name='in_flight_datetime')
    p01.value = datetime.datetime(2023, 2, 2, 10, 30, 15, 652082)
    params.append(p01)

    p02 = arcpy.Parameter(name='in_blue_band')
    p02.value = os.path.join(in_folder, 'ms_ref_blue.tif')
    params.append(p02)

    p03 = arcpy.Parameter(name='in_green_band')
    p03.value = os.path.join(in_folder, 'ms_ref_green.tif')
    params.append(p03)

    p04 = arcpy.Parameter(name='in_red_band')
    p04.value = os.path.join(in_folder, 'ms_ref_red.tif')
    params.append(p04)

    p05 = arcpy.Parameter(name='in_redge_band')
    p05.value = os.path.join(in_folder, 'ms_ref_redge.tif')
    params.append(p05)

    p06 = arcpy.Parameter(name='in_nir_band')
    p06.value = os.path.join(in_folder, 'ms_ref_nir.tif')
    params.append(p06)

    p07 = arcpy.Parameter(name='in_dsm_band')
    p07.value = os.path.join(in_folder, 'ms_dsm.tif')
    params.append(p07)

    p08 = arcpy.Parameter(name='in_dtm_band')
    p08.value = os.path.join(in_folder, 'ms_dtm.tif')
    params.append(p08)

    try:
        # import geoprocessor
        from geoprocessors import ingestnewuavcapture
        ingestnewuavcapture.execute(params)

    except Exception as e:
        raise e


def test_classifyuavcapture(in_folder, in_date, project_folder):

    # init params
    params = []

    p00 = arcpy.Parameter(name='in_project_file')
    p00.value = os.path.join(project_folder, 'meta.json')
    params.append(p00)

    p01 = arcpy.Parameter(name='in_capture_datetime')
    p01.value = in_date
    params.append(p01)

    p02 = arcpy.Parameter(name='in_include_prior')
    p02.value = False
    params.append(p02)

    p03 = arcpy.Parameter(name='in_roi_feat')
    p03.value = os.path.join(in_folder, 'rois.shp')
    params.append(p03)

    p04 = arcpy.Parameter(name='in_variables')
    p04.value = 'NDVI;NDREI;NGRDI;OSAVI;Mean;Minimum;Maximum;StanDev;Range;CHM'
    params.append(p04)

    try:
        # import geoprocessor
        from geoprocessors import classifyuavcapture
        classifyuavcapture.execute(params)

    except Exception as e:
        raise e


def test_generatefractions(in_folder, project_folder):

    # init params
    params = []

    p00 = arcpy.Parameter(name='in_project_file')
    p00.value = os.path.join(project_folder, 'meta.json')
    params.append(p00)

    p01 = arcpy.Parameter(name='in_capture_datetime')
    p01.value = '2022-02-02 10:30:15'
    params.append(p01)

    try:
        # import geoprocessor
        from geoprocessors import generatefractions
        generatefractions.execute(params)

    except Exception as e:
        raise e


def test_generatetrend(in_folder, project_folder):

    # init params
    params = []

    p00 = arcpy.Parameter(name='in_project_file')
    p00.value = os.path.join(project_folder, 'meta.json')
    params.append(p00)

    p01 = arcpy.Parameter(name='in_capture_datetime')
    p01.value = '2022-02-02 10:30:15'
    params.append(p01)

    p02 = arcpy.Parameter(name='in_rehab_or_capture_month')
    p02.value = 'Month of Rehabilitation'
    params.append(p02)

    try:
        # import geoprocessor
        from geoprocessors import generatetrend
        generatetrend.execute(params)

    except Exception as e:
        raise e


def test_detectuavchange(in_folder, project_folder):

    # init params
    params = []

    p00 = arcpy.Parameter(name='in_project_file')
    p00.value = os.path.join(project_folder, 'meta.json')
    params.append(p00)

    p01 = arcpy.Parameter(name='in_uav_from_date')
    p01.value = '2022-02-02 10:30:15'
    params.append(p01)

    p01 = arcpy.Parameter(name='in_uav_to_date')
    p01.value = '2023-02-02 10:30:15'
    params.append(p01)

    try:
        # import geoprocessor
        from geoprocessors import detectuavchange
        detectuavchange.execute(params)

    except Exception as e:
        raise e


def test_detectfractionchange(in_folder, project_folder):

    # init params
    params = []

    p00 = arcpy.Parameter(name='in_project_file')
    p00.value = os.path.join(project_folder, 'meta.json')
    params.append(p00)

    p01 = arcpy.Parameter(name='in_capture_datetime')
    p01.value = '2022-02-02 10:30:15'
    params.append(p01)

    p02 = arcpy.Parameter(name='in_s2_from_year')
    p02.value = 2017
    params.append(p02)

    p03 = arcpy.Parameter(name='in_s2_to_year')
    p03.value = 2023
    params.append(p03)

    p04 = arcpy.Parameter(name='in_s2_month')
    p04.value = 2
    params.append(p04)

    try:
        # import geoprocessor
        from geoprocessors import detectfracchange
        detectfracchange.execute(params)

    except Exception as e:
        raise e
