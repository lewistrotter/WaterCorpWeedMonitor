
import os
import pandas as pd
import numpy as np
import xarray as xr
import arcpy

from osgeo import gdal
from scipy.stats import kendalltau


def add_raster_to_map(
        in_ras: str
) -> None:
    """
    Takes any raster file path and visualises it in arcgis pro if active map available.

    :param in_ras: Raster file path.
    :return: None.
    """

    # re-enable add to map
    arcpy.env.addOutputsToMap = True

    try:
        # read current project and add tif
        aprx = arcpy.mp.ArcGISProject('CURRENT')
        mp = aprx.activeMap
        mp.addDataFromPath(in_ras)

    except Exception as e:
        raise e

    # disable add to map
    arcpy.env.addOutputsToMap = False

    return


def apply_classified_symbology(
        in_ras: str
) -> None:

    # re-enable add to map
    arcpy.env.addOutputsToMap = True

    try:
        # read current project and map
        aprx = arcpy.mp.ArcGISProject('CURRENT')
        mp = aprx.activeMap

        # get all layers ona ctive map
        lyrs = mp.listLayers()

        # find layer that has same data source as input
        lyr = None
        for lyr in lyrs:
            if lyr.dataSource == in_ras:
                break

        # return if no layer found
        if lyr is None:
            raise ValueError('Could not find UAV classified layer on map.')

        # get symbology from layer and force symbology type
        sym = lyr.symbology
        sym.updateColorizer('RasterUniqueValueColorizer')
        sym.colorizer.classificationField = 'Class_name'

        # iterate classes and update color mapping
        for group in sym.colorizer.groups:
            for item in group.items:
                if item.label == 'Native':
                    item.color = {'RGB': [17, 99, 39, 100]}
                elif item.label == 'Weed':
                    item.color = {'RGB': [242, 132, 128, 100]}
                elif item.label == 'Other':
                    item.color = {'RGB': [255, 239, 204, 100]}

        # update symbology
        lyr.symbology = sym

    except Exception as e:
        raise e

    # disable add to map
    arcpy.env.addOutputsToMap = False

    return


def convert_xr_vars_to_raster(
        da: xr.Dataset,
) -> list:
    """
    Takes a 2D xarray Dataset object and returns a single GeoTiff
    where each band represents a variable in the input NetCDF. Working
    is done in termporary scratch folder

    :param da: xarray Dataset.
    :return: arcpy.Raster composite.
    """

    # get scratch folder
    scratch_folder = arcpy.env.scratchFolder

    # init out band list
    out_band_tifs = []

    try:
        # convert each dataset var to a real tif band
        for var in list(da.data_vars):
            # set up output nc and tif files in scratch
            out_nc = os.path.join(scratch_folder, f'{var}.nc')
            out_tif = os.path.join(scratch_folder, f'{var}.tif')

            # export current var as netcdf
            da[var].to_netcdf(out_nc)

            # convert netcdf to raster
            dataset = gdal.Open(out_nc, gdal.GA_ReadOnly)
            dataset = gdal.Translate(out_tif, dataset)
            dataset = None

            # append tif to bands list
            out_band_tifs.append(out_tif)

        # composite
        out_ras = arcpy.sa.CompositeBand(out_band_tifs)

    except Exception as e:
        raise e

    return out_ras


def get_raster_bbox(
        in_ras: arcpy.Raster,
        out_epsg: int = None
) -> tuple:
    """
    Takes an ArcPy Raster object and extracts
    coordinate bbox from it. Coordinates in bbox
    are in order x_min, y_min, x_max, y_max.
    Optionally can reproject bbox to new coordinates.

    :param in_ras: ArcPy Raster object.
    :param out_epsg: EPSG to reproject bbox to.
    :return: Bbox tuple.
    """

    try:
        # create copy raster
        tmp_ras = in_ras

        # reproject if requested
        if out_epsg is not None:
            tmp_ras = arcpy.ia.Reproject(tmp_ras, {'wkid': out_epsg})

        # get description object from input raster
        extent = arcpy.Describe(tmp_ras).extent

        # extract sw, ne corners
        x_min, y_min = float(extent.XMin), float(extent.YMin)
        x_max, y_max = float(extent.XMax), float(extent.YMax)

        # combine int uple
        bbox = x_min, y_min, x_max, y_max

    except:
        return ()

    return bbox


def expand_bbox(
        bbox: tuple,
        by_metres: float
) -> tuple:
    """
    Expands bbox extent by given value. Expected
    that input is in metres.

    :param bbox: Bbox tuple.
    :return: Expanded bbox in same units.
    """

    # unpack bbox
    x_min, y_min, x_max, y_max = bbox

    # minus 30 x and y min
    x_min -= by_metres
    y_min -= by_metres

    # plus x max and y max
    x_max += by_metres
    y_max += by_metres

    return x_min, y_min, x_max, y_max









def fix_xr_attrs(
        ds: xr.Dataset,
        out_vars: list,
        out_datetime: str,
        out_epsg: int
) -> xr.Dataset:
    """
    Fix xarray dataset metadata

    :param ds: Xarray Dataset.
    :return: Xarray Dataset.
    """

    for band in ds:
        if len(ds[band].shape) == 0:
            crs_name = band
            crs_wkt = str(ds[band].attrs.get('spatial_ref'))
            ds = ds.drop_vars(crs_name)
            break

    ds = ds.assign_coords({'spatial_ref': out_epsg})
    ds['spatial_ref'].attrs = {
        'spatial_ref': crs_wkt,
        'grid_mapping_name': crs_name
    }

    if 'time' not in ds:
        dt = pd.to_datetime(out_datetime, format='%Y-%m-%d')
        ds = ds.assign_coords({'time': dt.to_numpy()})
        ds = ds.expand_dims('time')

    for dim in ds.dims:
        if dim in ['x', 'y', 'lat', 'lon']:
            ds[dim].attrs = {
                # 'units': 'metre'  # TODO: how to get units?
                'resolution': np.mean(np.diff(ds[dim])),
                'crs': f'EPSG:{out_epsg}'
            }

    for i, band in enumerate(ds):
        ds[band].attrs = {
            'units': '1',
            # 'nodata': self.nodata,  TODO: implemented out_nodata
            'crs': f'EPSG:{out_epsg}',
            'grid_mapping': 'spatial_ref',
        }

        ds = ds.rename({band: out_vars[i]})

    ds.attrs = {
        'crs': f'EPSG:{out_epsg}',
        'grid_mapping': 'spatial_ref'
    }

    return ds
























def get_opcs(n):
    if n == 3:
        slope = [-1, 0, 1]
        lin_ss, lin_c = 2, 1
        curve = [1, -2, 1]
        crv_ss, crv_c = 6, 3
    elif n == 4:
        slope = [-3, -1, 1, 3]
        lin_ss, lin_c = 20, 2
        curve = [1, -1, -1, 1]
        crv_ss, crv_c = 4, 1
    elif n == 5:
        slope = [-2, -1, 0, 1, 2]
        lin_ss, lin_c = 10, 1
        curve = [2, -1, -2, -1, 2]
        crv_ss, crv_c = 14, 1
    elif n == 6:
        slope = [-5, -3, -1, 1, 3, 5]
        lin_ss, lin_c = 70, 2
        curve = [5, -1, -4, -4, -1, 5]
        crv_ss, crv_c = 84, 1.5
    elif n == 7:
        slope = [-3, -2, -1, 0, 1, 2, 3]
        lin_ss, lin_c = 28, 1
        curve = [5, 0, -3, -4, -3, 0, 5]
        crv_ss, crv_c = 84, 1
    elif n == 8:
        slope = [-7, -5, -3, -1, 1, 3, 5, 7]
        lin_ss, lin_c = 168, 2
        curve = [7, 1, -3, -5, -5, -3, 1, 7]
        crv_ss, crv_c = 168, 1
    elif n == 9:
        slope = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        lin_ss, lin_c = 60, 1
        curve = [28, 7, -8, -17, -20, -17, -8, 7, 28]
        crv_ss, crv_c = 2772, 3
    elif n == 10:
        slope = [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]
        lin_ss, lin_c = 330, 2
        curve = [6, 2, -1, -3, -4, -4, -3, -1, 2, 6]
        crv_ss, crv_c = 132, 0.5
    elif n == 11:
        slope = [-5 - 4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        lin_ss, lin_c = 110, 1
        curve = [15, 6, -1, -6, -9, -10, -9, -6, -1, 6, 15]
        crv_ss, crv_c = 858, 1
    elif n == 12:
        slope = [-11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11]
        lin_ss, lin_c = 572, 2
        curve = [55, 25, 1, -17, -29, -35, -35, -29, -17, 1, 25, 55]
        crv_ss, crv_c = 12012, 3
    else:
        raise ValueError('Number not supported.')

    return slope, lin_ss, lin_c, curve, crv_ss, crv_c


def apply_fit(vec, coeffs, ss, c):
    return np.sum((vec * coeffs)) / ss * c


def apply_mk_tau(vec):
    x = np.arange(len(vec))
    t, _ = kendalltau(x, vec)

    return t


def apply_mk_pvalue(vec):
    x = np.arange(len(vec))
    _, p = kendalltau(x, vec)

    return p








def apply_ndvi_layer_symbology(
        in_ras: str
) -> None:

    # re-enable add to map
    arcpy.env.addOutputsToMap = True

    try:
        # read current project and map
        aprx = arcpy.mp.ArcGISProject('CURRENT')
        mp = aprx.activeMap

        # get all layers ona ctive map
        lyrs = mp.listLayers()

        # find layer that has same data source as input
        lyr = None
        for lyr in lyrs:
            if lyr.dataSource == in_ras:
                break

        # return if no layer found
        if lyr is None:
            arcpy.AddWarning('Could not find UAV ndvi layer on map.')
            return

        # get symbology from layer and force symbology type
        sym = lyr.symbology
        sym.updateColorizer('RasterStretchColorizer')

        # apply stretch properties
        sym.colorizer.stretchType = "PercentClip"
        sym.colorizer.minPercent = 0.5
        sym.colorizer.maxPercent = 0.5

        # apply color ramp
        cramp = aprx.listColorRamps('Precipitation')[0]
        sym.colorizer.colorRamp = cramp
        sym.colorizer.invertColorRamp = False
        sym.colorizer.gamma = 1

        # update symbology
        lyr.symbology = sym

    except Exception as e:
        arcpy.AddWarning('Could not apply symbology to UAV ndvi layer. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # disable add to map
    arcpy.env.addOutputsToMap = False

    return


def apply_fraction_layer_symbology(
        in_ras: str
) -> None:

    # re-enable add to map
    arcpy.env.addOutputsToMap = True

    try:
        # read current project and map
        aprx = arcpy.mp.ArcGISProject('CURRENT')
        mp = aprx.activeMap

        # get all layers ona ctive map
        lyrs = mp.listLayers()

        # find layer that has same data source as input
        lyr = None
        for lyr in lyrs:
            if lyr.dataSource == in_ras:
                break

        # return if no layer found
        if lyr is None:
            arcpy.AddWarning('Could not find Sentinel 2 fractional layer on map.')
            return

        # get symbology from layer and force symbology type
        sym = lyr.symbology
        sym.updateColorizer('RasterStretchColorizer')

        # apply stretch properties
        sym.colorizer.stretchType = "PercentClip"
        sym.colorizer.minPercent = 0.25
        sym.colorizer.maxPercent = 0.25

        # apply color ramp
        cramp = aprx.listColorRamps('Inferno')[0]
        sym.colorizer.colorRamp = cramp
        sym.colorizer.invertColorRamp = False
        sym.colorizer.gamma = 1

        # update symbology
        lyr.symbology = sym

    except Exception as e:
        arcpy.AddWarning('Could not apply symbology to Sentinel 2 fractional layer. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # disable add to map
    arcpy.env.addOutputsToMap = False

    return