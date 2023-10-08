
import os
import pandas as pd
import numpy as np
import xarray as xr
import arcpy

from osgeo import gdal


def clear_tmp_folder(
        tmp_folder: str
) -> None:
    """
    Clears temp folder.
    :param tmp_folder: Path to temp folder.
    :return: None.
    """

    # iter files in temp folder
    for file in os.listdir(tmp_folder):
        try:
            # try and delete current file via arcpy safely
            arcpy.management.Delete(in_data=os.path.join(tmp_folder, file))
        except:
            pass

    # iter files in temp folder
    for file in os.listdir(tmp_folder):
        try:
            # try and delete current file forcefully
            os.remove(os.path.join(tmp_folder, file))
        except:
            pass

    return


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
            # reproject to user srs via geoprocessor (ia func has issues)
            srs = arcpy.SpatialReference(out_epsg)
            tmp_ras = arcpy.management.ProjectRaster(in_raster=in_ras,
                                                     out_raster='tmp_bbx_prj.tif',
                                                     out_coor_system=srs)

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


def add_raster_to_map(
        in_ras: str
) -> None:
    """
    Takes any raster file path and visualises it in arcgis pro if
    active map available.

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
    """
    Takes any raster file path and visualises it in arcgis pro if
    active map available. Applies classified symbology for
    natives, weeds and others.

    :param in_ras: Raster file path.
    :return: None.
    """

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


def apply_uav_change_symbology(
        in_ras: str
) -> None:
    """
    Takes any raster file path and visualises it in arcgis pro if
    active map available. Applies change class symbology for
    natives, weeds and others. Use for UAV change only.

    :param in_ras: Raster file path.
    :return: None.
    """

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
            raise ValueError('Could not find UAV change layer on map.')

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
                elif item.label == 'Other->Native':
                    item.color = {'RGB': [0, 115, 76, 100]}
                elif item.label == 'Other->Weed':
                    item.color = {'RGB': [255, 170, 0, 100]}
                elif item.label == 'Native->Other':
                    item.color = {'RGB': [255, 255, 0, 100]}
                elif item.label == 'Native->Weed':
                    item.color = {'RGB': [255, 0, 0, 100]}
                elif item.label == 'Weed->Other':
                    item.color = {'RGB': [115, 223, 255, 100]}
                elif item.label == 'Weed->Native':
                    item.color = {'RGB': [0, 92, 230, 100]}
                elif item.label == 'No Change':
                    item.color = {'RGB': [255, 255, 255, 0]}

        # update symbology
        lyr.symbology = sym

    except Exception as e:
        raise e

    # disable add to map
    arcpy.env.addOutputsToMap = False

    return


def apply_frac_change_symbology(
        in_ras: str
) -> None:
    """
    Takes any raster file path and visualises it in arcgis pro if
    active map available. Applies change class symbology for
    natives, weeds and others. Use for fractional change only.

    :param in_ras: Raster file path.
    :return: None.
    """

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
            raise ValueError('Could not find UAV change layer on map.')

        # get symbology from layer and force symbology type
        sym = lyr.symbology
        sym.updateColorizer('RasterUniqueValueColorizer')
        sym.colorizer.classificationField = 'Class_name'

        # iterate classes and update color mapping
        for group in sym.colorizer.groups:
            for item in group.items:
                if item.label == 'Native':
                    item.color = {'RGB': [0, 92, 230, 100]}
                elif item.label == 'Weed':
                    item.color = {'RGB': [255, 0, 0, 100]}
                elif item.label == 'Other':
                    item.color = {'RGB': [255, 170, 0, 100]}
                elif item.label == 'Native & Other':
                    item.color = {'RGB': [255, 255, 0, 100]}
                elif item.label == 'Weed & Other':
                    item.color = {'RGB': [0, 230, 169, 100]}
                elif item.label == 'Native & Weed':
                    item.color = {'RGB': [255, 115, 223, 100]}
                elif item.label == 'Native & Weed & Other':
                    item.color = {'RGB': [130, 130, 130, 100]}
                elif item.label == 'No Change':
                    item.color = {'RGB': [255, 255, 255, 0]}

        # update symbology
        lyr.symbology = sym

    except Exception as e:
        raise e

    # disable add to map
    arcpy.env.addOutputsToMap = False

    return


def concat_netcdf_files(
        nc_files: list
) -> xr.Dataset:
    """
    Concatenates multiple NetCDF files into one single
    xr Dataset.

    :param nc_files: List of NetCDFs file paths.
    :return: Xarray Dataset of concatenated NetCDFs.
    """

    try:
        # init new data array list
        ds_list = []

        # iter each nc, load and append
        for nc in nc_files:
            try:
                with xr.open_dataset(nc) as ds:
                    ds.load()
                ds_list.append(ds)
            except:
                print(f'Could not open nc: {nc}.')
                pass

        # combine all netcdfs into one and sort by date
        ds = xr.concat(ds_list, 'time').sortby('time')

    except Exception as e:
        raise e

    return ds


def merge_netcdf_files(
        nc_files: list,
) -> xr.Dataset:
    """
    Merges multiple NetCDF files into one single
    xr Dataset.

    :param nc_files: List of NetCDFs file paths.
    :return: Xarray Dataset of merged NetCDFs.
    """

    try:
        # init new data array list
        ds_list = []

        # iter each nc, load and append
        for nc in nc_files:
            try:
                with xr.open_dataset(nc) as ds:
                    ds.load()
                ds_list.append(ds)
            except:
                print(f'Could not open nc: {nc}.')
                pass

        # merge vars
        ds = xr.merge(ds_list)

    except Exception as e:
        raise e

    return ds


def raster_to_xr(
        in_ras: str,
        out_nc: str,
        epsg: int,
        datetime: str,
        var_names: list,
        dtype: str
) -> xr.Dataset:
    """
    Takes a path to a single or multiband raster and exports
    it to a NetCDF file. The dataset is read in as xr Dataset
    and returned. The attributes and crs information is corrected.
    Variable names can also be provided in order of raster bands
    for renaming.

    :param in_ras: Path to a raster.
    :param out_nc: Path to an output NetCDF.
    :param epsg: EPSG code of current raster.
    :param datetime: Datetime value of current raster.
    :param var_names: List of band/var names.
    :param dtype: Datatype of current raster (int16, 32 or float32, 64).
    :return:
    """

    try:
        # open raster via path and extract geotrans
        dataset = gdal.Open(in_ras, gdal.GA_ReadOnly)
        geotrans = dataset.GetGeoTransform()

        # extract useful info
        x_res, y_res = geotrans[1], geotrans[5]
        nodata_value = dataset.GetRasterBand(1).GetNoDataValue()

        # get dtype
        if dtype == 'int16':
            dtype = gdal.GDT_Int16
        elif dtype == 'int32':
            dtype = gdal.GDT_Int32
        elif dtype == 'float32':
            dtype = gdal.GDT_Float32
        elif dtype == 'float64':
            dtype = gdal.GDT_Float64
        else:
            dtype = gdal.GDT_Float32

        # create output options (need output type and epsg)
        opts = gdal.TranslateOptions(xRes=x_res,
                                     yRes=y_res,
                                     noData=nodata_value,
                                     outputSRS=f'EPSG:{epsg}',
                                     outputType=dtype)

        # translate from gtiff to netcdf
        gdal.Translate(out_nc, dataset, options=opts)
        dataset = None

        # read it in as netcdf
        with xr.open_dataset(out_nc) as ds:
            ds.load()

        # fix netcdf attributes and band names
        ds = fix_xr_attrs(ds=ds,
                          epsg=epsg,
                          var_names=var_names,
                          datetime=datetime)

        # overwrite netcdf with clean file
        ds.to_netcdf(out_nc)
        ds.close()

        # read it in as netcdf again
        with xr.open_dataset(out_nc) as ds:
            ds.load()

    except Exception as e:
        raise e

    return ds


def multi_band_xr_to_raster(
        da: xr.Dataset,
        out_folder: str
) -> arcpy.Raster:
    """
    Takes a 2D xarray Dataset object and returns a single GeoTiff
    where each band represents a variable in the input NetCDF. Working
    is done in termporary scratch folder

    :param da: xarray Dataset.
    :param out_folder: Path to output folder.
    :return: arcpy.Raster composite.
    """

    # init out band list
    out_band_tifs = []

    try:
        # convert each dataset var to a real tif band
        for var in list(da.data_vars):
            # set up output nc and tif files in scratch
            out_nc = os.path.join(out_folder, f'{var}.nc')
            out_tif = os.path.join(out_folder, f'{var}.tif')

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


def netcdf_to_crf(
        in_nc: str,
        out_crf: str
) -> str:
    """
    Convert any NetCDF (2D, 3D) to a Cloud Raster
    Format (CRF) file.

    :param in_nc: Path to NetCDF.
    :param out_crf: Path to output CRF.
    :return: Returns output CRF path.
    """

    try:
        # convert netcdf to crf file
        arcpy.management.CopyRaster(in_raster=in_nc,
                                    out_rasterdataset=out_crf)

    except Exception as e:
        raise e

    return out_crf


def fix_xr_attrs(
        ds: xr.Dataset,
        epsg: int,
        datetime: str,
        var_names: list
) -> xr.Dataset:
    """
    Fixes xr Dataset spatial information,
    band names and attributes. Makes it
    compatible with ArcGIS.

    :param ds: Xarray Dataset.
    :return: Xarray Dataset.
    """

    # iter bands and store wkt, delete spatial band
    crs_name, crs_wkt = None, None
    for band in ds:
        if len(ds[band].shape) == 0:
            crs_name = band
            crs_wkt = str(ds[band].attrs.get('spatial_ref'))
            ds = ds.drop_vars(crs_name)
            break

    # check we got something
    if crs_name is None or crs_wkt is None:
        raise ValueError('Could not find expected CRS band.')

    # assign new spatial ref coord for epsg
    ds = ds.assign_coords({'spatial_ref': epsg})
    ds['spatial_ref'].attrs = {
        'spatial_ref': crs_wkt,
        'grid_mapping_name': crs_name
    }

    # add datetime dimension/coord if provided
    if datetime is not None and 'time' not in ds:
        dt = pd.to_datetime(datetime, format='%Y-%m-%d')
        ds = ds.assign_coords({'time': dt.to_numpy()})
        ds = ds.expand_dims('time')

    # add resoltuon and crs info on to x, y dims
    for dim in ds.dims:
        if dim in ['x', 'y', 'lat', 'lon']:
            ds[dim].attrs = {
                'resolution': np.mean(np.diff(ds[dim])),
                'crs': f'EPSG:{epsg}'
            }

    # iter bands and add crs info on
    for i, band in enumerate(ds):
        ds[band].attrs = {
            'units': '1',
            'crs': f'EPSG:{epsg}',
            'grid_mapping': 'spatial_ref',
        }

        # rename band if applicable
        if var_names is not None:
            if len(var_names) == len(ds):
                ds = ds.rename({band: var_names[i]})

    # overwrite attribute info
    ds.attrs = {
        'crs': f'EPSG:{epsg}',
        'grid_mapping': 'spatial_ref'
    }

    return ds


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


def delete_visual_rasters(
        rasters: list
) -> None:
    """

    :param rasters:
    :return:
    """

    # force input to be list
    if not isinstance(rasters, list):
        rasters = [rasters]

    # iterate rasters and delete, ignore errors
    for raster in rasters:
        try:
            arcpy.management.Delete(raster)
        except:
            pass

    return
