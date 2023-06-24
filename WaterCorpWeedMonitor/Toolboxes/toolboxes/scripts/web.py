
import os
import datetime
import shutil
import requests
import numpy as np
import pandas as pd
import xarray as xr
import arcpy

from osgeo import gdal

from scripts import shared

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# region GDAL SET UP

gdal.SetConfigOption('GDAL_HTTP_UNSAFESSL', 'YES')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'tif')
gdal.SetConfigOption('GDAL_HTTP_MULTIRANGE', 'YES')
gdal.SetConfigOption('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES')

# endregion


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# region CLASSES

class Download:
    def __init__(
            self,
            date,
            collection,
            assets,
            coordinates,
            out_bbox,
            out_epsg,
            out_res,
            out_path,
            out_extension
    ):
        """
        A custom object representing a single DEA STAC feature. A Download contains
        variables required to create DEA STAC WCS urls and can download data
        using GDAL.

        :param date: String representing a feature date.
        :param collection: String representing feature's DEA collection name.
        :param assets: List of strings of DEA asset names.
        :param coordinates: List of floats representing feature geometry.
        :param out_bbox: List of floats representing output bbox extent.
        :param out_epsg: Integer representing output bbox EPSG code.
        :param out_res: Float representing output pixel resolution.
        :param out_path: String representing export location.
        :param out_extension: String representing output filetype extension.
        """

        self.date = date
        self.collection = collection
        self.assets = assets
        self.coordinates = coordinates
        self.out_bbox = out_bbox
        self.out_epsg = out_epsg
        self.out_res = out_res
        self.out_path = out_path
        self.out_extension = out_extension
        self.__oa_mask_dataset = None
        self.__oa_prob_dataset = None
        self.__band_dataset = None

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.__dict__)

    def convert_datetime_to_date(self):
        """
        Convert datetime string in date field to just date.
        :return: Nothing.
        """

        if isinstance(self.date, str):
            date = pd.to_datetime(self.date)
            self.date = date.strftime('%Y-%m-%d')

    def build_oa_mask_wcs_url(self):
        """

        :return:
        """

        url = build_wcs_url(self.collection,
                            'oa_s2cloudless_mask',
                            self.date,
                            self.out_bbox,
                            self.out_epsg,
                            self.out_res)

        return url

    def build_oa_prob_wcs_url(self):
        """

        :return:
        """

        url = build_wcs_url(self.collection,
                            'oa_s2cloudless_prob',
                            self.date,
                            self.out_bbox,
                            self.out_epsg,
                            self.out_res)

        return url

    def build_band_wcs_url(self):
        """

        :return:
        """

        url = build_wcs_url(self.collection,
                            self.assets,
                            self.date,
                            self.out_bbox,
                            self.out_epsg,
                            self.out_res)

        return url

    def build_output_filepath(self):
        """

        :return:
        """

        out_file = os.path.join(self.out_path, 'R' + self.date + self.out_extension)

        return out_file

    def set_oa_mask_dataset_via_wcs(self):
        """

        :return:
        """

        try:
            url = self.build_oa_mask_wcs_url()
            self.__oa_mask_dataset = gdal.Open(url, gdal.GA_ReadOnly)
        except Exception as e:
            raise e

    def set_oa_prob_dataset_via_wcs(self):
        """

        :return:
        """

        try:
            url = self.build_oa_prob_wcs_url()
            self.__oa_prob_dataset = gdal.Open(url, gdal.GA_ReadOnly)
        except Exception as e:
            raise e

    def set_band_dataset_via_wcs(self):
        """

        :return:
        """

        try:
            url = self.build_band_wcs_url()
            self.__band_dataset = gdal.Open(url, gdal.GA_Update)
        except Exception as e:
            raise e

    def get_percent_out_of_bounds_mask_pixels(self):
        """

        :return:
        """

        if self.__oa_mask_dataset is not None:
            mask_arr = self.__oa_mask_dataset.ReadAsArray()

            invalid_size = np.sum(mask_arr == 0)  # assuming 0 is always out of bounds...
            total_size = mask_arr.size
            percent_out_of_bounds = invalid_size / total_size * 100

            return percent_out_of_bounds

        return

    def get_percent_invalid_mask_pixels(self, quality_flags):
        """

        :param quality_flags:
        :return:
        """

        if self.__oa_mask_dataset is not None:
            mask_arr = self.__oa_mask_dataset.ReadAsArray()

            invalid_size = np.sum(~np.isin(mask_arr, quality_flags + [0]))  # including 0 as valid
            total_size = mask_arr.size
            percent_invalid = invalid_size / total_size * 100

            return percent_invalid

        return

    def set_band_dataset_nodata_via_mask(self, quality_flags, no_data_value):
        """

        :param quality_flags:
        :param no_data_value:
        :return:
        """

        mask_arr = self.__oa_mask_dataset.ReadAsArray()
        mask_arr = np.isin(mask_arr, quality_flags)  # not including 0 as valid

        # prob_arr = self.__oa_prob_dataset.ReadAsArray()
        # & (prob_arr > 0.9)

        band_arr = self.__band_dataset.ReadAsArray()
        band_arr = np.where(band_arr == -999, no_data_value, band_arr)  # set DEA nodata (-999) to user no data
        band_arr = np.where(mask_arr, band_arr, no_data_value)

        self.__band_dataset.WriteArray(band_arr)

    def export_band_dataset_to_netcdf_file(self, nodata_value):
        """

        :param nodata_value:
        :return:
        """

        options = {'noData': nodata_value}

        out_filepath = self.build_output_filepath()
        gdal.Translate(out_filepath,
                       self.__band_dataset,
                       **options)

        self.fix_netcdf_metadata()

    def fix_netcdf_metadata(self):
        """

        :return:
        """

        filepath = self.build_output_filepath()
        ds = xr.open_dataset(filepath)  # using 'with' has caused issues before, avoiding

        # FIXME: newer versions of gdal may break this
        for band in ds:
            if len(ds[band].shape) == 0:
                crs_name = band
                crs_wkt = str(ds[band].attrs.get('spatial_ref'))
                ds = ds.drop_vars(crs_name)
                break

        ds = ds.assign_coords({'spatial_ref': self.out_epsg})
        ds['spatial_ref'].attrs = {
            'spatial_ref': crs_wkt,
            'grid_mapping_name': crs_name
        }

        if 'time' not in ds:
            dt = pd.to_datetime(self.date, format='%Y-%m-%d')
            ds = ds.assign_coords({'time': dt.to_numpy()})
            ds = ds.expand_dims('time')

        for dim in ds.dims:
            if dim in ['x', 'y', 'lat', 'lon']:
                ds[dim].attrs = {
                    #'units': 'metre'  # TODO: how to get units?
                    'resolution': np.mean(np.diff(ds[dim])),
                    'crs': f'EPSG:{self.out_epsg}'
                }

        for i, band in enumerate(ds):
            ds[band].attrs = {
                'units': '1',
                #'nodata': self.nodata,  TODO: implemented out_nodata
                'crs': f'EPSG:{self.out_epsg}',
                'grid_mapping': 'spatial_ref',
            }

            ds = ds.rename({band: self.assets[i]})

        # TODO: we wipe gdal, history, conventions, other metadata
        ds.attrs = {
            'crs': f'EPSG:{self.out_epsg}',
            'grid_mapping': 'spatial_ref'
        }

        ds.to_netcdf(filepath)
        ds.close()

    def close_datasets(self):
        """

        :return:
        """

        self.__oa_mask_dataset = None
        self.__oa_prob_dataset = None
        self.__oa_band_dataset = None

    def is_mask_valid(self, quality_flags, max_out_of_bounds, max_invalid_pixels):
        """

        :param quality_flags:
        :param max_out_of_bounds:
        :param max_invalid_pixels:
        :return:
        """

        if self.__oa_mask_dataset is None:
            return False
        #elif self.__oa_prob_dataset is None:  # disabled for now
            #return False

        pct_out_of_bounds = self.get_percent_out_of_bounds_mask_pixels()
        if pct_out_of_bounds is not None and pct_out_of_bounds > max_out_of_bounds:
            return False

        pct_invalid = self.get_percent_invalid_mask_pixels(quality_flags)
        if pct_invalid is not None and pct_invalid > max_invalid_pixels:
            return False

        return True

# endregion


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# region CONSTRUCTORS

def build_stac_query_url(
        collection,
        start_date,
        end_date,
        bbox,
        limit,
        full=False
):
    """
    Takes key query parameters (collection, date range, bbox, limit) required
    by STAC endpoint to perform a query.
    :param collection: String representing a DEA collection name.
    :param start_date: String representing query start date (YYYY-MM-DD).
    :param end_date: String representing query end date (YYYY-MM-DD).
    :param bbox: Tuple of coordinates representing query bbox.
    :param limit: Integer representing max features to return per query.
    :param full: Boolean indicating whether to do a full, deep search of STAC.
    :return: String representing STAC query url.
    """

    url = 'https://explorer.sandbox.dea.ga.gov.au/stac/search?'
    url += f'&collection={collection}'
    url += f'&time={start_date}/{end_date}'
    url += f'&bbox={",".join(map(str, bbox))}'
    url += f'&limit={limit}'
    url += f'&_full={full}'

    return url


def build_wcs_url(
        collection,
        assets,
        date,
        bbox,
        epsg,
        res
):
    """
    Takes key query parameters (collection, assets, date, bbox, epsg, res)
    and constructs a WCS url to download data from DEA public database.
    :param collection: String representing a DEA collection name.
    :param assets: List of strings representing DEA asset names.
    :param date: String representing query date (YYYY-MM-DD).
    :param bbox: Tuple of coordinates representing bbox of output data.
    :param epsg: Integer representing EPSG code of output data.
    :param res: Float representing pixel resolution of output data.
    :return: String representing WCS url.
    """

    if isinstance(assets, (list, tuple)):
        assets = ','.join(map(str, assets))

    if isinstance(bbox, (list, tuple)):
        bbox = ','.join(map(str, bbox))

    url = 'https://ows.dea.ga.gov.au/wcs?service=WCS'
    url += '&VERSION=1.0.0'
    url += '&REQUEST=GetCoverage'
    url += '&COVERAGE={}'.format(collection)
    url += '&MEASUREMENTS={}'.format(assets)
    url += '&TIME={}'.format(date)
    url += '&BBOX={}'.format(bbox)
    url += '&CRS=EPSG:{}'.format(epsg)
    url += '&RESX={}'.format(res)
    url += '&RESY={}'.format(res)
    url += '&FORMAT=GeoTIFF'

    return url


def query_stac_endpoint(stac_url):
    """
    Takes a single DEA STAC endpoint query url and returns all available features
    found for the search parameters.

    :param stac_url: String containing a valid DEA STAC endpoint query url.
    :return: List of dictionaries representing returned STAC metadata.
    """

    features = []
    while stac_url:
        try:
            with requests.get(stac_url) as response:
                response.raise_for_status()
                result = response.json()

            if len(result) > 0:
                features += result.get('features')

            stac_url = None
            for link in result.get('links'):
                if link.get('rel') == 'next':
                    stac_url = link.get('href')
                    break

        # TODO: improve specific exception type. implement retry.
        except Exception as e:
            raise ValueError(e)

    return features


def fetch_all_stac_features(
        collections,
        start_date,
        end_date,
        bbox,
        limit
):
    """
    Iterates through provided DEA collections and queries DEA STAC endpoint
    for features existing for each. Once all collections are obtained, all
    results are merged and a list of STAC metadata dictionaries are merged.

    :param collections: List of strings representing DEA STAC collection names.
    :param start_date: String representing query start date (YYYY-MM-DD).
    :param end_date: String representing query end date (YYYY-MM-DD).
    :param bbox: Tuple of coordinates representing query bbox.
    :param limit: Integer representing max features to return per query.
    :return: List of dictionaries representing all returned STAC metadata merged.
    """

    all_features = []
    for collection in collections:
        arcpy.AddMessage(f'Querying STAC endpoint for {collection} data.')

        stac_url = build_stac_query_url(collection,
                                        start_date,
                                        end_date,
                                        bbox,
                                        limit)

        new_features = query_stac_endpoint(stac_url)

        if len(new_features) > 0:
            all_features += new_features

    return all_features


def convert_stac_features_to_downloads(
        features,
        assets,
        out_bbox,
        out_epsg,
        out_res,
        out_path,
        out_extension
):
    """
    Iterates through raw DEA STAC query results and converts them into
    more sophisticated Download objects.

    :param features: List of raw DEA STAC result dictionaries.
    :param assets: List of strings represented requested assets.
    :param out_bbox: Tuple of floats representing output bbox.
    :param out_epsg: Integer representing output EPSG code.
    :param out_res: Float representing output pixel resolution.
    :param out_path: String representing output path for data export.
    :param out_extension: String representing output file extension.
    :return: List of Download objects.
    """

    # TODO: clean up and error handling needed

    downloads = []
    for feature in features:
        collection = feature.get('collection')

        if 'properties' in feature:
            date = feature.get('properties').get('datetime')

            if 'geometry' in feature:
                coordinates = feature.get('geometry').get('coordinates')[0]

                download = Download(
                    date=date,
                    collection=collection,
                    assets=assets,
                    coordinates=coordinates,
                    out_bbox=out_bbox,
                    out_epsg=out_epsg,
                    out_res=out_res,
                    out_path=out_path,
                    out_extension=out_extension
                )

                downloads.append(download)

    arcpy.AddMessage(f'Found a total of {len(downloads)} STAC features.')

    return downloads


def group_downloads_by_solar_day(downloads):
    """
    Takes a list of download objects and groups them into solar day,
    ensuring each DEA STAC download includes contiguous scene pixels
    from a single satellite pass. Download datestimes are converted to
    date, sorted by date, grouped by date and the first date in each
    group is selected.

    :param downloads: List of Download objects.
    :return: List of Download objects grouped by solar day.
    """

    for download in sorted(downloads, key=lambda d: d.date):
        download.convert_datetime_to_date()

    clean_downloads = []
    processed_dates = []
    for download in downloads:
        if download.date not in processed_dates:
            clean_downloads.append(download)
            processed_dates.append(download.date)

    num_removed = len(downloads) - len(clean_downloads)
    arcpy.AddMessage(f'Grouped {num_removed} downloads by solar day.')

    return clean_downloads


def remove_existing_downloads(downloads, existing_dates):
    """

    :param downloads:
    :param existing_dates:
    :return:
    """

    clean_downloads = []
    for download in sorted(downloads, key=lambda d: d.date):
        if download.date not in existing_dates:
            clean_downloads.append(download)

    num_removed = len(downloads) - len(clean_downloads)
    arcpy.AddMessage(f'Removed {num_removed} downloads; already downloaded.')

    return clean_downloads


def remove_downloads_for_current_month(downloads):
    """

    :param downloads:
    :return:
    """

    now_year = datetime.datetime.now().year
    now_month = datetime.datetime.now().month
    now_date = f'{now_year}-{str(now_month).zfill(2)}-01'

    clean_downloads = []
    for download in sorted(downloads, key=lambda d: d.date):
        if download.date < now_date:
            clean_downloads.append(download)

    num_removed = len(downloads) - len(clean_downloads)
    arcpy.AddMessage(f'Removed {num_removed} downloads; month not complete yet.')

    return clean_downloads


def validate_and_download(
        download,
        quality_flags,
        max_out_of_bounds,
        max_invalid_pixels,
        nodata_value
):
    """
    Takes a single download object, checks if download is valid based on
    number of invalid pixels in mask band, and if valid, downloads the raw
    band data to a specified location and file format captured within the
    download.

    :param download: Download object.
    :param quality_flags: List of integers representing valid mask values.
    :param max_out_of_bounds: Float representing max percentage of out of bounds pixels.
    :param max_invalid_pixels: Float representing max percentage of invalid pixels.
    :param nodata_value: Float representing the value used to represent no data pixels.
    :return: String message indicating success or failure of download.
    """

    try:
        download.set_oa_mask_dataset_via_wcs()
        #download.set_oa_prob_dataset_via_wcs()  # disabled for now
        is_valid = download.is_mask_valid(quality_flags,
                                          max_out_of_bounds,
                                          max_invalid_pixels)

        if is_valid is True:
            download.set_band_dataset_via_wcs()

            download.set_band_dataset_nodata_via_mask(quality_flags,
                                                      nodata_value)

            download.export_band_dataset_to_netcdf_file(nodata_value)
            message = f'Download {download.date}: successful.'
        else:
            message = f'Download {download.date}: too many invalid pixels.'
    except:
        message = f'Download {download.date}: error.'

    download.close_datasets()

    return message


def combine_netcdf_files(data_folder, out_nc):
    """

    :param data_folder:
    :return:
    """

    files = []
    for file in os.listdir(data_folder):
        if file.endswith('.nc'):
            files.append(os.path.join(data_folder, file))

    if len(files) < 2:
        return

    ds_list = []
    for file in files:
        ds = xr.open_dataset(file)
        ds_list.append(ds)

    ds = xr.concat(ds_list, dim='time').sortby('time')
    ds.to_netcdf(out_nc)
    ds.close()

    for ds in ds_list:
        ds.close()

    try:
        shutil.rmtree(data_folder)
    except:
        arcpy.AddMessage('Could not delete NetCDFs data folder.')

    return


def downloads_to_folder(data_folder, out_folder):
    """

    :param data_folder:
    :param out_folder:
    :return:
    """

    for file in os.listdir(data_folder):
        in_file = os.path.join(data_folder, file)
        out_file = os.path.join(out_folder, file)
        shutil.move(in_file, out_file)

    try:
        shutil.rmtree(data_folder)
    except:
        arcpy.AddMessage('Could not delete NetCDFs data folder.')


def get_s2_wc_downloads(
        grid_tif: str,
        out_folder: str,

) -> list:

    # set sentinel 2 data date range
    start_date, end_date = '2017-01-01', '2039-12-31'

    # set dea sentinel 2 collection 3 names
    collections = [
        'ga_s2am_ard_3',
        'ga_s2bm_ard_3'
    ]

    # reproject grid to wgs84 bounding box
    tmp_grd = arcpy.Raster(grid_tif)
    tmp_prj = arcpy.ia.Reproject(tmp_grd, {'wkid': 4326})

    # get bounding box in wgs84 for stac query
    stac_bbox = shared.get_bbox_from_raster(tmp_prj)
    if len(stac_bbox) != 4:
        raise ValueError('Could not generate STAC bounding box.')

    try:
        # get all stac features from 2017 to now
        stac_features = fetch_all_stac_features(collections=collections,
                                                start_date=start_date,
                                                end_date=end_date,
                                                bbox=stac_bbox,
                                                limit=100)

    except Exception as e:
        arcpy.AddMessage(str(e))
        raise ValueError('Unable to fetch STAC features. See messages.')

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

    # reproject grid to albers now
    tmp_grd = arcpy.Raster(grid_tif)
    tmp_prj = arcpy.ia.Reproject(tmp_grd, {'wkid': 3577})

    # get bounding box in wgs84 for stac query
    out_bbox = shared.get_bbox_from_raster(tmp_prj)
    if len(out_bbox) != 4:
        raise ValueError('Could not generate output bounding box.')

    # add 30 metres on every side to prevent gaps
    out_bbox = shared.expand_box_by_metres(bbox=out_bbox, metres=30)

    # set raw output nc folder (one nc per date)
    raw_ncs_folder = os.path.join(out_folder, 'raw_ncs')
    if not os.path.exists(raw_ncs_folder):
        os.mkdir(raw_ncs_folder)

    try:
        # prepare downloads from raw stac features
        downloads = convert_stac_features_to_downloads(features=stac_features,
                                                       assets=assets,
                                                       out_bbox=out_bbox,
                                                       out_epsg=3577,
                                                       out_res=10,
                                                       out_path=raw_ncs_folder,
                                                       out_extension='.nc')

    except Exception as e:
        arcpy.AddMessage(str(e))
        raise ValueError('Unable to convert STAC features to downloads. See messages.')

    # group downloads captured on same solar day
    downloads = group_downloads_by_solar_day(downloads=downloads)
    if len(downloads) == 0:
        arcpy.AddWarning('No valid downloads were found.')
        return []

    # remove downloads if current month (we want complete months)
    downloads = remove_downloads_for_current_month(downloads)
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
        downloads = remove_existing_downloads(downloads, exist_dates)

        # if nothing left, leave
        if len(downloads) == 0:
            arcpy.AddWarning('No new satellite downloads were found.')
            return []

    return downloads

# endregion
