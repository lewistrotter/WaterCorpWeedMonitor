
import os
import datetime
import requests
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp
import arcpy

from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from scripts import shared

# config global gdal environment
gdal.SetConfigOption('GDAL_HTTP_UNSAFESSL', 'YES')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'tif')
gdal.SetConfigOption('GDAL_HTTP_MULTIRANGE', 'YES')
gdal.SetConfigOption('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES')
gdal.SetConfigOption('GDAL_HTTP_CONNECTTIMEOUT', '30')


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
        self.__band_dataset = None

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.__dict__)

    def convert_datetime_to_date(self):
        """
        Convert datetime string in date field to just date.
        :return: None.
        """

        if isinstance(self.date, str):
            date = pd.to_datetime(self.date)
            self.date = date.strftime('%Y-%m-%d')

    def build_oa_mask_wcs_url(self):
        """
        Creates a WCS URL for s2 cloudless mask band.

        :return: URL string.
        """

        url = build_wcs_url(self.collection,
                            'oa_s2cloudless_mask',
                            self.date,
                            self.out_bbox,
                            self.out_epsg,
                            self.out_res)

        return url

    def build_band_wcs_url(self):
        """
        Creates a WCS URL for band(s).

        :return: URL string.
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
        Creates an output path for downloaded file.

        :return: File path string.
        """

        out_file = os.path.join(self.out_path, 'R' + self.date + self.out_extension)

        return out_file

    def set_oa_mask_dataset_via_wcs(self):
        """
        Set s2 cloudless mask gdal dataset via WCS URL.

        :return: None.
        """

        try:
            url = self.build_oa_mask_wcs_url()
            self.__oa_mask_dataset = gdal.Open(url, gdal.GA_ReadOnly)

        except Exception as e:
            raise e

    def set_band_dataset_via_wcs(self):
        """
        Set band gdal dataset via WCS URL.

        :return: None.
        """

        try:
            url = self.build_band_wcs_url()
            self.__band_dataset = gdal.Open(url, gdal.GA_Update)

        except Exception as e:
            raise e

    def get_percent_out_of_bounds_mask_pixels(self):
        """
        Calculate the percent (0 - 100) of invalid pixels
        that are out of bounds of scene (i.e., not cloud pixels).

        :return: Percent out of bounds or None.
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
        Calculate the percent (0 - 100) of invalid pixels
        that are clouds within the scene.

        :param quality_flags: Pixel values containing invalid classes.
        :return: Percent out of bounds or None.
        """

        if self.__oa_mask_dataset is not None:
            mask_arr = self.__oa_mask_dataset.ReadAsArray()

            invalid_size = np.sum(~np.isin(mask_arr, quality_flags + [0]))  # including 0 as valid
            total_size = mask_arr.size
            percent_invalid = invalid_size / total_size * 100

            return percent_invalid

        return

    def is_mask_valid(self, quality_flags, max_out_of_bounds, max_invalid_pixels):
        """
        Determine if mask contains too many invalid
        pixels or not.

        :param quality_flags: Set pixel mask classes.
        :param max_out_of_bounds:  Percentage of pixels out of bounds.
        :param max_invalid_pixels: Percentage of invalid pixels.
        :return: True or False boolean.
        """

        if self.__oa_mask_dataset is None:
            return False

        pct_out_of_bounds = self.get_percent_out_of_bounds_mask_pixels()
        if pct_out_of_bounds is not None and pct_out_of_bounds > max_out_of_bounds:
            return False

        pct_invalid = self.get_percent_invalid_mask_pixels(quality_flags)
        if pct_invalid is not None and pct_invalid > max_invalid_pixels:
            return False

        return True

    def set_band_dataset_nodata_via_mask(self, quality_flags, no_data_value):
        """
        Set band dataset invalid pixels to nodata value.

        :param quality_flags: Invalid pixel mask classes.
        :param no_data_value: Value to convert invalid pixels to.
        :return: None.
        """

        mask_arr = self.__oa_mask_dataset.ReadAsArray()
        mask_arr = np.isin(mask_arr, quality_flags)  # not including 0 as valid

        band_arr = self.__band_dataset.ReadAsArray()
        band_arr = np.where(band_arr == -999, no_data_value, band_arr)  # set DEA nodata (-999) to user no data
        band_arr = np.where(mask_arr, band_arr, no_data_value)

        self.__band_dataset.WriteArray(band_arr)

    def export_band_dataset_to_netcdf_file(self, nodata_value):
        """
        Export downloaded gdal dataset to NetCDF file.

        :param nodata_value: Set no data value for output.
        :return: None.
        """

        options = {'noData': nodata_value, 'xRes': 10.0, 'yRes': 10.0}

        out_filepath = self.build_output_filepath()
        gdal.Translate(out_filepath,
                       self.__band_dataset,
                       **options)

        self.fix_netcdf_metadata()

    def fix_netcdf_metadata(self):
        """
        Fix NetCDF attributes.

        :return: None.
        """

        filepath = self.build_output_filepath()

        ds = xr.open_dataset(filepath)
        ds = ds.load()
        ds.close()

        ds = shared.fix_xr_attrs(ds=ds,
                                 datetime=self.date,
                                 epsg=self.out_epsg,
                                 var_names=self.assets)

        ds.to_netcdf(filepath)
        ds.close()

    def close_datasets(self):
        """
        Close private gdal datasets.

        :return: None
        """

        self.__oa_mask_dataset = None
        self.__band_dataset = None


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


def query_stac_endpoint(
        stac_url
):
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


def group_downloads_by_solar_day(
        downloads
):
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


def remove_existing_downloads(
        downloads,
        existing_dates
):
    """
    Removes any downloads based on datetime if provided
    in the existing dates list input.

    :param downloads: List of download objects.
    :param existing_dates: List of strings in format YYYY-MM-DD.
    :return: List of clean downloads.
    """

    clean_downloads = []
    for download in sorted(downloads, key=lambda d: d.date):
        if download.date not in existing_dates:
            clean_downloads.append(download)

    num_removed = len(downloads) - len(clean_downloads)
    arcpy.AddMessage(f'Removed {num_removed} downloads; already downloaded.')

    return clean_downloads


def remove_downloads_for_current_month(
        downloads
):
    """
    Remove any downloads falling into current year and month.

    :param downloads: List of download objects.
    :return: List of clean downloads.
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


def quick_fetch(
        start_date: str,
        end_date: str,
        stac_bbox: tuple,
        out_bbox: tuple,
        out_folder: str,
) -> list:
    """

    :param start_date:
    :param end_date:
    :param stac_bbox:
    :param out_bbox:
    :param out_folder:
    :return:
    """

    # set dea sentinel 2 collection 3 names
    collections = ['ga_s2am_ard_3', 'ga_s2bm_ard_3']

    # check stac bbox is valid
    if len(stac_bbox) != 4:
        raise ValueError('STAC bounding box needs 2 coordinate pairs.')

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

    # check if output bbox is valid
    if len(out_bbox) != 4:
        raise ValueError('Output bounding box needs 2 coordinate pairs.')

    try:
        # prepare downloads from raw stac features
        downloads = convert_stac_features_to_downloads(features=stac_features,
                                                       assets=assets,
                                                       out_bbox=out_bbox,
                                                       out_epsg=3577,
                                                       out_res=10.0,
                                                       out_path=out_folder,
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
    for file in os.listdir(out_folder):
        if file.endswith('.nc'):
            file_date = file.replace('R', '').replace('.nc', '')
            exist_dates.append(file_date)

    # remove downloads that already exist in sat folder
    if len(exist_dates) > 0:
        downloads = remove_existing_downloads(downloads, exist_dates)

        # if nothing left, leave
        if len(downloads) == 0:
            arcpy.AddWarning('No new satellite downloads were found.')
            return []

    return downloads


def quick_download(
        downloads: list,
        quality_flags: list,
        max_out_of_bounds: int = 1,
        max_invalid_pixels: int = 1,
        nodata_value: int = -999
) -> list:
    """
    Takes a list of valid downloads captured from DEA STAC and
    downloads them on seperate threads. Includes a progressor
    if called from ArcGIS Pro UI.

    :param downloads: List of Download class objects.
    :param quality_flags: List of OA Mask classes.
    :param max_out_of_bounds: Max percentage out of bounds allowed.
    :param max_invalid_pixels: Max percentage out of invalid pixels allowed.
    :param nodata_value: Output NoData value.
    :return: List of download results.
    """

    # set progressor
    arcpy.SetProgressor('step', 'Downloading Sentinel 2 data...', 0, len(downloads), 1)

    # set relevant download parameters
    num_cpu = int(np.ceil(mp.cpu_count() / 2))

    try:
        i = 0
        results = []
        with ThreadPoolExecutor(max_workers=num_cpu) as pool:
            futures = []
            for download in downloads:
                task = pool.submit(validate_and_download,
                                   download,
                                   quality_flags,
                                   max_out_of_bounds,
                                   max_invalid_pixels,
                                   nodata_value)
                futures.append(task)

            for future in as_completed(futures):
                arcpy.AddMessage(future.result())
                results.append(future.result())

                i += 1
                if i % 1 == 0:
                    arcpy.SetProgressorPosition(i)

    except Exception as e:
        raise e

    return results


def delete_error_downloads(
    results: list,
    nc_folder: str
) -> None:

    # get list of all netcdf download errors
    errors = [d for d in results if 'error' in d]

    # when error(s)...
    if len(errors) > 0:
        # prepare error netcdf file paths
        errors = [d.split(' ')[1] for d in errors]
        errors = [d.replace(':', '') for d in errors]
        errors = [f'R{d}.nc' for d in errors]

        # iter each error nc and delete it
        for error in errors:
            try:
                os.remove(os.path.join(nc_folder, error))
            except:
                pass

    return
