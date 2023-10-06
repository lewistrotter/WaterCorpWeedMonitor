
import os
import numpy as np
import pandas as pd
import xarray as xr
import arcpy

from scipy.stats import zscore

from scripts import shared


def remove_xr_outliers(
        ds: xr.Dataset,
        max_z_value: float
) -> xr.Dataset:
    """
    Takes a raw multiband xarray dataset and calculates outliers based
    on z-score. If any band pixel has an outlier, then all bands for that
    pixel are set to nan. This is done per date in dataset.

    :param ds: Xarray Dataset with multiple raw bands.
    :param max_z_value: Maximum z-score value allowed.
    :return: Outlier-free Xarray Dataset.
    """

    try:
        # apply zscore per pixel across whole time series (0 is time dim)
        z_mask = xr.apply_ufunc(zscore, ds, 0)

        # flag any outlier occurrences as true if > z-score
        z_mask = np.abs(z_mask) > max_z_value

        # set pixel to nan when any band variable is outlier true
        z_mask = z_mask.to_array().max('variable')

        # set pixels to nan where outlier detected
        ds = ds.where(~z_mask)

    except Exception as e:
        raise e

    return ds


def resample_xr_monthly_medians(
        ds: xr.Dataset
) -> xr.Dataset:
    """
    Takes a xarray Dataset and resamples it to
    monthly medians. It then interpolates missing
    values.

    :param ds: Raw xarray Dataset.
    :return: Monthly median xarray Dataset.
    """

    try:
        # resample to monthly medians
        ds = ds.resample(time='1MS').median('time')

        # interpolate nan
        ds = ds.interpolate_na('time')

    except Exception as e:
        raise e

    return ds


def create_roi_freq_points(
        da_hr: xr.DataArray,
        ds_lr: xr.Dataset,
        out_shp: str
) -> str:
    """

    :param da_hr:
    :param ds_lr:
    :param out_shp:
    :return:
    """

    # check input is high-res is 2d
    if len(da_hr.shape) != 2:
        raise ValueError('High-resolution xr can only be 2D.')

    try:
        # extract raw x, y arrays
        xs = ds_lr['x'].values
        ys = ds_lr['y'].values

        # get mean x, y pixel sizes
        x_res = np.mean(np.diff(xs)) / 2
        y_res = np.mean(np.diff(ys)) / 2

        i = 0
        rows = []
        for x in xs:
            for y in ys:
                # build pixel-sized window
                x_slice = slice(x - x_res, x + x_res)
                y_slice = slice(y - y_res, y + y_res)

                # extract high-res values within window
                arr = da_hr.sel(x=x_slice, y=y_slice).values

                # if values exist...
                if arr.size != 0 and ~np.all(arr == -999):
                    # flatten array
                    arr = arr.flatten()

                    # extract nodata values, proceed if < 25% nans
                    nans = arr[arr == -999]
                    if len(nans) / len(arr) < 0.25:
                        # remove nans
                        arr = arr[arr != -999]

                        # get num classes/counts in win, prepare labels, calc freq
                        classes, counts = np.unique(arr, return_counts=True)
                        classes = [f'c_{c}' for c in classes]
                        freqs = (counts / np.sum(counts)).astype('float16')

                        # init fraction map
                        class_map = {
                            'c_0': 0.0,
                            'c_1': 0.0,
                            'c_2': 0.0
                        }

                        # update existing map with new freqs
                        class_map.update(dict(zip(classes, freqs)))

                        # add rid and x, y to dict
                        class_map['rid'] = i
                        class_map['xy'] = (x, y)

                        # add to list
                        rows.append(class_map)

                        # increment
                        i += 1

    except Exception as e:
        raise e

    # check if anything came back
    if len(rows) == 0:
        raise ValueError('Could not extract any high-res pixels.')

    try:
        # get folder and file name
        out_folder = os.path.dirname(out_shp)
        out_file = os.path.basename(out_shp)

        # create new empty roi shapefile
        out_srs = arcpy.SpatialReference(3577)
        arcpy.management.CreateFeatureclass(out_path=out_folder,
                                            out_name=out_file,
                                            geometry_type='POINT',
                                            spatial_reference=out_srs)

        # add required class and other fields
        arcpy.management.AddFields(in_table=out_shp,
                                   field_description='rid LONG;c_0 FLOAT;c_1 FLOAT;c_2 FLOAT')

        # populate shapefile with field values
        fields = ['rid', 'c_0', 'c_1', 'c_2', 'SHAPE@XY']
        with arcpy.da.InsertCursor(in_table=out_shp, field_names=fields) as cursor:
            for row in rows:
                # unpack values
                rid = row['rid']
                c_0 = row['c_0']
                c_1 = row['c_1']
                c_2 = row['c_2']
                xy = row['xy']

                # insert into shapefile
                cursor.insertRow([rid, c_0, c_1, c_2, xy])

    except Exception as e:
        raise e

    return out_shp


def extract_xr_to_roi_points(
        roi_shp: str,
        in_ds: xr.Dataset
) -> str:

    # check if ds time dimension valid
    if 'time' not in in_ds:
        raise ValueError('Dataset needs time dimension.')

    # create name map
    band_map = {
        'nbart_blue': 'blue',
        'nbart_green': 'green',
        'nbart_red': 'red',
        'nbart_red_edge_1': 'redge_1',
        'nbart_red_edge_2': 'redge_2',
        'nbart_red_edge_3': 'redge_3',
        'nbart_nir_1': 'nir_1',
        'nbart_nir_2': 'nir_2',
        'nbart_swir_2': 'swir_2',
        'nbart_swir_3': 'swir_3'
    }

    try:
        # create xr copy and rename bands
        ds = in_ds.copy(deep=True)
        ds = ds.rename(band_map)

        # remove all roi shapefile fields except required
        keep_fields = ['rid', 'c_0', 'c_1', 'c_2']
        arcpy.management.DeleteField(in_table=roi_shp,
                                     drop_field=keep_fields,
                                     method='KEEP_FIELDS')

        # prepare fields and types
        fields = ';'.join([f'{band} FLOAT' for band in list(band_map.values())])
        arcpy.management.AddFields(in_table=roi_shp,
                                   field_description=fields)

        # iterate each row  # TODO: make this safe
        fields = list(band_map.values()) + ['SHAPE@XY']
        with arcpy.da.UpdateCursor(roi_shp, fields) as cursor:
            for row in cursor:
                # extract x, y of point
                x, y = row[-1]

                # get the closest pixel within 50 cm
                arr = ds.sel(x=x, y=y, method='nearest', tolerance=0.5)
                vals = arr.to_array().data

                # check if nans, fill with 0 if yes
                if np.isnan(vals).any():
                    vals = np.where(np.isnan(vals), 0, vals)

                # move band values to row band fields, update
                row[:-1] = vals
                cursor.updateRow(row)

    except Exception as e:
        raise e

    return roi_shp


def gwr(
        in_rois: str,
        classvalue: str,
        classdesc: str,
        out_prediction_shp: str,
        out_accuracy_csv: str
) -> str:
    """

    :param in_rois:
    :param classvalue:
    :param classdesc:
    :param out_prediction_shp:
    :param out_accuracy_csv:
    :return:
    """

    # create expected list of fields
    data_vars = [
        'blue',
        'green',
        'red',
        'redge_1',
        'redge_2',
        'redge_3',
        'nir_1',
        'nir_2',
        'swir_2',
        'swir_3'
    ]

    # get band field names from shapefile
    fields = [f.name for f in arcpy.ListFields(in_rois)]

    # check if we have all vars in shapefile
    for var in data_vars:
        if var not in fields:
            raise ValueError('Missing expected band.')

    # set success flag
    flag = True

    try:
        # perform geographically weighted regression
        arcpy.stats.GWR(in_features=in_rois,
                        dependent_variable=classvalue,
                        model_type='CONTINUOUS',
                        explanatory_variables=data_vars,
                        output_features=out_prediction_shp,
                        neighborhood_type='NUMBER_OF_NEIGHBORS',
                        neighborhood_selection_method='GOLDEN_SEARCH',
                        local_weighting_scheme='GAUSSIAN')

    except Exception as e:
        arcpy.AddWarning(f'Warning, error during GWR: {str(e)}')
        flag = False

    try:
        # check fallback
        if flag is False:

            # use fallback if needed
            arcpy.stats.GWR(in_features=in_rois,
                            dependent_variable=classvalue,
                            model_type='CONTINUOUS',
                            explanatory_variables=data_vars,
                            output_features=out_prediction_shp,
                            neighborhood_type='NUMBER_OF_NEIGHBORS',
                            neighborhood_selection_method='USER_DEFINED',
                            number_of_neighbors=100)

    except Exception as e:
        raise e

    try:
        # convert table to arr
        fields = [classvalue, 'PREDICTED']
        arr = arcpy.da.FeatureClassToNumPyArray(out_prediction_shp, fields)

        # extract real (x) and predicted (y) values
        x, y = arr[classvalue], arr['PREDICTED']
        r2 = 1 - np.sum((y - x) ** 2) / np.sum((x - np.mean(x)) ** 2)
        r2 = round(r2, 4)

        # delete arr
        arr = None

        # notify user
        arcpy.AddMessage(f'> R-Squared for {classdesc}: {str(r2)}')

        # build dataframe and export as csv
        df = pd.DataFrame([r2], columns=['R2'])
        df.to_csv(out_accuracy_csv)

    except:
        raise ValueError('Could not read accuracy messages.')
        pass

    return out_prediction_shp



def old_regress(
        in_rois: str,
        classvalue: str,
        classdesc: str,
        out_regress_shp: str,
        out_accuracy_csv: str
) -> str:
    """

    :param in_rois:
    :param classvalue:
    :param classdesc:
    :param out_regress_shp:
    :param out_accuracy_csv:
    :return:
    """

    # create expected list of fields
    data_vars = [
        'blue',
        'green',
        'red',
        'redge_1',
        'redge_2',
        'redge_3',
        'nir_1',
        'nir_2',
        'swir_2',
        'swir_3'
    ]

    # get band field names from shapefile
    fields = [f.name for f in arcpy.ListFields(in_rois)]

    # check if we have all vars in shapefile
    for var in data_vars:
        if var not in fields:
            raise ValueError('Missing expected band.')

    # convert csv to dbf for function
    tmp_cmat_dbf = os.path.basename(out_accuracy_csv)
    tmp_cmat_dbf = tmp_cmat_dbf.split('.')[0] + '.dbf'


    try:
        # train and predict regression
        arcpy.stats.Forest(prediction_type='PREDICT_FEATURES',
                           in_features=in_rois,
                           variable_predict=classvalue,
                           explanatory_variables=data_vars,
                           features_to_predict=in_rois,
                           output_features=out_regress_shp,
                           explanatory_variable_matching=data_vars,
                           number_of_trees=250,
                           percentage_for_training=10,
                           output_validation_table=tmp_cmat_dbf,
                           number_validation_runs=5)

    except Exception as e:
        raise e

    try:
        # create full path to dbf and conbvert to output csv
        tmp_cmat_dbf = os.path.join(arcpy.env.workspace, tmp_cmat_dbf)
        arcpy.conversion.ExportTable(in_table=tmp_cmat_dbf,
                                     out_table=out_accuracy_csv)

        # read csv with pandas and get average r-squares
        med_r2 = pd.read_csv(out_accuracy_csv)['R2'].median().round(3)
        arcpy.AddMessage(f'> R-Squared for {classdesc}: {str(med_r2)}')

    except:
        raise ValueError('Could not read accuracy messages.')
        pass

    return out_regress_shp


def force_pred_zero_to_one(
        in_shp: str
) -> str:
    """

    :param in_shp:
    :return:
    """

    try:
        fields = ['PREDICTED']
        with arcpy.da.UpdateCursor(in_shp, fields) as cursor:
            for row in cursor:
                if row[0] < 0.0:
                    row[0] = 0.0
                elif row[0] > 1.0:
                    row[0] = 1.0

                cursor.updateRow(row)

    except Exception as e:
        raise e

    return in_shp


def roi_points_to_raster(
        in_shp: str,
        out_ras: str
) -> str:
    """

    :param in_shp:
    :param out_ras:
    :return:
    """

    try:
        # convert prediction values at points to raster
        arcpy.conversion.PointToRaster(in_features=in_shp,
                                       value_field='PREDICTED',
                                       out_rasterdataset='tmp_gwr_ras.tif',
                                       cellsize=10.0)

        # apply cubic resampling to smooth pixels out
        arcpy.management.Resample(in_raster='tmp_gwr_ras.tif',
                                  out_raster=out_ras,
                                  cell_size=2.5,
                                  resampling_type='BILINEAR')

    except Exception as e:
        raise e

    return out_ras
