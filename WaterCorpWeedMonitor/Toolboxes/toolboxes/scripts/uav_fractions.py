
import os
import numpy as np
import pandas as pd
import xarray as xr
import arcpy

def build_rois_from_raster(
        in_ras: arcpy.Raster,
        out_rois: str
) -> str:
    """
    Takes an ArcPy Raster class object and builds regions of interest polygons
    from its pixels extents. Basically, the rois are the centroid point of each
    pixel with a square buffer of the pixel size in x and y dimensions. The
    output is a path to the shapefile of rois. This also creates three fields
    for classes 0-2.

    :param in_ras: ArcPy Raster.
    :param out_rois: String to region of interest shapefile path.
    :return: String to region of interest shapefile path.
    """

    try:
        # get a centroid point for each grid cell on sentinel raster
        tmp_pnts = r'memory\tmp_points'
        arcpy.conversion.RasterToPoint(in_raster=in_ras,
                                       out_point_features=tmp_pnts,
                                       raster_field='Value')

        # create square polygons around points
        arcpy.analysis.GraphicBuffer(in_features=tmp_pnts,
                                     out_feature_class=out_rois,
                                     buffer_distance_or_field='5 Meters')  # half s2 cell size

        # add required fields to envelope shapefile
        arcpy.management.AddFields(in_table=out_rois,
                                   field_description="c_0 FLOAT;c_1 FLOAT;c_2 FLOAT;inc SHORT")

    except Exception as e:
        raise e

    return out_rois


def calculate_roi_freqs(
        rois: str,
        da_hr: xr.DataArray
) -> str:
    """
    Takes pre-created region of interest polygons and overlaps them with the pixels
    of a high resolution xarray DataArray. The frequency of all DataArray pixel classes
    falling within reach region of interest is added to the regoin of interest attrobute
    table as seperate fields.

    :param rois: Path to region of interest shapefile.
    :param da_hr: A high-resolution classified raster as a xarray DataArray.
    :return: Path to output region of interest shapefile.
    """

    try:
        # extract all high-res class values within each low res pixel
        with arcpy.da.UpdateCursor(rois, ['c_0', 'c_1', 'c_2', 'inc', 'SHAPE@']) as cursor:
            for row in cursor:
                # get x and y window by slices for each polygon
                x_slice = slice(row[-1].extent.XMin, row[-1].extent.XMax)
                y_slice = slice(row[-1].extent.YMax, row[-1].extent.YMin)

                # extract window values from high-res
                arr = da_hr.sel(x=x_slice, y=y_slice).values

                # FIXME: something not working here, 1 pixel coming back?

                # if values exist...
                if arr.size != 0 and ~np.all(arr == -999):
                    # flatten array and remove -999s
                    arr = arr.flatten()
                    arr = arr[arr != -999]

                    # get num classes/counts in win, prepare labels, calc freq
                    classes, counts = np.unique(arr, return_counts=True)
                    classes = [f'c_{c}' for c in classes]
                    freqs = (counts / np.sum(counts)).astype('float16')

                    # init fraction map
                    class_map = {
                        'c_0': 0.0,
                        'c_1': 0.0,
                        'c_2': 0.0,
                        'inc': 1
                    }

                    # project existing classes and freqs onto map and update row
                    class_map.update(dict(zip(classes, freqs)))
                    row[0:4] = list(class_map.values())
                else:
                    # flag row to be excluded
                    row[3] = 0

                # update row
                cursor.updateRow(row)

    except Exception as e:
        raise e

    return rois


def convert_rois_to_rasters(
        rois: str,
        out_folder: str
) -> tuple[str, str, str]:
    """
    Takes a filled in region of interest shapefile and
    converts each fraction class within (0-2) to seperate
    fraction rasters for use in regression model.

    :param rois:
    :return:
    """

    try:
        # convert other class (c_) rois to a raster
        out_c_0 = os.path.join(out_folder, 'roi_frac_c_0.tif')
        arcpy.conversion.PolygonToRaster(in_features=rois,
                                         value_field='c_0',
                                         out_rasterdataset=out_c_0,
                                         cellsize=10)

        # convert native class (c_1) rois to a raster
        out_c_1 = os.path.join(out_folder, 'roi_frac_c_1.tif')
        arcpy.conversion.PolygonToRaster(in_features=rois,
                                         value_field='c_1',
                                         out_rasterdataset=out_c_1,
                                         cellsize=10)

        # convert weed class (c_2) rois to a raster
        out_c_2 = os.path.join(out_folder, 'roi_frac_c_2.tif')
        arcpy.conversion.PolygonToRaster(in_features=rois,
                                         value_field='c_2',
                                         out_rasterdataset=out_c_2,
                                         cellsize=10)

    except Exception as e:
        raise e

    return out_c_0, out_c_1, out_c_2


def regress(
        exp_vars: str,
        sample_points: str,
        classvalue: str,
        classdesc: str,
) -> arcpy.Raster:
    """

    :param exp_vars:
    :param pnts:
    :param classvalue:
    :param classdesc:
    :return:
    """

    try:
        # train regression model for current target raster
        tmp_ecd = os.path.join(arcpy.env.scratchFolder, f'ecd_{classvalue}.ecd')
        arcpy.ia.TrainRandomTreesRegressionModel(in_rasters=exp_vars,
                                                 in_target_data=sample_points,
                                                 out_regression_definition=tmp_ecd,
                                                 target_value_field=classvalue,
                                                 max_num_trees=250,
                                                 max_tree_depth=50,
                                                 max_samples=-1,
                                                 percent_testing=20)

        # predict using trained model ecd file
        ras_reg = arcpy.ia.PredictUsingRegressionModel(in_rasters=exp_vars,
                                                       in_regression_definition=tmp_ecd)

        # read ecd file (it is just a json file)
        df = pd.read_json(tmp_ecd)

        # prepare regression r2 metrics and notify
        r2_train = round(df['Definitions'][0]['Train R2 at train locations'], 2)
        r2_valid = round(df['Definitions'][0]['Test R2 at train locations'], 2)
        arcpy.AddMessage(f'  - R2 Train: {str(r2_train)} / Validation: {str(r2_valid)}')

        # prepare error r2 metrics and notify
        er_train = round(df['Definitions'][0]['Train Error at train locations'], 2)
        er_valid = round(df['Definitions'][0]['Test Error at train locations'], 2)
        arcpy.AddMessage(f'  - Error Train: {str(er_train)} / Validation: {str(er_valid)}')

    except Exception as e:
        raise e

    return ras_reg


def regress_deprecated(
        in_rois: str,
        in_classvalue: str,
        in_class_desc: str,
        in_exp_vars: list,
        out_regress_tif: str,
        out_cmatrix_csv: str,
) -> None:

    # convert csv to dbf for function
    out_cmatrix_dbf = os.path.splitext(out_cmatrix_csv)[0] + '.dbf'

    # perform regression
    # FIXME: this fails if we run via PyCharm - works ok via toolbox... threading?
    arcpy.stats.Forest(prediction_type='PREDICT_RASTER',
                       in_features=in_rois,
                       variable_predict=in_classvalue,
                       explanatory_rasters=in_exp_vars,
                       output_raster=out_regress_tif,
                       explanatory_rasters_matching=in_exp_vars,
                       number_of_trees=100,
                       percentage_for_training=25,
                       number_validation_runs=5,
                       output_validation_table=out_cmatrix_dbf)

    # create output confususion matrix as a csv
    #out_cmx_fn = f'cm_{dt}_{classvalue}_{class_desc}.csv'.replace('-', '_')
    #out_cmx_csv = os.path.join(fraction_folder, out_cmx_fn)

    # convert dbf to csv
    arcpy.conversion.ExportTable(in_table=out_cmatrix_dbf,
                                 out_table=out_cmatrix_csv)

    # delete dbf
    arcpy.management.Delete(out_cmatrix_dbf)

    # read csv with pandas and get average r-squares
    avg_r2 = pd.read_csv(out_cmatrix_csv)['R2'].mean().round(3)
    arcpy.AddMessage(f'> Average R2 for {in_classvalue} ({in_class_desc}): {str(avg_r2)}')

    return