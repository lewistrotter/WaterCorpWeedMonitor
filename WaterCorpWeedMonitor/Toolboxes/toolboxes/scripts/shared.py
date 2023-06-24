
import os
import pandas as pd
import numpy as np
import xarray as xr
import arcpy

from osgeo import gdal
from scipy.stats import kendalltau

def get_bbox_from_raster(
        in_raster: arcpy.Raster
) -> tuple:
    """

    :param in_raster:
    :return:
    """

    try:
        # get description object from input raster
        extent = arcpy.Describe(in_raster).extent

        # extract sw, ne corners
        x_min, y_min = float(extent.XMin), float(extent.YMin)
        x_max, y_max = float(extent.XMax), float(extent.YMax)

        # combine int uple
        bbox = x_min, y_min, x_max, y_max

    except:
        return ()

    return bbox


def expand_box_by_metres(
        bbox: tuple,
        metres: float
) -> tuple:
    """

    :param bbox:
    :return:
    """

    # unpack bbox
    x_min, y_min, x_max, y_max = bbox

    # minus 30 x and y min
    x_min -= metres
    y_min -= metres

    # plus x max and y max
    x_max += metres
    y_max += metres

    return x_min, y_min, x_max, y_max


def split_rois(
        in_roi_feat,
        out_folder,
        pct_split=0.5,
        equal_classes=False
):
    """
    Split shapefile of rois into training and validate shapefiles.

    :param in_roi_feat: Full path to roi shapefile.
    :param out_folder: Folder path for output shapefiles
    :param pct_split: Percentage of train/validate split (e.g., 0.5 for 50%)
    :param equal_classes: Force all classes to have equal number of rois (50).
    :return: Paths to train and validate rois
    """

    # read shapefile into pandas
    with arcpy.da.SearchCursor(in_roi_feat, ['FID', 'Classvalue']) as cursor:
        rois = [{'fid': row[0], 'classvalue': row[1]} for row in cursor]
        df = pd.DataFrame.from_dict(rois).sort_values(by='classvalue')

    # ensure each class <= 50 samples via randstrat sampling
    if equal_classes is True:
        df = df.groupby('classvalue', group_keys=False).apply(lambda x: x.sample(50))  # TODO: user change?

    # subset into train set randomly (e.g., 50%)
    df_train = df.groupby('classvalue', group_keys=False)
    df_train = df_train.apply(lambda x: x.sample(frac=pct_split))

    # set remaining as validate (50%)
    df_valid = df[~df.isin(df_train)].dropna().astype('int16')

    # get fids of train, validate sets as string
    train_fids = str(tuple(df_train['fid'].values))
    valid_fids = str(tuple(df_valid['fid'].values))

    # select train records based on fid
    out_train_shp = os.path.join(out_folder, 'train.shp')
    arcpy.analysis.Select(in_features=in_roi_feat,
                          out_feature_class=out_train_shp,
                          where_clause=f'FID IN {train_fids}')

    # select valid records based on fid
    out_valid_shp = os.path.join(out_folder, 'valid.shp')
    arcpy.analysis.Select(in_features=in_roi_feat,
                          out_feature_class=out_valid_shp,
                          where_clause=f'FID IN {valid_fids}')

    return out_train_shp, out_valid_shp


def classify(
        in_train_roi,
        in_comp_tif,
        out_ecd,
        out_class_tif,
        num_trees=250,
        num_depth=50
):
    """
    Takes training roi and a composited raster and classifies it.

    :param in_train_roi: Path to a training roi shapefile.
    :param in_comp_tif: Path to composite raster.
    :param out_ecd: Path to an output ecd file.
    :param out_class_tif: Path of output classified raster.
    :param num_trees: Number of random forest trees.
    :param num_depth: Random forest tree depth number.
    :return: Path to output classified raster.
    """

    # train the rf classifier
    ecd = arcpy.sa.TrainRandomTreesClassifier(in_raster=in_comp_tif,
                                              in_training_features=in_train_roi,
                                              out_classifier_definition=out_ecd,
                                              max_num_trees=num_trees,
                                              max_tree_depth=num_depth,
                                              max_samples_per_class=1000)

    # classify composite via trained model and save
    ras_classified = arcpy.sa.ClassifyRaster(in_raster=in_comp_tif,
                                             in_classifier_definition=ecd)

    # save raster
    ras_classified.save(out_class_tif)

    return out_class_tif


def assess_accuracy(
        out_class_tif,
        in_valid_shp,
        out_cmatrix
):
    """
    Generates a confusion matrix from classified raster and validation
    shapefile.

    :param out_class_tif: Path to classified raster.
    :param in_valid_shp: Path to validation shapefile of points.
    :param out_cmatrix: Path to output confusion matrix csv file.
    :return: Lists of class codes, prod. acc, user acc., overall acc., kappa.
    """

    # create random 50k points equal spread across class rois
    arcpy.sa.CreateAccuracyAssessmentPoints(in_class_data=in_valid_shp,
                                            out_points=r'memory\valid_pts',
                                            target_field='GROUND_TRUTH',
                                            num_random_points=50000,
                                            sampling='EQUALIZED_STRATIFIED_RANDOM')

    # add the classified values to the assessment points... how obtuse
    out_acc_pnts = r'memory\acc_pnts'  # os.path.join(out_folder, 'acc_pnts.shp')
    arcpy.sa.UpdateAccuracyAssessmentPoints(in_class_data=out_class_tif,
                                            in_points=r'memory\valid_pts',
                                            out_points=out_acc_pnts,
                                            target_field='CLASSIFIED')

    # TODO: check if any updateaccuracyassessment points are -1 in classified column - they have nans in them

    # compute confusion matrix
    #out_cmatrix = os.path.join(out_folder, 'cmatrix.csv')
    arcpy.sa.ComputeConfusionMatrix(in_accuracy_assessment_points=out_acc_pnts,
                                    out_confusion_matrix=out_cmatrix)

    # get confusion matrix as pandas
    df = pd.read_csv(out_cmatrix)

    # get class codes
    codes = list(df.columns[2:-3].values)

    # get raw accuracy measures
    p_acc = df.iloc[-2, 2:-3].astype(np.float32)
    u_acc = df.iloc[0:-3, -2].astype(np.float32)
    oa = df.iloc[-2, -2].astype(np.float32)
    kappa = df.iloc[-1, -1].astype(np.float32)

    # clean measures
    p_acc = p_acc
    u_acc = u_acc.astype(np.float32)
    oa = oa.astype(np.float32)
    kappa = kappa.astype(np.float32)

    return codes, p_acc, u_acc, oa, kappa


def extract_var_importance(
        in_ecd: str
) -> list:
    """
    Extracts variable importance info from the ecd file.

    :param in_ecd: Input classifier ecd file.
    :return: List of variable importance
    """

    # set var importance output
    var_imps = []

    try:
        # read ecd file (its just a json)
        df = pd.read_json(in_ecd)

        # get list of variable importance
        var_imps = df['Definitions'][0]['VariableImportance']

        # convert to rounded list 2 decimal places
        var_imps = [round(v, 2) for v in var_imps]

    except Exception as e:
        arcpy.AddWarning('Could not extract variable importance.')
        raise  # return

    return var_imps


def export_xr_vars_into_tifs(
        da: xr.Dataset,
        out_folder: str
) -> list:

    # init out band list
    out_band_tifs = []

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

    return out_band_tifs


def build_lr_freqs_rois_from_hr_xr(
        ras_lr: arcpy.Raster,
        da_hr: xr.DataArray
) -> str:

    # get a centroid point for each grid cell on sentinel raster
    tmp_pnts = r'memory\tmp_points'
    arcpy.conversion.RasterToPoint(in_raster=ras_lr,
                                   out_point_features=tmp_pnts,
                                   raster_field='Value')

    # convert points into 10m buffer circles (5 m half cell res)
    tmp_buff = r'memory\tmp_circle_buff'
    arcpy.analysis.PairwiseBuffer(in_features=tmp_pnts,
                                  out_feature_class=tmp_buff,
                                  buffer_distance_or_field='5 Meters')

    # convert circle buffers to square rectangles
    tmp_env = r'memory\tmp_square_buff'
    arcpy.management.FeatureEnvelopeToPolygon(in_features=tmp_buff,
                                              out_feature_class=tmp_env)

    # add required fields to envelope shapefile
    arcpy.management.AddFields(in_table=tmp_env,
                               field_description="c_0 FLOAT;c_1 FLOAT;c_2 FLOAT;inc SHORT")

    # extract all high-res class values within each low res pixel
    with arcpy.da.UpdateCursor(tmp_env, ['c_0', 'c_1', 'c_2', 'inc', 'SHAPE@']) as cursor:
        for row in cursor:
            # get x and y window by slices for each polygon
            x_slice = slice(row[-1].extent.XMin, row[-1].extent.XMax)
            y_slice = slice(row[-1].extent.YMin, row[-1].extent.YMax)

            # extract window values from high-res
            arr = da_hr.sel(x=x_slice, y=y_slice).values

            # if values exist...
            if arr.size != 0 and ~np.all(arr == -999):
                # flatten array
                arr = arr.flatten()

                # remove all -999 values
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

                # project existing classes and freqs onto map
                class_map.update(dict(zip(classes, freqs)))

                # fill in row values
                row[0:4] = list(class_map.values())
            else:
                # set all freqs to zero when nodata exists
                row[3] = 0

            # update row
            cursor.updateRow(row)

    return tmp_env


def regress(
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


def add_raster_to_map(
        in_ras: str
) -> None:

    # re-enable add to map
    arcpy.env.addOutputsToMap = True

    try:
        # read current project and add tif
        aprx = arcpy.mp.ArcGISProject('CURRENT')
        mp = aprx.activeMap
        mp.addDataFromPath(in_ras)

    except Exception as e:
        arcpy.AddWarning('Could not add raster to active map. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # disable add to map
    arcpy.env.addOutputsToMap = False

    return


def apply_uav_classified_layer_symbology(
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
            arcpy.AddWarning('Could not find UAV classified layer on map.')
            return

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
        arcpy.AddWarning('Could not apply symbology to UAV classified layer. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # disable add to map
    arcpy.env.addOutputsToMap = False

    return


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