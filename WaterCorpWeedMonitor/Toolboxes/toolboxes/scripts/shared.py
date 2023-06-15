
import os
import pandas as pd
import numpy as np
import arcpy

from scipy.stats import kendalltau


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


def extract_var_importance(in_ecd):
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