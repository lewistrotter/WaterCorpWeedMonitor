
import os
import numpy as np
import pandas as pd
import arcpy


def make_grayscale(
        in_band_list: list,
        levels: int
) -> arcpy.Raster:
    """
    Takes a list of raster bands and does a pca on them. The
    first principal component is selected and then converted
    to a grayscale, then rescaled to 0-255.

    :param in_band_list: List of band paths.
    :param levels: Number of grayscale levels.
    :return: Grayscale as a arcpy.Raster object.
    """

    try:
        # generate pca and return first principal band only
        tmp_pca = arcpy.sa.PrincipalComponents(in_raster_bands=in_band_list,
                                               number_components=1)

        # quantise values into n grayscale levels and normalise
        tmp_slc = arcpy.sa.Slice(in_raster=tmp_pca,
                                 number_zones=levels,
                                 slice_type='EQUAL_INTERVAL',
                                 base_output_zone=0)

        # rescale quantised values back to 16 classes ranging 0 to 255 (grayscale)
        tmp_rsc = arcpy.sa.RescaleByFunction(in_raster=tmp_slc,
                                             transformation_function='LINEAR',
                                             from_scale=0,
                                             to_scale=255)

    except Exception as e:
        raise e

    return tmp_rsc

def validate_rois(
        in_roi_feat: str,
        extent: arcpy.Extent
) -> None:
    """
    Takes a layer or path to training are features and checks if they're
    valid. Basically, if the data is missing required fields, is not in
    correct projection, has < 10 polygons per class exist, has too many
    polygons outside raster extent, has more than 3 classes, etc. an
    error will result. Catch the error on the other end to handle invalid
    features.

    :param in_roi_feat: A layer or path of training features.
    :param extent: An arcpy.Extent object with raster extent.
    :return: None.
    """

    try:
        # get file path, spatial reference and fields names
        fc_desc = arcpy.Describe(in_roi_feat)
        fc_srs = fc_desc.spatialReference
        fields = [f.name for f in arcpy.ListFields(in_roi_feat)]

    except Exception as e:
        raise e

    # check if spatial reference is in wgs84 utm zone 50s (32750)
    if fc_srs == 'Unknown' or fc_srs.factoryCode != 3577:
        raise ValueError('Training areas not projected in WGS84 UTM Zone 50S (32750).')

    # check if required fields exist
    req_fields = ['Classname', 'Classvalue']
    for required_field in req_fields:
        if required_field not in fields:
            raise ValueError(f'Field {req_fields} missing from training area feature.')

    try:
        # extract unique class names, values and geometries
        cnames, cvalues, geoms = [], [], []
        with arcpy.da.SearchCursor(in_roi_feat, ['Classname', 'Classvalue', 'Shape@']) as cursor:
            for row in cursor:
                cnames.append(row[0])
                cvalues.append(row[1])
                geoms.append(row[2])

        # get arrays of unique class names, values and counts
        unq_cn, cnt_cn = np.unique(cnames, return_counts=True)
        unq_cv, cnt_cv = np.unique(cvalues, return_counts=True)

    except Exception as e:
        raise e

    # check if not 3 class names/values each with >= 10 geometries, error
    if len(unq_cn) != 3 or np.any(cnt_cn < 10) or len(unq_cv) != 3 or np.any(cnt_cv < 10):
        raise ValueError('Training features must have 3 classes (Native, Weed, Other), each with >= 10 polygons.')

    try:
        # count number of rois within uav capture extent
        num_rois_in = 0
        for geom in geoms:
            if extent.contains(geom):
                num_rois_in += 1

    except Exception as e:
        raise e

    # check if at least 30 rois in uav capture extent (10 each), else error
    if num_rois_in < 30:
        raise ValueError('Not enough training area features within UAV image extent.')

    return


def split_rois(
        in_roi_feat: str,
        out_folder: str,
        pct_split: float,
        equal_classes: bool
) -> tuple:
    """
    Split shapefile of rois into training and validation sets. Each set
    is a seperate shapefile.

    :param in_roi_feat: Path to roi shapefile.
    :param out_folder: Folder path for output train/validate shapefiles
    :param pct_split: Percentage of train/validate split (e.g., 0.5 for 50%)
    :param equal_classes: Force all classes to have equal number of rois (10 each).
    :return: Paths to new train and validate rois shapefiles.
    """

    try:
        # read shapefile into pandas
        with arcpy.da.SearchCursor(in_roi_feat, ['FID', 'Classvalue']) as cursor:
            rois = [{'fid': row[0], 'classvalue': row[1]} for row in cursor]
            df = pd.DataFrame.from_dict(rois).sort_values(by='classvalue')

        # ensure each class <= 10 samples via randstrat sampling
        if equal_classes is True:
            df = df.groupby('classvalue', group_keys=False).apply(lambda x: x.sample(10))

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

    except Exception as e:
        raise e

    return out_train_shp, out_valid_shp


def classify(
        in_train_roi: str,
        in_comp_ras: arcpy.Raster,
        out_ecd: str,
        out_class_tif: str,
        num_trees: int,
        num_depth: int
) -> str:
    """
    Takes training roi shapefile path and a composited raster and classifies
    it using a random forest classification. A path to the resulting
    classified raster is provided on output.

    :param in_train_roi: Path to a training roi shapefile.
    :param in_comp_ras: Path to composite raster.
    :param out_ecd: Path to an output ecd file.
    :param out_class_tif: Path of output classified raster.
    :param num_trees: Number of random forest trees.
    :param num_depth: Random forest tree depth number.
    :return: Path to output classified raster.
    """

    try:
        # train the rf classifier
        ecd = arcpy.sa.TrainRandomTreesClassifier(in_raster=in_comp_ras,
                                                  in_training_features=in_train_roi,
                                                  out_classifier_definition=out_ecd,
                                                  max_num_trees=num_trees,
                                                  max_tree_depth=num_depth,
                                                  max_samples_per_class=1000)

        # classify composite via trained model and save
        ras_classified = arcpy.sa.ClassifyRaster(in_raster=in_comp_ras,
                                                 in_classifier_definition=ecd)

        # save raster
        ras_classified.save(out_class_tif)

    except Exception as e:
        raise e

    return out_class_tif


def extract_var_importance(
        in_ecd: str
) -> list:
    """
    Extracts variable importance info from an existing ecd file.

    :param in_ecd: Path to an ecd file.
    :return: List of variable importance
    """

    # set var importance output
    var_imps = []

    try:
        # read ecd file (it is just a json file)
        df = pd.read_json(in_ecd)

        # get list of variable importance
        var_imps = df['Definitions'][0]['VariableImportance']

        # convert to rounded list 2 decimal places
        var_imps = [round(v, 2) for v in var_imps]

    except Exception as e:
        raise e

    return var_imps


def assess_accuracy(
        out_class_tif: str,
        in_valid_shp: str,
        out_cmatrix: str
) -> tuple:
    """
    Generates a confusion matrix from classified raster and validation
    shapefile. Output is several variables about validation accuracy
    and a path to the resulting confusion matrix.

    :param out_class_tif: Path to classified raster.
    :param in_valid_shp: Path to validation shapefile of points.
    :param out_cmatrix: Path to output confusion matrix csv file.
    :return: Lists of class codes, prod. acc, user acc., overall acc., kappa.
    """

    try:
        # create random 5k points equal spread across class rois
        arcpy.sa.CreateAccuracyAssessmentPoints(in_class_data=in_valid_shp,
                                                out_points=r'memory\valid_pts',
                                                target_field='GROUND_TRUTH',
                                                num_random_points=5000,
                                                sampling='EQUALIZED_STRATIFIED_RANDOM')

        # add the classified values to the assessment points... how obtuse
        out_acc_pnts = r'memory\acc_pnts'
        arcpy.sa.UpdateAccuracyAssessmentPoints(in_class_data=out_class_tif,
                                                in_points=r'memory\valid_pts',
                                                out_points=out_acc_pnts,
                                                target_field='CLASSIFIED')

        # compute confusion matrix
        arcpy.sa.ComputeConfusionMatrix(in_accuracy_assessment_points=out_acc_pnts,
                                        out_confusion_matrix=out_cmatrix)

        # get confusion matrix as pandas
        df = pd.read_csv(out_cmatrix)

        # get class codes. expects only 3 classes
        codes = list(df.columns[2:-3].values)

        # get accuracy measures as floats
        p_acc = df.iloc[-2, 2:-3].astype(np.float32)
        u_acc = df.iloc[0:-3, -2].astype(np.float32)
        oa = df.iloc[-2, -2].astype(np.float32)
        kappa = df.iloc[-1, -1].astype(np.float32)

    except Exception as e:
        raise e

    return codes, p_acc, u_acc, oa, kappa


def rebuild_raster_class_attributes(
        in_ras: str
) -> None:
    """
    When a filter or tool is applied to a classified raster, the class
    information is stripped. This adds a table back on and rebuilds
    the class names based on the value field.

    :param in_ras: Path to the classified raster.
    :return: None.
    """

    try:
        # rebuild raster attribute table
        arcpy.management.BuildRasterAttributeTable(in_raster=in_ras,
                                                   overwrite='OVERWRITE')

        # add expected columns back on
        arcpy.management.AddFields(in_table=in_ras,
                                   field_description='Classvalue LONG;Class_name TEXT')

        # loop table rows and update values
        fields = ['Value', 'Classvalue', 'Class_name']
        with arcpy.da.UpdateCursor(in_ras, fields) as cursor:
            for row in cursor:
                if row[0] == 0:
                    label = 'Other'
                elif row[0] == 1:
                    label = 'Native'
                elif row[0] == 2:
                    label = 'Weed'
                else:
                    label = None

                if label is not None:
                    row[1] = row[0]
                    row[2] = label
                    cursor.updateRow(row)

    except Exception as e:
        raise e

    return
