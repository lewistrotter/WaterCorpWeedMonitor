
import os
import json
import datetime
import pandas as pd
import numpy as np
import arcpy

from scripts import shared


def execute(
        parameters
        # messages # TODO: implement
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    in_project_file = parameters[0].valueAsText
    in_flight_datetime = parameters[1].value
    #in_include_prior = parameters[2].value  # this parameter is only for ui control
    in_roi_feat = parameters[3].value

    # inputs for testing only
    # in_project_file = r'C:\Users\Lewis\Desktop\testing\city beach demo\meta.json'
    # in_flight_datetime = '2023-06-16 10:58:42'  # '2023-06-07 12:34:34' tmp2
    # #in_include_prior = False  # this parameter is only for ui control
    # in_roi_feat = r'D:\Work\Curtin\Water Corp Project - General\Processed\City Beach\Classification\Final\train_test_rois_smaller_bc_grp_nvwvo_wgs_z50s.shp'

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE ENVIRONMENT

    arcpy.SetProgressor('default', 'Preparing environment...')

    # check if user has spatial analyst, error if not
    if arcpy.CheckExtension('Spatial') != 'Available':
        arcpy.AddError('Spatial Analyst license is unavailable.')
        raise  # return
    # TODO: remove below if wc has no ia
    elif arcpy.CheckExtension('ImageAnalyst') != 'Available':
        arcpy.AddError('Image Analyst license is unavailable.')
        raise  # return
    else:
        arcpy.CheckOutExtension('Spatial')
        arcpy.CheckOutExtension('ImageAnalyst')  # TODO: remove if wc has no ia

    # set data overwrites and mapping
    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = False

    # set current workspace to scratch folder
    arcpy.env.workspace = arcpy.env.scratchFolder

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK PROJECT FOLDER STRUCTURE

    arcpy.SetProgressor('default', 'Checking project folders and files...')

    # check if input project file exists
    if not os.path.exists(in_project_file):
        arcpy.AddError('Project file does not exist.')
        raise  # return

    # get top-level project folder from project file
    in_project_folder = os.path.dirname(in_project_file)

    # check if required project folders already exist, error if so
    sub_folders = ['grid', 'uav_captures', 'sat_captures', 'visualise']
    for sub_folder in sub_folders:
        sub_folder = os.path.join(in_project_folder, sub_folder)
        if not os.path.exists(sub_folder):
            arcpy.AddError('Project is missing required folders.')
            raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ AND CHECK METADATA

    arcpy.SetProgressor('default', 'Reading and checking metadata...')

    try:
        # read project json file
        with open(in_project_file, 'r') as fp:
            meta = json.load(fp)

    except Exception as e:
        arcpy.AddError('Could not read metadata. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # check if any captures exist (will be >= 4)
    if len(meta) < 4:
        arcpy.AddError('Project has no UAV capture data.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT SELECTED UAV CAPTURE METADATA

    arcpy.SetProgressor('default', 'Extracting selected UAV capture metadata...')

    # exclude top-level metadata items
    exclude_keys = ['project_name', 'date_created', 'date_rehab']

    # extract selected metadata item based on capture date
    meta_item = None
    for k, v in meta.items():
        if k not in exclude_keys:
            if v['capture_date'] == in_flight_datetime:
                meta_item = v

    # check if meta item exists, else error
    if meta_item is None:
        arcpy.AddError('Could not find selected UAV capture in metadata file.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ CAPTURE RASTER BANDS

    arcpy.SetProgressor('default', 'Reading capture raster bands...')

    # build capture folder and band folder
    capture_folder = os.path.join(in_project_folder, 'uav_captures', meta_item['capture_folder'])
    bands_folder = os.path.join(capture_folder, 'bands')

    # create raw band map to maintain band order
    band_map = {
        'blue': os.path.join(bands_folder, 'blue.tif'),
        'green': os.path.join(bands_folder, 'green.tif'),
        'red': os.path.join(bands_folder, 'red.tif'),
        'redge': os.path.join(bands_folder, 'redge.tif'),
        'nir': os.path.join(bands_folder, 'nir.tif'),
    }

    # check if bands exist
    for v in band_map.values():
        if not os.path.exists(v):
            arcpy.AddError('Some required UAV bands missing from capture folder.')
            raise  # return

    try:
        # open bands as seperate rasters
        blue = arcpy.Raster(band_map['blue'])
        green = arcpy.Raster(band_map['green'])
        red = arcpy.Raster(band_map['red'])
        redge = arcpy.Raster(band_map['redge'])
        nir = arcpy.Raster(band_map['nir'])

    except Exception as e:
        arcpy.AddError('Could not read raster bands. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE VEGETATION INDICES

    arcpy.SetProgressor('default', 'Calculating vegetation indices...')

    # create vegetation index map to maintain band order
    vi_map = {
        'ndvi': os.path.join(bands_folder, 'vi_ndvi.tif'),
        'endvi': os.path.join(bands_folder, 'vi_endvi.tif'),
        'ndrei': os.path.join(bands_folder, 'vi_ndrei.tif'),
        'ngrdi': os.path.join(bands_folder, 'vi_ngrdi.tif'),
        'rgbvi': os.path.join(bands_folder, 'vi_rgbvi.tif'),
        'osavi': os.path.join(bands_folder, 'vi_osavi.tif'),
        'kndvi': os.path.join(bands_folder, 'vi_kndvi.tif'),
    }

    # set up step-wise progressor
    arcpy.SetProgressor('step', 'Calculating vegetation indices...', 0, len(vi_map))

    try:
        # iter each vi and calculate
        for k, v in vi_map.items():
            if k == 'ndvi':
                ras = (nir - red) / (nir + red)
            elif k == 'endvi':
                ras = ((nir + green) - (2 * blue)) / ((nir + green) + (2 * blue))
            elif k == 'ndrei':
                ras = (nir - redge) / (nir + redge)
            elif k == 'ngrdi':
                ras = (green - red) / (green + red)
            elif k == 'rgbvi':
                ras = (green ** 2 - (red * blue)) / (green ** 2 + (red * blue))
            elif k == 'osavi':
                ras = (1 + 0.16) * ((nir - red) / (nir + red + 0.16))
            elif k == 'kndvi':
                ras = arcpy.sa.TanH(((nir - red) / (nir + red)) ** 2)
            else:
                raise ValueError('Requested vegetation index does not exist.')

            # save to associated path
            ras.save(v)

            # increment progressor
            arcpy.SetProgressorPosition()

    except Exception as e:
        arcpy.AddError('Could not calculate vegetation indices. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # reset progressor
    arcpy.ResetProgressor()

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE GLCM TEXTURES

    arcpy.SetProgressor('default', 'Calculating GLCM textures...')

    # generate pca and return first principal band only
    tmp_pca = arcpy.sa.PrincipalComponents(in_raster_bands=list(band_map.values()),
                                           number_components=1)

    # quantise values into 16 grayscale levels and normalise
    tmp_slc = arcpy.sa.Slice(in_raster=tmp_pca,
                             number_zones=16,
                             slice_type='EQUAL_INTERVAL',
                             base_output_zone=0)

    # rescale quantised values back to 16 classes ranging 0 to 255 (grayscale)
    tmp_rsc = arcpy.sa.RescaleByFunction(in_raster=tmp_slc,
                                         transformation_function='LINEAR',
                                         from_scale=0,
                                         to_scale=255)

    # save grayscale to scratch - need a physical file
    #tmp_gry.save('tmp_gry.tif')

    # create texture indices map to maintain band order
    tx_map = {
        'mean': os.path.join(bands_folder, 'tx_mean.tif'),
        'max': os.path.join(bands_folder, 'tx_max.tif'),
        'min': os.path.join(bands_folder, 'tx_min.tif'),
        'stdev': os.path.join(bands_folder, 'tx_stdev.tif'),
        'range': os.path.join(bands_folder, 'tx_range.tif'),
        #'contrast': os.path.join(bands_folder, 'tx_contrast.tif'),
        #'correlation': os.path.join(bands_folder, 'tx_correlation.tif'),
        #'dissimilarity': os.path.join(bands_folder, 'tx_dissimilarity.tif'),
        #'entropy': os.path.join(bands_folder, 'tx_entropy.tif'),
        #'homogeneity': os.path.join(bands_folder, 'tx_homogeneity.tif'),
        #'secmoment': os.path.join(bands_folder, 'tx_secmoment.tif'),
        #'variance': os.path.join(bands_folder, 'tx_variance.tif'),
    }

    # set up step-wise progressor
    arcpy.SetProgressor('step', 'Calculating GLCM textures...', 0, len(tx_map))

    # TODO: calculate glcm textures...
    # for now, fall back on esri tech

    try:
        # setup neighbourhood object
        win = arcpy.ia.NbrRectangle(5, 5, 'CELL')

        # iter each texture and calculate
        for k, v in tx_map.items():
            if k == 'mean':
                ras = arcpy.ia.FocalStatistics(in_raster=tmp_rsc,
                                               neighborhood=win,
                                               statistics_type='MEAN')
            elif k == 'max':
                ras = arcpy.ia.FocalStatistics(in_raster=tmp_rsc,
                                               neighborhood=win,
                                               statistics_type='MAXIMUM')
            elif k == 'min':
                ras = arcpy.ia.FocalStatistics(in_raster=tmp_rsc,
                                               neighborhood=win,
                                               statistics_type='MINIMUM')
            elif k == 'stdev':
                ras = arcpy.ia.FocalStatistics(in_raster=tmp_rsc,
                                               neighborhood=win,
                                               statistics_type='STD')
            elif k == 'range':
                ras = arcpy.ia.FocalStatistics(in_raster=tmp_rsc,
                                               neighborhood=win,
                                               statistics_type='RANGE')
            else:
                raise ValueError('Requested texture index does not exist.')

            # save to associated path
            ras.save(v)

            # increment progressor
            arcpy.SetProgressorPosition()

    except Exception as e:
        arcpy.AddError('Could not calculate textures indices. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # reset progressor
    arcpy.ResetProgressor()

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE CANOPY HEIGHT MODEL (CHM)

    arcpy.SetProgressor('default', 'Calculating canopy height model...')

    # create chm map to maintain band order
    chm_map = {
        'chm': os.path.join(bands_folder, 'chm.tif')
    }

    # create dsm and dtm file paths
    dsm_path = os.path.join(bands_folder, 'dsm.tif')
    dtm_path = os.path.join(bands_folder, 'dtm.tif')

    # check if both exist
    for v in [dsm_path, dtm_path]:
        if not os.path.exists(v):
            arcpy.AddError('Digital terrain/surface elevation bands missing from capture folder.')
            raise  # return

    try:
        # read dsm and dtm rasters
        dsm = arcpy.Raster(dsm_path)
        dtm = arcpy.Raster(dtm_path)

        # calculate chm via dsm - dtm
        ras = dsm - dtm

        # save to associated path
        ras.save(os.path.join(bands_folder, chm_map['chm']))

    except Exception as e:
        arcpy.AddError('Could not calculate canopy height model. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ REGIONS OF INTEREST FEATURE

    arcpy.SetProgressor('default', 'Reading field training area features...')

    try:
        # get file path, spatial reference and fields
        fc_desc = arcpy.Describe(in_roi_feat)
        fc_srs = fc_desc.spatialReference
        fields = [f.name for f in arcpy.ListFields(in_roi_feat)]

    except Exception as e:
        arcpy.AddError('Cannot read featureclass. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK REGIONS OF INTEREST FEATURE

    arcpy.SetProgressor('default', 'Checking field training area features...')

    # check if spatial reference is in wgs84 utm zone 50s)
    if fc_srs == 'Unknown' or fc_srs.factoryCode != 32750:
        arcpy.AddError('Training areas must be projected in WGS84 UTM Zone 50S (32750).')
        raise  # return

    # check if required fields exist
    req_fields = ['Classname', 'Classvalue']
    for required_field in req_fields:
        if required_field not in fields:
            arcpy.AddError(f'Field {req_fields} missing from training area feature.')
            raise  # return

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

    # check if not 3 class names/values each with >= 20 geometries, error
    if len(unq_cn) != 3 or np.any(cnt_cn < 20) or len(unq_cv) != 3 or np.any(cnt_cv < 20):
        arcpy.AddError('Training area features must have 3 classes (Native, Weed, Other), each with >= 20 polygons.')
        raise  # return

    # count number of rois within uavc capture extent
    num_rois_in = 0
    for geom in geoms:
        if blue.extent.contains(geom):
            num_rois_in += 1

    # check if at least 60 rois in uav capturee xtent (20 each), error otherwise
    if num_rois_in < 60:
        arcpy.AddError('Not enough training area features within UAV image extent.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CLASSIFY UAV CAPTURE

    arcpy.SetProgressor('default', 'Classifying UAV capture...')

    # create classify folder if not already
    classify_folder = os.path.join(capture_folder, 'classify')
    if not os.path.exists(classify_folder):
        os.mkdir(classify_folder)

    # build variable name and path lists (we use former later)
    var_names = list(band_map.keys()) + list(vi_map.keys()) + list(tx_map.keys()) + list(chm_map.keys())
    var_paths = list(band_map.values()) + list(vi_map.values()) + list(tx_map.values()) + list(chm_map.values())

    try:
        # TODO: uncomment below if wc has no ia
        #with arcpy.EnvManager(pyramid='NONE'):
            #tmp_comp = os.path.join(classify_folder, 'comp')  # dont use tif, arcpy bug
            #arcpy.management.CompositeBands(in_rasters=variables,
                                            #out_raster=tmp_comp)

        # TODO: remove below if wc has no ia
        # composite all variables
        tmp_cmp = arcpy.ia.CompositeBand(var_paths)

    except Exception as e:
        arcpy.AddError('Could not create composite of bands. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # set up step-wise progressor
    arcpy.SetProgressor('step', None, 0, 5)

    results = []
    for i in range(1, 6):

        arcpy.AddMessage(f'Classifying UAV capture ({i} / 5)...')

        try:
            # split rois into train / valid splits (50% / 50%)
            train_shp, valid_shp = shared.split_rois(in_roi_feat=in_roi_feat,
                                                     out_folder=classify_folder,
                                                     pct_split=0.5,
                                                     equal_classes=False)

            # classify the uav capture via random forest, save raster and output path
            out_ecd = os.path.join(classify_folder, f'ecd_{i}.ecd')
            out_class_tif = os.path.join(classify_folder, f'rf_{i}.tif')
            out_class_tif = shared.classify(in_train_roi=train_shp,
                                            in_comp_tif=tmp_cmp,
                                            out_ecd=out_ecd,
                                            out_class_tif=out_class_tif,
                                            num_trees=250,
                                            num_depth=50)

            # validate the classification via confuse matrix
            out_cmatrix = os.path.join(classify_folder, f'cmatrix_{i}.csv')
            codes, p_acc, u_acc, oa, kappa = shared.assess_accuracy(out_class_tif=out_class_tif,
                                                                    in_valid_shp=valid_shp,
                                                                    out_cmatrix=out_cmatrix)

            # extract variable importance list
            var_imps = shared.extract_var_importance(in_ecd=out_ecd)

            # show user basic accuracy metrics
            arcpy.AddMessage(f'> Overall Accuracy: {str(np.round(oa, 3))}.')
            arcpy.AddMessage(f'> Kappa: {str(np.round(kappa, 3))}.')

            # store results
            results.append({
                'cv': i,
                'codes': codes,
                'p_acc': p_acc,
                'u_acc': u_acc,
                'oa': oa,
                'kappa': kappa,
                'var_imps': var_imps,
                'out_ecd': out_ecd,
                'out_class_tif': out_class_tif,
                'out_cmatrix': out_cmatrix,
            })

            # increment progressor
            arcpy.SetProgressorPosition()

        except Exception as e:
            arcpy.AddError('Could not classify UAV capture. See messages.')
            arcpy.AddMessage(str(e))
            raise  # return

    # reset progressor
    arcpy.ResetProgressor()

    # endregion

    # display average variable importance

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region SHOW OVERALL VARIABLE IMPORTANCE

    arcpy.SetProgressor('default', 'Displaying overall variable importance...')

    try:
        # calculate average importance per var
        var_imps = np.mean(np.array([e['var_imps'] for e in results]), axis=0)

        # store average importance per var
        df_imps = pd.DataFrame({
            'Variable': var_names,
            'Importance': var_imps
        })

        # show average importance per var
        arcpy.AddMessage(df_imps.to_string())

        # export to classify folder
        out_imps_csv = os.path.join(classify_folder, 'rf_var_imps.csv')
        df_imps.to_csv(out_imps_csv)

    except Exception as e:
        arcpy.AddWarning('Could not build variable importances. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region SHOW OVERALL MODEL PERFORMANCE

    arcpy.SetProgressor('default', 'Displaying overall classification performance...')

    try:
        # show average overall accuraccy
        avg_oa = np.round(np.mean([_['oa'] for _ in results]), 3)
        arcpy.AddMessage(f'Mean Overall Accuracy: {str(np.round(avg_oa, 3))}.')

        # show average kappa
        avg_ka = np.round(np.mean([_['kappa'] for _ in results]), 3)
        arcpy.AddMessage(f'Mean Kappa: {str(np.round(avg_ka, 3))}.')

        # prepare average producer and user error
        avg_p_acc = np.mean([_['p_acc'] for _ in results], axis=0)
        avg_u_acc = np.mean([_['u_acc'] for _ in results], axis=0)

        # store average producer and user error
        df_result = pd.DataFrame({
            'MeanProdError': avg_p_acc.round(3),
            'MeanUserError': avg_u_acc.round(3)
        })

        # update index labels to class codes
        df_result.index = results[0]['codes']

        # show average producer and user error table and legend
        arcpy.AddMessage(df_result.to_string())
        arcpy.AddMessage('C_0: Other, C_1: Natives, C_2: Weeds')

    except Exception as e:
        arcpy.AddWarning('Could not extract accuracy metrics. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region SAVE OPTIMAL CLASSIFIED MODEL

    arcpy.SetProgressor('default', 'Saving optimal classification model...')

    # check if expected num result items exist, else error
    if len(results) != 5:
        arcpy.AddError('Number of result items is not equal to five.')
        raise  # return

    # get result item for best model based on kappa
    best_rf_model = results[np.argmax([_['kappa'] for _ in results])]

    # show user accuracy of best model
    arcpy.AddMessage(f"Best Model Iteration: {best_rf_model['cv']}.")
    arcpy.AddMessage(f"Best Overall Accuracy: {str(np.round(best_rf_model['oa'], 3))}.")
    arcpy.AddMessage(f"Best Kappa: {str(np.round(best_rf_model['kappa'], 3))}.")

    try:
        # make a copy of optimal model raster
        out_optimal_tif = os.path.join(classify_folder, 'rf_optimal.tif')
        arcpy.management.CopyRaster(in_raster=best_rf_model['out_class_tif'],
                                    out_rasterdataset=out_optimal_tif)

        # do the same for confusion matrix
        out_optimal_cmatrix = os.path.join(classify_folder, 'cmatrix_optimal.csv')
        arcpy.management.Copy(in_data=best_rf_model['out_cmatrix'],
                              out_data=out_optimal_cmatrix)

    except Exception as e:
        arcpy.AddError('Could not copy optimal classified raster/matrix. See messages.')
        arcpy.AddMessage(str(e))
        raise

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region UPDATE NEW CLASSIFICATION IN METADATA

    arcpy.SetProgressor('default', 'Updating metadata...')

    # set classified to true (this will update the main dict)
    meta_item['classified'] = True

    try:
        # write json metadata file to project folder top-level
        with open(in_project_file, 'w') as fp:
            json.dump(meta, fp)

    except Exception as e:
        arcpy.AddError('Could not write metadata. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region ADD NEW CLASSIFIED RASTER TO MAP

    arcpy.SetProgressor('default', 'Adding classified UAV capture to active map...')

    # re-enable add to map
    arcpy.env.addOutputsToMap = True

    try:
        # read current project and add tif
        aprx = arcpy.mp.ArcGISProject('CURRENT')
        mp = aprx.activeMap
        mp.addDataFromPath(out_optimal_tif)

    except Exception as e:
        arcpy.AddWarning('Could not add classified raster to active map. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # disable add to map
    arcpy.env.addOutputsToMap = False

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    try:
        # TODO: remove below if wc has no ia
        # close temp files
        del tmp_pca
        del tmp_slc
        del tmp_rsc
        #del tmp_gry  # TODO: remove if not doing glcm
        del tmp_cmp

        # TODO: uncomment below if wc has no ia
        # drop temp files (free up space)
        # arcpy.management.Delete(tmp_comp)
        arcpy.management.Delete(train_shp)
        arcpy.management.Delete(valid_shp)

        # iter each classfied result and delete
        for item in results:
            arcpy.management.Delete(item['out_ecd'])
            arcpy.management.Delete(item['out_class_tif'])
            arcpy.management.Delete(item['out_cmatrix'])

    except Exception as e:
        arcpy.AddWarning('Could not drop temporary files. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # free up spatial analyst
    arcpy.CheckInExtension('Spatial')
    arcpy.CheckInExtension('ImageAnalyst')  # TODO: remove if wc has no ia

    # set changed env variables back to default
    arcpy.env.overwriteOutput = False
    arcpy.env.addOutputsToMap = True

    # endregion

    return

# testing
#execute(None)




