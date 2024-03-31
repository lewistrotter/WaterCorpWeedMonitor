
def execute(
        parameters
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region IMPORTS

    import os
    import json
    import warnings
    import pandas as pd
    import numpy as np
    import arcpy

    from scripts import uav_classify, shared  # glcm

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region WARNINGS

    # disable warnings
    warnings.filterwarnings('ignore')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    in_project_file = parameters[0].valueAsText
    in_flight_datetime = parameters[1].value
    #in_include_prior = parameters[2].value  # this parameter is only for ui control
    in_roi_feat = parameters[3].value
    in_variables = parameters[4].valueAsText

    # inputs for testing only
    # in_project_file = r'C:\Users\Lewis\Desktop\testing2\meta.json'
    # in_flight_datetime = '2023-11-08 16:36:18'
    # #in_include_prior = False  # this parameter is only for ui control
    # in_roi_feat = r'D:\Work\Curtin\Water Corp Project - General\Testing\Tutorial Data\city_beach\rois.shp'
    # in_variables = 'NDVI;NDREI;NGRDI;OSAVI;Mean;Minimum;Maximum;StanDev;Range;Skew;Kurtosis;Entropy;Variance;CHM'

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE ENVIRONMENT

    arcpy.SetProgressor('default', 'Preparing environment...')

    # check if advanced license is available
    if arcpy.CheckProduct('ArcInfo') not in ['AlreadyInitialized', 'Available']:
        arcpy.AddError('Advanced ArcGIS Pro license is unavailable.')
        return

    # check if user has spatial/image analyst, error if not
    if arcpy.CheckExtension('Spatial') != 'Available':
        arcpy.AddError('Spatial Analyst license is unavailable.')
        return
    elif arcpy.CheckExtension('ImageAnalyst') != 'Available':
        arcpy.AddError('Image Analyst license is unavailable.')
        return
    else:
        arcpy.CheckOutExtension('Spatial')
        arcpy.CheckOutExtension('ImageAnalyst')

    # set data overwrites and mapping
    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = False

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK PROJECT FOLDER STRUCTURE AND FILES

    arcpy.SetProgressor('default', 'Checking project folders and files...')

    # check if input project file exists
    if not os.path.exists(in_project_file):
        arcpy.AddError('Project file does not exist.')
        return

    # get top-level project folder from project file
    in_project_folder = os.path.dirname(in_project_file)

    # check if required project folders already exist, error if so
    sub_folders = ['boundary', 'grid', 'uav_captures', 'sat_captures', 'visualise']
    for sub_folder in sub_folders:
        sub_folder = os.path.join(in_project_folder, sub_folder)
        if not os.path.exists(sub_folder):
            arcpy.AddError('Project is missing required folders.')
            return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CREATE AND SET WORKSPACE TO TEMPORARY FOLDER

    arcpy.SetProgressor('default', 'Preparing workspace...')

    # create temp folder if it does not already exist
    tmp = os.path.join(in_project_folder, 'tmp')
    if not os.path.exists(tmp):
        os.mkdir(tmp)

    # clear temp folder (errors skipped)
    shared.clear_tmp_folder(tmp_folder=tmp)

    # set temp folder to arcpy workspace
    arcpy.env.workspace = tmp

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
        return

    # check if any captures exist (will be >= 6), else error
    if len(meta) < 6:
        arcpy.AddError('Project has no UAV capture data.')
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT SELECTED UAV CAPTURE METADATA

    arcpy.SetProgressor('default', 'Extracting selected UAV capture metadata...')

    # exclude top-level metadata items
    exclude_keys = ['project_name', 'date_created', 'date_rehab', 'sat_shift_x', 'sat_shift_y']

    # extract selected metadata item based on capture date
    meta_item = None
    for k, v in meta.items():
        if k not in exclude_keys:
            if v['capture_date'] == in_flight_datetime:
                meta_item = v

    # check if meta item exists, else error
    if meta_item is None:
        arcpy.AddError('Could not find selected UAV capture in metadata file.')
        return

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
            return

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
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT SELECTED UAV CAPTURE METADATA

    arcpy.SetProgressor('default', 'Preparing requested variables...')

    # unpack variables
    variables = in_variables.split(';')

    # check if we have something, else error
    if len(variables) == 0:
        arcpy.AddError('No variables selected. Must select at least one.')

    # prepare lowercase
    variables = [v.lower() for v in variables]

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE VEGETATION INDICES

    arcpy.SetProgressor('default', 'Calculating vegetation indices...')

    # create vegetation index map to maintain band order
    tmp_map = {
        'ndvi': os.path.join(bands_folder, 'vi_ndvi.tif'),
        'ndrei': os.path.join(bands_folder, 'vi_ndrei.tif'),
        'ngrdi': os.path.join(bands_folder, 'vi_ngrdi.tif'),
        'rgbvi': os.path.join(bands_folder, 'vi_rgbvi.tif'),
        'osavi': os.path.join(bands_folder, 'vi_osavi.tif'),
    }

    # remove unwanted variables based on user selection
    vi_map = {}
    for k, v in tmp_map.items():
        if k in variables:
            vi_map[k] = v

    # set up step-wise progressor
    arcpy.SetProgressor('step', None, 0, len(vi_map))

    if len(vi_map) > 0:
        try:
            # iter each vi and calculate
            for k, v in vi_map.items():
                if k == 'ndvi':
                    ras = (nir - red) / (nir + red)
                elif k == 'ndrei':
                    ras = (nir - redge) / (nir + redge)
                elif k == 'ngrdi':
                    ras = (green - red) / (green + red)
                elif k == 'rgbvi':
                    ras = (green ** 2 - (red * blue)) / (green ** 2 + (red * blue))
                elif k == 'osavi':
                    ras = (1 + 0.16) * ((nir - red) / (nir + red + 0.16))

                # save to associated path
                ras.save(v)

                # notify user
                arcpy.AddMessage(f'Vegetation index {k} done.')

                # increment progressor
                arcpy.SetProgressorPosition()

        except Exception as e:
            arcpy.AddError('Could not calculate vegetation indices. See messages.')
            arcpy.AddMessage(str(e))
            return

    # reset progressor
    arcpy.ResetProgressor()

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region GENERATE GRAYSCALE PCA

    arcpy.SetProgressor('default', 'Generating grayscale image...')

    try:
        # create band path list
        band_list = list(band_map.values())

        # generate pca, grayscale and normalise 0-255
        tmp_gry = uav_classify.make_grayscale(in_band_list=band_list,
                                              levels=16)

        # save grayscale to tif
        tmp_gry.save('tmp_gry.tif')

    except Exception as e:
        arcpy.AddError('Could not calculate grayscale. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE GLCM TEXTURES

    arcpy.SetProgressor('default', 'Calculating textures...')

    # create texture indices map to maintain band order
    tmp_map = {
        'mean': os.path.join(bands_folder, 'tx_mean.tif'),
        'minimum': os.path.join(bands_folder, 'tx_min.tif'),
        'maximum': os.path.join(bands_folder, 'tx_max.tif'),
        'standev': os.path.join(bands_folder, 'tx_standev.tif'),
        'range': os.path.join(bands_folder, 'tx_range.tif'),
        #'contrast': os.path.join(bands_folder, 'tx_contrast.tif'),
        #'dissimilarity': os.path.join(bands_folder, 'tx_dissimilarity.tif'),
        'skew': os.path.join(bands_folder, 'tx_skew.tif'),
        'kurtosis': os.path.join(bands_folder, 'tx_kurtosis.tif'),
        'entropy': os.path.join(bands_folder, 'tx_entropy.tif'),
        #'homogeneity': os.path.join(bands_folder, 'tx_homogeneity.tif'),
        #'secondmoment': os.path.join(bands_folder, 'tx_secondmoment.tif'),
        'variance': os.path.join(bands_folder, 'tx_variance.tif')
    }

    # remove unwanted variables based on user selection
    tx_map = {}
    for k, v in tmp_map.items():
        if k in variables:
            tx_map[k] = v

    # set up step-wise progressor
    arcpy.SetProgressor('step', None, 0, len(tx_map))

    if len(tx_map) > 0:
        try:
            # setup neighbourhood object
            win = arcpy.sa.NbrRectangle(5, 5, 'CELL')

            # calculate glcm textures... disabled, were not worth processing time
            #glcm.quick_glcm(in_gray_ras='tmp_gry.tif',
                            #textures_map=tx_map,
                            #levels=18,
                            #kernel_size=5,
                            #v_block_size=50)

            # iter each texture and calculate
            for k, v in tx_map.items():
                if k == 'mean':
                    ras = arcpy.sa.FocalStatistics(in_raster=tmp_gry,
                                                   neighborhood=win,
                                                   statistics_type='MEAN')
                elif k == 'minimum':
                    ras = arcpy.sa.FocalStatistics(in_raster=tmp_gry,
                                                   neighborhood=win,
                                                   statistics_type='MINIMUM')
                elif k == 'maximum':
                    ras = arcpy.sa.FocalStatistics(in_raster=tmp_gry,
                                                   neighborhood=win,
                                                   statistics_type='MAXIMUM')
                elif k == 'standev':
                    ras = arcpy.sa.FocalStatistics(in_raster=tmp_gry,
                                                   neighborhood=win,
                                                   statistics_type='STD')
                elif k == 'range':
                    ras = arcpy.sa.FocalStatistics(in_raster=tmp_gry,
                                                   neighborhood=win,
                                                   statistics_type='RANGE')
                elif k == 'skew':
                    ras = uav_classify.calc_skew(in_gray_ras='tmp_gry.tif',
                                                 win_size=5)
                elif k == 'kurtosis':
                    ras = uav_classify.calc_kurtosis(in_gray_ras='tmp_gry.tif',
                                                     win_size=5)
                elif k == 'entropy':
                    ras = uav_classify.calc_entropy(in_gray_ras='tmp_gry.tif',
                                                    win_size=5)
                elif k == 'variance':
                    ras = uav_classify.calc_variance(in_gray_ras='tmp_gry.tif',
                                                     win_size=5)

                # save to associated path
                ras.save(v)

                # notify user
                arcpy.AddMessage(f'Texture index {k} done.')

                # increment progressor
                arcpy.SetProgressorPosition()

        except Exception as e:
            arcpy.AddError('Could not calculate textures. See messages.')
            arcpy.AddMessage(str(e))
            return

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

    # remove unwanted variables based on user selection
    for k in chm_map:
        if k not in variables:
            del chm_map[k]

    # create dsm and dtm file paths
    dsm_path = os.path.join(bands_folder, 'dsm.tif')
    dtm_path = os.path.join(bands_folder, 'dtm.tif')

    # check if both exist
    for v in [dsm_path, dtm_path]:
        if not os.path.exists(v):
            arcpy.AddError('Digital terrain/surface elevation bands missing from capture folder.')
            return

    if len(chm_map) > 0:
        try:
            # read dsm and dtm rasters
            dsm = arcpy.Raster(dsm_path)
            dtm = arcpy.Raster(dtm_path)

            # calculate chm via dsm - dtm
            ras = dsm - dtm

            # save to associated path
            ras.save(os.path.join(bands_folder, chm_map['chm']))

            # notify user
            arcpy.AddMessage('Canopy Height Model done.')

        except Exception as e:
            arcpy.AddError('Could not calculate canopy height model. See messages.')
            arcpy.AddMessage(str(e))
            return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region VALIDATE REGIONS OF INTEREST FEATURES

    arcpy.SetProgressor('default', 'Validating field training features...')

    # project roi shapefile to albers
    arcpy.management.Project(in_dataset=in_roi_feat,
                             out_dataset='tmp_roi_prj.shp',
                             out_coor_system=arcpy.SpatialReference(3577))

    try:
        # validate features, will throw exception if problem
        uav_classify.validate_rois(in_roi_feat='tmp_roi_prj.shp',
                                   extent=blue.extent)

    except Exception as e:
        arcpy.AddError('Training features are invalid. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CLASSIFY UAV CAPTURE

    arcpy.SetProgressor('default', 'Classifying UAV capture...')

    # build variable name and path lists (we use former later)
    var_names = list(band_map.keys()) + list(vi_map.keys()) + list(tx_map.keys()) + list(chm_map.keys())
    var_paths = list(band_map.values()) + list(vi_map.values()) + list(tx_map.values()) + list(chm_map.values())

    try:
        # composite all variables
        tmp_cmp = arcpy.sa.CompositeBand(var_paths)

    except Exception as e:
        arcpy.AddError('Could not create composite of UAV bands. See messages.')
        arcpy.AddMessage(str(e))
        return

    # set up step-wise progressor
    arcpy.SetProgressor('step', None, 0, 5)

    # iter five times...
    results = []
    for i in range(1, 6):
        try:
            # notify user
            arcpy.AddMessage(f'Classifying UAV capture ({i} / 5)...')

            # split rois into train / valid splits (50% / 50%)
            train_shp, valid_shp = uav_classify.split_rois(in_roi_feat='tmp_roi_prj.shp',
                                                           out_folder=tmp,
                                                           pct_split=0.5,
                                                           equal_classes=False)

            # classify the uav capture via rf, need full path for non-arcpy
            out_ecd = os.path.join(tmp, f'ecd_{i}.ecd')
            out_class_tif = os.path.join(tmp, f'rf_{i}.tif')
            out_class_tif = uav_classify.classify(in_train_roi=train_shp,
                                                  in_comp_ras=tmp_cmp,
                                                  out_ecd=out_ecd,
                                                  out_class_tif=out_class_tif,
                                                  num_trees=500,
                                                  num_depth=100)

            # extract variable importance list
            var_imps = uav_classify.extract_var_importance(in_ecd=out_ecd)

            # validate the classification via confuse matrix, need full path for non-arcpy
            out_cmatrix = os.path.join(tmp, f'cmatrix_{i}.csv')
            codes, p_acc, u_acc, oa, kappa = uav_classify.assess_accuracy(out_class_tif=out_class_tif,
                                                                          in_valid_shp=valid_shp,
                                                                          out_cmatrix=out_cmatrix)

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
            return

    # reset progressor
    arcpy.ResetProgressor()

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region SHOW OVERALL VARIABLE IMPORTANCE

    arcpy.SetProgressor('default', 'Displaying overall variable importance...')

    # create classify folder if not already
    classify_folder = os.path.join(capture_folder, 'classify')
    if not os.path.exists(classify_folder):
        os.mkdir(classify_folder)

    try:
        # calculate average importance per var
        var_imps = np.mean(np.array([v['var_imps'] for v in results]), axis=0)

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
        arcpy.AddError('Could not get overall variable importance. See messages.')
        arcpy.AddMessage(str(e))
        return

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
            'MeanProdAcc': avg_p_acc.round(3),
            'MeanUserAcc': avg_u_acc.round(3)
        })

        # update index labels to class codes
        df_result.index = results[0]['codes']

        # show average producer and user error table and legend
        arcpy.AddMessage(df_result.to_string())
        arcpy.AddMessage('C_0: Other, C_1: Natives, C_2: Weeds')

    except Exception as e:
        arcpy.AddError('Could not extract accuracy metrics. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region SAVE OPTIMAL CLASSIFIED MODEL

    arcpy.SetProgressor('default', 'Saving optimal classification model...')

    # check if expected num result items exist, else error
    if len(results) != 5:
        arcpy.AddError('Number of result items is not equal to five.')
        return

    try:
        # extract classed tif paths
        classed_tifs = [r['out_class_tif'] for r in results]

        # get majority classes per pixel and save it
        tmp_cls = arcpy.sa.CellStatistics(classed_tifs, 'MAJORITY')
        tmp_cls.save('tmp_cls.tif')

        # apply a 5x5 majority focal window filter to remove noise and save
        win = arcpy.sa.NbrRectangle(5, 5, 'CELL')
        tmp_maj = arcpy.sa.FocalStatistics('tmp_cls.tif', win, 'MAJORITY')
        tmp_maj.save('tmp_maj.tif')

        # copy optimal model raster to classify folder
        out_optimal_tif = os.path.join(classify_folder, 'rf_optimal.tif')
        arcpy.management.CopyRaster(in_raster='tmp_maj.tif',
                                    out_rasterdataset=out_optimal_tif)

        # rebuild attribute table class information after stripping above
        uav_classify.rebuild_raster_class_attributes(in_ras=out_optimal_tif)

        # get result item for best model based on kappa
        best_rf_model = results[np.argmax([_['kappa'] for _ in results])]

        # get related confusion matrix
        out_optimal_cmatrix = os.path.join(classify_folder, 'cmatrix_optimal.csv')
        arcpy.management.Copy(in_data=best_rf_model['out_cmatrix'],
                              out_data=out_optimal_cmatrix)

    except Exception as e:
        arcpy.AddError('Could not save optimal classified raster/matrix. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region UPDATE NEW CLASSIFICATION IN METADATA

    arcpy.SetProgressor('default', 'Updating metadata...')

    # set classified to true (this will update the main dict)
    meta_item['classified'] = True

    # reset fractional data as incase this was a reclssification
    meta_item['fractions'] = []

    try:
        # write json metadata file to project folder top-level
        with open(in_project_file, 'w') as fp:
            json.dump(meta, fp)

    except Exception as e:
        arcpy.AddError('Could not write metadata. See messages.')
        arcpy.AddMessage(str(e))
        return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region ADD CLASSIFIED RASTER TO ACTIVE MAP

    arcpy.SetProgressor('default', 'Visualising result...')

    # build visualise folder path and
    visualise_folder = os.path.join(in_project_folder, 'visualise')

    try:
        # create uav classified raster for visualise folder
        tmp_cls = arcpy.Raster(out_optimal_tif)

        # get flight date code
        flight_date = meta_item['capture_folder']

        # create uav rgb classified path to visualise folder
        out_tif = os.path.join(visualise_folder, 'uav_classified' + '_' + flight_date + '.tif')

        # delete previously created visual raster and re-save
        shared.delete_visual_rasters(rasters=[out_tif])
        tmp_cls.save(out_tif)

        # visualise it on active map and symbolise it to class colors
        shared.add_raster_to_map(in_ras=out_tif)
        shared.apply_classified_symbology(in_ras=out_tif)

    except Exception as e:
        arcpy.AddWarning('Could not visualise classified image. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    arcpy.SetProgressor('default', 'Cleaning up environment...')

    try:
        # close temp files
        del blue
        del green
        del red
        del redge
        del nir
        del ras
        del dsm
        del dtm
        del tmp_gry
        del tmp_cmp
        del tmp_cls
        del tmp_maj

    except Exception as e:
        arcpy.AddWarning('Could not drop temporary files. See messages.')
        arcpy.AddMessage(str(e))
        pass

    # clear temp folder (errors skipped)
    shared.clear_tmp_folder(tmp_folder=tmp)

    # free up spatial analyst and image analyst
    arcpy.CheckInExtension('Spatial')
    arcpy.CheckInExtension('ImageAnalyst')

    # set changed env variables back to default
    arcpy.env.overwriteOutput = False
    arcpy.env.addOutputsToMap = True

    # endregion

    return

# testing
# execute(None)
