
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

    in_project_file = parameters[0].valueAsText
    in_flight_datetime = parameters[1].value
    in_roi_feat = parameters[2].value

    # TODO: uncomment these when testing
    # in_project_file = r'C:\Users\Lewis\Desktop\testing\project_1\meta.json'
    # in_flight_datetime = '2023-05-05 13:28:03'
    # in_roi_feat = r'D:\Work\Curtin\Water Corp Project - General\Processed\City Beach\Classification\Final\train_test_rois_smaller_bc_grp_nvwvo_wgs_z50s.shp'

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region PREPARE ENVIRONMENT

    arcpy.SetProgressor('default', 'Preparing environment...')

    if arcpy.CheckExtension('Spatial') != 'Available':
        arcpy.AddError('Spatial Analyst license is unavailable.')
        raise  # return
    else:
        arcpy.CheckOutExtension('Spatial')

    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = False

    # TODO: enable if move to non-memory temp files
    # TODO: ensure all the input/out pathes changed too if do this
    #tmp_folder = os.path.join(in_project_file, 'tmp')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK PROJECT FOLDER STRUCTURE

    arcpy.SetProgressor('default', 'Checking project folders...')

    if not os.path.exists(in_project_file):
        arcpy.AddError('Project file does not exist.')
        raise  # return

    in_project_folder = os.path.dirname(in_project_file)

    sub_folders = ['grid', 'uav_captures', 'sat_captures', 'tmp', 'visualise']
    for sub_folder in sub_folders:
        sub_folder = os.path.join(in_project_folder, sub_folder)
        if not os.path.exists(sub_folder):
            arcpy.AddError('Project folder is missing expected folders.')
            raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ AND CHECK METADATA

    arcpy.SetProgressor('default', 'Reading and checking metadata...')

    with open(in_project_file, 'r') as fp:
        meta = json.load(fp)

    if len(meta['data']) == 0:
        arcpy.AddError('Project capture data does not exist.')
        raise  # return

    meta_item = None
    for item in meta['data']:
        if in_flight_datetime == item['capture_date']:
            meta_item = item

    if meta_item is None:
        arcpy.AddError('Could not find selected capture in metadata file.')
        raise  # return

    # TODO: other checks

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ CAPTURE RASTER BANDS

    arcpy.SetProgressor('default', 'Reading capture raster bands...')

    capture_folder = os.path.join(in_project_folder, 'uav_captures', meta_item['capture_folder'])
    bands_folder = os.path.join(capture_folder, 'bands')

    band_map = {
        'blue': os.path.join(bands_folder, 'blue.tif'),
        'green': os.path.join(bands_folder, 'green.tif'),
        'red': os.path.join(bands_folder, 'red.tif'),
        'redge': os.path.join(bands_folder, 'redge.tif'),
        'nir': os.path.join(bands_folder, 'nir.tif'),
    }

    for k, v in band_map.items():
        if not os.path.exists(v):
            arcpy.AddError('Could not find all required capture bands.')
            raise  # return

    try:
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

    vi_map = {
        'ndvi': os.path.join(bands_folder, 'vi_ndvi.tif'),
        'endvi': os.path.join(bands_folder, 'vi_endvi.tif'),
        'ndrei': os.path.join(bands_folder, 'vi_ndrei.tif'),
        'ngrdi': os.path.join(bands_folder, 'vi_ngrdi.tif'),
        'rgbvi': os.path.join(bands_folder, 'vi_rgbvi.tif'),
        'osavi': os.path.join(bands_folder, 'vi_osavi.tif'),
        'kndvi': os.path.join(bands_folder, 'vi_kndvi.tif'),
    }

    try:
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
                raise ValueError('Vegetation index does not exist.')

            ras.save(v)

    except Exception as e:
        arcpy.AddError('Could not read calculate vegetation indices. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE GLCM TEXTURES

    arcpy.SetProgressor('default', 'Calculating GLCM textures...')

    # generate pca and return first principal band only
    tmp_pca = r'memory\tmp_pca'  # os.path.join(tmp_folder, 'tmp_pca.tif')
    out_raster = arcpy.sa.PrincipalComponents(in_raster_bands=list(band_map.values()),
                                              number_components=1)
    out_raster.save(tmp_pca)

    # convert pca band 1 to 0-255 grayscale
    tmp_rescale = os.path.join(bands_folder, 'pca.tif')
    out_raster = arcpy.sa.RescaleByFunction(in_raster=tmp_pca,
                                            transformation_function="LINEAR",
                                            from_scale=0,
                                            to_scale=255)
    out_raster.save(tmp_rescale)

    tx_map = {
        'mean': os.path.join(bands_folder, 'tx_mean.tif'),
        'contrast': os.path.join(bands_folder, 'tx_contrast.tif'),
        'correlation': os.path.join(bands_folder, 'tx_correlation.tif'),
        'dissimilarity': os.path.join(bands_folder, 'tx_dissimilarity.tif'),
        'entropy': os.path.join(bands_folder, 'tx_entropy.tif'),
        'homogeneity': os.path.join(bands_folder, 'tx_homogeneity.tif'),
        'secmoment': os.path.join(bands_folder, 'tx_secmoment.tif'),
        'variance': os.path.join(bands_folder, 'tx_variance.tif'),
    }

    # TODO: calculate glcm textures
    ####

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CALCULATE CANOPY HEIGHT MODEL (CHM)

    arcpy.SetProgressor('default', 'Calculating canopy height...')

    chm_map = {
        'dsm': os.path.join(bands_folder, 'dsm.tif'),
        'dtm': os.path.join(bands_folder, 'dtm.tif'),
    }

    for k, v in chm_map.items():
        if not os.path.exists(v):
            arcpy.AddError('Could not find surface/terrain model bands.')
            raise  # return

    try:
        dsm = arcpy.Raster(chm_map['dsm'])
        dtm = arcpy.Raster(chm_map['dtm'])

    except Exception as e:
        arcpy.AddError('Could not read terrain/surface elevation bands. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    try:
        chm = dsm - dtm
        chm.save(os.path.join(bands_folder, 'chm.tif'))
        chm_map.update({'chm': os.path.join(bands_folder, 'chm.tif')})

    except Exception as e:
        arcpy.AddError('Could not calculate canopy height. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region READ REGIONS OF INTEREST FEATURE

    arcpy.SetProgressor('default', 'Reading region of interest features...')

    try:
        fc_desc = arcpy.Describe(in_roi_feat)
        fc_srs = fc_desc.spatialReference
        fields = [f.name for f in arcpy.ListFields(in_roi_feat)]

    except Exception as e:
        arcpy.AddError('Cannot read featureclass. See messages.')
        arcpy.AddMessage(str(e))
        raise  # return

    if fc_srs == 'Unknown' or fc_srs.factoryCode != 32750:
        arcpy.AddError('Featureclass must be projected in WGS84 UTM Zone 50S (32750).')
        raise  # return

    req_fields = ['Classname', 'Classvalue']
    for required_field in req_fields:
        if required_field not in fields:
            arcpy.AddError(f'Field {req_fields} missing from region of interest feature.')

    cnames, cvalues, geoms = [], [], []
    with arcpy.da.SearchCursor(in_roi_feat, ['Classname', 'Classvalue', 'Shape@']) as cursor:
        for row in cursor:
            cnames.append(row[0])
            cvalues.append(row[1])
            geoms.append(row[2])

    unq_cn, cnt_cn = np.unique(cnames, return_counts=True)
    unq_cv, cnt_cv = np.unique(cvalues, return_counts=True)

    if len(unq_cn) != 3 or np.any(cnt_cn < 20) or len(unq_cv) != 3 or np.any(cnt_cv < 20):
        arcpy.AddError('Regions of interest must have 3 classes (Native, Weed, Other) each with >= 20 polygons.')
        raise  # return

    num_rois_in = 0
    for geom in geoms:
        if blue.extent.contains(geom):
            num_rois_in += 1

    if num_rois_in < 30:
        arcpy.AddError('Not enough region of interest areas within UAV image extent.')
        raise  # return

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CLASSIFY UAV CAPTURE

    arcpy.SetProgressor('default', 'Classifying UAV capture...')

    classify_folder = os.path.join(capture_folder, 'classify')
    if not os.path.exists(classify_folder):
        os.mkdir(classify_folder)

    variables = list(band_map.values()) + list(vi_map.values()) + list(tx_map.values())  # TODO: implement chm?

    with arcpy.EnvManager(pyramid='NONE'):
        tmp_comp = os.path.join(classify_folder, 'comp')  # dont use tif, arcpy bug
        arcpy.management.CompositeBands(in_rasters=variables,
                                        out_raster=tmp_comp)

    results = []
    for i in range(1, 6):

        arcpy.AddMessage(f'Classifying UAV capture ({i} / 5)...')

        # split rois into train / valid splits (50% / 50%)
        train_shp, valid_shp = shared.split_rois(in_roi_feat=in_roi_feat,
                                                 out_folder=classify_folder,
                                                 pct_split=0.5,
                                                 equal_classes=False)

        # classify the uav capture via random forest
        out_ecd = os.path.join(classify_folder, f'ecd_{i}.ecd')
        out_class_tif = os.path.join(classify_folder, f'rf_{i}.tif')
        out_class_tif = shared.classify(in_train_roi=train_shp,
                                        in_comp_tif=tmp_comp,
                                        out_ecd=out_ecd,
                                        out_class_tif=out_class_tif,
                                        num_trees=100,  # TODO: let user set
                                        num_depth=30)   # TODO: let user set

        # validate the classification via confuse matrix
        out_cmatrix = os.path.join(classify_folder, f'cmatrix_{i}.csv')
        codes, p_acc, u_acc, oa, kappa = shared.assess_accuracy(out_class_tif=out_class_tif,
                                                                in_valid_shp=valid_shp,
                                                                out_cmatrix=out_cmatrix)

        arcpy.AddMessage(f'> Overall Accuracy: {str(np.round(oa, 3))}.')
        arcpy.AddMessage(f'> Kappa: {str(np.round(kappa, 3))}.')
        # TODO: p/u acc?

        results.append({
            'cv': i,
            'codes': codes,
            'p_acc': p_acc,
            'u_acc': u_acc,
            'oa': oa,
            'kappa': kappa,
            'out_ecd': out_ecd,
            'out_class_tif': out_class_tif,
            'out_cmatrix': out_cmatrix,
        })

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region SHOW OVERALL MODEL PERFORMANCE

    arcpy.SetProgressor('default', 'Displaying overall classification performance...')

    avg_oa = np.round(np.mean([_['oa'] for _ in results]), 3)
    arcpy.AddMessage(f'Mean Overall Accuracy: {str(np.round(avg_oa, 3))}.')

    avg_ka = np.round(np.mean([_['kappa'] for _ in results]), 3)
    arcpy.AddMessage(f'Mean Kappa: {str(np.round(avg_ka, 3))}.')

    avg_p_acc = np.mean([_['p_acc'] for _ in results], axis=0)
    avg_u_acc = np.mean([_['u_acc'] for _ in results], axis=0)

    df_result = pd.DataFrame({
        'MeanProdError': avg_p_acc.round(3),
        'MeanUserError': avg_u_acc.round(3)
    })
    # TODO: display p/u error?

    # update index labels to class codes
    df_result.index = results[0]['codes']

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region SAVE OPTIMAL CLASSIFIED MODEL

    arcpy.SetProgressor('default', 'Saving optimal classification model...')

    best_rf_model = results[np.argmax([_['kappa'] for _ in results])]

    arcpy.AddMessage(f"Best Model Iteration: {best_rf_model['cv']}.")
    arcpy.AddMessage(f"Best Overall Accuracy: {str(np.round(best_rf_model['oa'], 3))}.")
    arcpy.AddMessage(f"Best Kappa: {str(np.round(best_rf_model['kappa'], 3))}.")

    out_datetime = meta_item['capture_folder'].replace('uav_', '')

    out_optimal_tif = os.path.join(classify_folder, f'rf_optimal_{out_datetime}.tif')
    arcpy.management.CopyRaster(in_raster=best_rf_model['out_class_tif'],
                                out_rasterdataset=out_optimal_tif)

    out_optimal_cmatrix = os.path.join(classify_folder, f'cmatrix_optimal_{out_datetime}.csv')
    arcpy.management.Copy(in_data=best_rf_model['out_cmatrix'],
                          out_data=out_optimal_cmatrix)

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region UPDATE NEW CLASSIFICATION TO METADATA

    arcpy.SetProgressor('default', 'Updating metadata...')

    meta_item['classified'] = True  # this will update the meta dict iterable too
    with open(in_project_file, 'w') as fp:
        json.dump(meta, fp)

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region ADD NEW CLASSIFIED RASTER TO MAP

    arcpy.SetProgressor('default', 'Adding classified UAV capture to active map...')

    arcpy.env.addOutputsToMap = True

    try:
        aprx = arcpy.mp.ArcGISProject('CURRENT')
        mp = aprx.activeMap
        mp.addDataFromPath(out_optimal_tif)

    except Exception as e:
        arcpy.AddWarning('Could not add classified raster to active map. See messages.')
        arcpy.AddMessage(str(e))

    arcpy.env.addOutputsToMap = False

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region END ENVIRONMENT

    try:
        arcpy.management.Delete(tmp_comp)
        arcpy.management.Delete(train_shp)
        arcpy.management.Delete(valid_shp)

        for item in results:
            arcpy.management.Delete(item['out_ecd'])
            arcpy.management.Delete(item['out_class_tif'])
            arcpy.management.Delete(item['out_cmatrix'])

    except Exception as e:
        arcpy.AddWarning('Could not drop temporary files. See messages.')
        arcpy.AddMessage(str(e))

    arcpy.CheckInExtension('Spatial')

    arcpy.env.overwriteOutput = False
    arcpy.env.addOutputsToMap = True

    # endregion

    return

# testing
#execute(None)




