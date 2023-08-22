# old fractional mapping geoprocessor with train/predict regress approach
# def execute(
#         parameters
#         # messages # TODO: implement
# ):
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region IMPORTS
#
#     import os
#     import json
#     import numpy as np
#     import xarray as xr
#     import arcpy
#
#     from scripts import web, uav_fractions, shared
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region EXTRACT PARAMETERS
#
#     # inputs from arcgis pro ui
#     # in_project_file = parameters[0].valueAsText
#     # in_flight_datetime = parameters[1].value
#
#     # inputs for testing only
#     in_project_file = r'C:\Users\Lewis\Desktop\testing\city beach dev\meta.json'
#     in_flight_datetime = '2023-06-09 13:03:39'
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region PREPARE ENVIRONMENT
#
#     arcpy.SetProgressor('default', 'Preparing environment...')
#
#     # check if user has spatial analyst, error if not
#     if arcpy.CheckExtension('Spatial') != 'Available':
#         arcpy.AddError('Spatial Analyst license is unavailable.')
#         return
#     elif arcpy.CheckExtension('ImageAnalyst') != 'Available':
#         arcpy.AddError('Image Analyst license is unavailable.')
#         return
#     else:
#         arcpy.CheckOutExtension('Spatial')
#         arcpy.CheckOutExtension('ImageAnalyst')
#
#     # set data overwrites and mapping
#     arcpy.env.overwriteOutput = True
#     arcpy.env.addOutputsToMap = False
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region CHECK PROJECT FOLDER STRUCTURE AND FILES
#
#     arcpy.SetProgressor('default', 'Checking project folders and files...')
#
#     # check if input project file exists
#     if not os.path.exists(in_project_file):
#         arcpy.AddError('Project file does not exist.')
#         return
#
#     # get top-level project folder from project file
#     in_project_folder = os.path.dirname(in_project_file)
#
#     # check if required project folders already exist, error if so
#     sub_folders = ['grid', 'uav_captures', 'sat_captures', 'visualise']
#     for sub_folder in sub_folders:
#         sub_folder = os.path.join(in_project_folder, sub_folder)
#         if not os.path.exists(sub_folder):
#             arcpy.AddError('Project folder is missing expected folders.')
#             return
#
#     # check if uav grid file exists
#     grid_tif = os.path.join(in_project_folder, 'grid', 'grid.tif')
#     if not os.path.exists(grid_tif):
#         arcpy.AddError('UAV grid raster does not exist.')
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region READ AND CHECK METADATA
#
#     arcpy.SetProgressor('default', 'Reading and checking metadata...')
#
#     try:
#         # read project json file
#         with open(in_project_file, 'r') as fp:
#             meta = json.load(fp)
#
#     except Exception as e:
#         arcpy.AddError('Could not read metadata. See messages.')
#         arcpy.AddMessage(str(e))
#         return
#
#     # check if any captures exist (will be >= 4)
#     if len(meta) < 4:
#         arcpy.AddError('Project has no UAV capture data.')
#         return
#
#     # check and get start of rehab date
#     rehab_start_date = meta.get('date_rehab')
#     if rehab_start_date is None:
#         arcpy.AddError('Project has no start of rehab date.')
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region EXTRACT SELECTED UAV CAPTURE METADATA
#
#     arcpy.SetProgressor('default', 'Extracting selected UAV capture metadata...')
#
#     # exclude top-level metadata items
#     exclude_keys = ['project_name', 'date_created', 'date_rehab']
#
#     # extract selected metadata item based on capture date
#     meta_item = None
#     for k, v in meta.items():
#         if k not in exclude_keys:
#             if v['capture_date'] == in_flight_datetime:
#                 meta_item = v
#
#     # check if meta item exists, else error
#     if meta_item is None:
#         arcpy.AddError('Could not find selected UAV capture in metadata file.')
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region FETCH CLEAN DEA STAC SENTINEL 2 DOWNLOADS
#
#     arcpy.SetProgressor('default', 'Fetching clean DEA STAC downloads...')
#
#     # set sat folder for nc outputs
#     sat_folder = os.path.join(in_project_folder, 'sat_captures')
#
#     try:
#         # set query date range, collections and assets
#         start_date, end_date = '2016-01-01', '2039-12-31'
#
#         # read grid as raster
#         tmp_grd = arcpy.Raster(grid_tif)
#
#         # get stac and output coordinate bbox based on grid exent
#         stac_bbox = shared.get_raster_bbox(in_ras=tmp_grd, out_epsg=4326)
#
#         # get output netcdf bbox in albers and expand
#         out_bbox = shared.get_raster_bbox(in_ras=tmp_grd, out_epsg=3577)
#         out_bbox = shared.expand_bbox(bbox=out_bbox, by_metres=30.0)
#
#         # set output folder for raw sentinel 2 cubes and check
#         raw_ncs_folder = os.path.join(sat_folder, 'raw_ncs')
#         if not os.path.exists(raw_ncs_folder):
#             os.mkdir(raw_ncs_folder)
#
#         # query and prepare downloads
#         downloads = web.quick_fetch(start_date=start_date,
#                                     end_date=end_date,
#                                     stac_bbox=stac_bbox,
#                                     out_bbox=out_bbox,
#                                     out_folder=raw_ncs_folder)
#
#     except Exception as e:
#         arcpy.AddError('Unable to download Sentinel 2 data from DEA. See messages.')
#         arcpy.AddMessage(str(e))
#         return
#
#     # check if downloads returned, else leave
#     if len(downloads) == 0:
#         arcpy.AddWarning('No valid satellite downloads were found.')
#         return  # TODO: carry on in case fractionals remain unprocessed?
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region DOWNLOAD WCS DATA
#
#     arcpy.SetProgressor('default', 'Downloading Sentinel 2 data...')
#
#     try:
#         # download everything and return success or fail statuses
#         results = web.quick_download(downloads=downloads,
#                                      quality_flags=[1],
#                                      max_out_of_bounds=1,
#                                      max_invalid_pixels=1,
#                                      nodata_value=-999)
#
#     except Exception as e:
#         arcpy.AddError('Unable to download Sentinel 2 data from DEA. See messages.')
#         arcpy.AddMessage(str(e))
#         return
#
#     # check if any valid downloads (non-cloud or new)
#     num_valid_downlaods = len([dl for dl in results if 'success' in dl])
#     if num_valid_downlaods == 0:
#         arcpy.AddMessage('No new valid satellite downloads were found.')
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region COMBINE SENTINEL 2 NETCDFS
#
#     arcpy.SetProgressor('default', 'Combining Sentinel 2 data...')
#
#     # get all raw nc dates in raw netcdf folder
#     nc_files = []
#     for file in os.listdir(raw_ncs_folder):
#         if file.startswith('R') and file.endswith('.nc'):
#             nc_files.append(os.path.join(raw_ncs_folder, file))
#
#     # check if anything came back, error if not
#     if len(nc_files) == 0:
#         arcpy.AddError('No NetCDF files were found.')
#         return
#
#     try:
#         # read all netcdfs into single dataset
#         ds = shared.combine_netcdf_files(nc_files=nc_files)
#
#     except Exception as e:
#         arcpy.AddError('Unable to combine Sentinel 2 NetCDFs. See messages.')
#         arcpy.AddMessage(str(e))
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region CLEAN SENTINEL 2 NETCDFS
#
#     arcpy.SetProgressor('default', 'Cleaning Sentinel 2 data...')
#
#     try:
#         # extract netcdf attributes
#         ds_attrs = ds.attrs
#         ds_band_attrs = ds[list(ds)[0]].attrs
#         ds_spatial_ref_attrs = ds['spatial_ref'].attrs
#
#         # set nodata (-999) to nan
#         ds = ds.where(ds != -999)
#
#         # set pixel to nan when any band has outlier within pval 0.001 per date
#         # TODO: just double check this func isnt setting all pixels nan when outlier
#         ds = uav_fractions.remove_xr_outliers(ds=ds,
#                                               max_z_value=3.29)
#
#         # resample to monthly medians and interpolate nan
#         ds = uav_fractions.resample_xr_monthly_medians(ds=ds)
#
#         # TODO: back and forward fill
#         #
#
#         # append attributes back on
#         ds.attrs = ds_attrs
#         ds['spatial_ref'].attrs = ds_spatial_ref_attrs
#         for var in ds:
#             ds[var].attrs = ds_band_attrs
#
#     except Exception as e:
#         arcpy.AddError('Unable to clean Sentinel 2 NetCDFs. See messages.')
#         arcpy.AddMessage(str(e))
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region EXPORT CLEAN SENTINEL 2 NETCDF
#
#     arcpy.SetProgressor('default', 'Exporting clean Sentinel 2 data...')
#
#     # set combined output nc folder
#     cmb_ncs_folder = os.path.join(sat_folder, 'cmb_ncs')
#     if not os.path.exists(cmb_ncs_folder):
#         os.mkdir(cmb_ncs_folder)
#
#     try:
#         # export combined monthly median as new netcdf
#         out_montly_meds_nc = os.path.join(cmb_ncs_folder, 'raw_monthly_meds.nc')
#         ds.to_netcdf(out_montly_meds_nc)
#         ds.close()
#
#     except Exception as e:
#         arcpy.AddError('Unable to export clean Sentinel 2 NetCDF. See messages.')
#         arcpy.AddMessage(str(e))
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region READ HIGH RES CLASSIFIED DRONE IMAGE AS XR DATASET
#
#     arcpy.SetProgressor('default', 'Reading high-resolution classified UAV data...')
#
#     # set up capture and classify folders
#     capture_folders = os.path.join(in_project_folder, 'uav_captures')
#     capture_folder = os.path.join(capture_folders, meta_item['capture_folder'])
#     classify_folder = os.path.join(capture_folder, 'classify')
#
#     # create optimal classified rf path
#     optimal_rf_tif = os.path.join(classify_folder, 'rf_optimal.tif')
#
#     # check if optimal rf exists, error if not
#     if not os.path.exists(optimal_rf_tif):
#         arcpy.AddError('No optimal classified UAV image exists.')
#         return
#
#     try:
#         # read classified uav image as xr (save netcdf to scratch)
#         tmp_class_nc = os.path.join(arcpy.env.scratchFolder, 'rf_optimal.nc')
#         ds_hr = shared.single_band_raster_to_xr(in_ras=optimal_rf_tif,
#                                                 out_nc=tmp_class_nc)
#
#         # convert to array and clean it up for efficiency
#         da_hr = ds_hr[['Band1']].to_array().squeeze(drop=True)
#         da_hr = xr.where(~np.isnan(da_hr), da_hr, -999).astype('int16')
#
#     except Exception as e:
#         arcpy.AddError('Could not read classified UAV image. See messages.')
#         arcpy.AddMessage(str(e))
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region READ LOW RES SENTINEL 2 MEDIAN XR DATASET
#
#     arcpy.SetProgressor('default', 'Reading low-resolution Sentinel 2 data...')
#
#     try:
#         # load monthly median low-res satellite data
#         with xr.open_dataset(out_montly_meds_nc) as ds_lr:
#             ds_lr.load()
#
#         # slice from 2017-01-01 up to now
#         ds_lr = ds_lr.sel(time=slice('2016-01-01', None))
#
#     except Exception as e:
#         arcpy.AddError('Could not load Sentinel 2 images. See messages.')
#         arcpy.AddMessage(str(e))
#         return
#
#     # check if any time slices exist
#     if len(ds_lr['time']) == 0:
#         arcpy.AddError('No time slices in Sentinel 2 data.')
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region BUILD ROI WINDOWS FROM SENTINEL 2 GRID PIXELS
#
#     arcpy.SetProgressor('default', 'Creating training areas...')
#
#     try:
#         # get an all-time max of sentinel 2 cube to remove nulls
#         tmp_da = ds_lr.max('time', keep_attrs=True)
#
#         # export temp max netcdf to scratch
#         tmp_max_nc = os.path.join(arcpy.env.scratchFolder, 'tmp_max.nc')
#         tmp_da.to_netcdf(tmp_max_nc)
#
#         # convert temporary netcdf to a crf
#         tmp_max_crf = os.path.join(arcpy.env.scratchFolder, 'tmp_max.crf')
#         shared.netcdf_to_crf(in_nc=tmp_max_nc,
#                              out_crf=tmp_max_crf)
#
#         # read temp crf in as reproject to 32750
#         tmp_max_cmp = arcpy.Raster(tmp_max_crf)
#         tmp_max_prj = arcpy.ia.Reproject(tmp_max_cmp, {'wkid': 32750})
#
#         # create grid of 10 m rois from crf pixels in scratch
#         tmp_rois = os.path.join(arcpy.env.scratchFolder, 'tmp_roi.shp')
#         tmp_rois = uav_fractions.build_rois_from_raster(in_ras=tmp_max_prj,
#                                                         out_rois=tmp_rois)
#
#     except Exception as e:
#         arcpy.AddError('Could not create training areas. See messages.')
#         arcpy.AddMessage(str(e))
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region CALCULATE CLASS FREQUENCIES PER ROI
#
#     arcpy.SetProgressor('default', 'Calculating class fractions in training areas...')
#
#     try:
#         # calculate freq of high-res class pixels per sentinel 2 roi window
#         tmp_rois = uav_fractions.calculate_roi_freqs(rois=tmp_rois,
#                                                      da_hr=da_hr)
#
#         # subset to valid rois (i.e., not all nans) only and save shapefile
#         rois = os.path.join(arcpy.env.scratchFolder, 'rois.shp')
#         arcpy.analysis.Select(in_features=tmp_rois,
#                               out_feature_class=rois,
#                               where_clause='inc = 1')
#
#     except Exception as e:
#         arcpy.AddError('Could not calculate class fractions. See messages.')
#         arcpy.AddMessage(str(e))
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region GENERATE RANDOM SAMPLES IN CLASS FRACTION ROIS
#
#     arcpy.SetProgressor('default', 'Generating random samples...')
#
#     try:
#         # random sample rois and extract class values at each point
#         tmp_smp = os.path.join(arcpy.env.scratchFolder, 'samples.shp')
#         uav_fractions.random_sample_rois(in_rois=rois,
#                                          out_pnts=tmp_smp,
#                                          num_per_roi=2)
#     except Exception as e:
#         arcpy.AddError('Could not random sample class fractions. See messages.')
#         arcpy.AddMessage(str(e))
#         return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region GENERATE FRACTIONAL DATA
#
#     arcpy.SetProgressor('default', 'Generating fractional maps...')
#
#     # create fractions folder if not exists
#     fractions_folder = os.path.join(capture_folder, 'fractions')
#     if not os.path.exists(fractions_folder):
#         os.mkdir(fractions_folder)
#
#     # get list of prior processed year-month fractional folders
#     exist_fractions = meta_item.get('fractions')
#     if exist_fractions is None:
#         arcpy.AddError('No fraction list detected.')
#         return
#
#     # set up step-wise progressor
#     arcpy.SetProgressor('step', None, 0, len(ds_lr['time']))
#
#     try:
#         # iterate each date...
#         for i in range(0, len(ds_lr['time'])):
#             # extract time slice array and get year-month
#             da = ds_lr.isel(time=i)
#             dt = str(da['time'].dt.strftime('%Y-%m').values)
#
#             # skip if fraction date already exists
#             if dt in exist_fractions:
#                 arcpy.AddMessage(f'Skipping fractions for date: {dt}, already done.')
#                 continue
#
#             # notify of date currently processing
#             arcpy.AddMessage(f'Generating fractions for date: {dt}.')
#
#             # set new folder named year-month
#             fraction_folder = os.path.join(fractions_folder, dt)
#             if not os.path.exists(fraction_folder):
#                 os.mkdir(fraction_folder)
#
#             # convert dataset to multiband composite raster and reproject it
#             tmp_s2_cmp = shared.multi_band_xr_to_raster(da=da)
#             tmp_s2_prj = arcpy.ia.Reproject(tmp_s2_cmp, {'wkid': 32750})
#
#             # save it to scratch (regress will fail otherwise) and read it
#             tmp_exp_vars = os.path.join(arcpy.env.scratchFolder, 'tmp_exp_vars.tif')
#             tmp_s2_prj.save(tmp_exp_vars)
#
#             # TODO: unpack bands...
#
#             # TODO: copy roi points
#
#             # extract multival each band
#
#             # iter each class for fractional mapping...
#             frac_ncs = {}
#             classes = {'c_0': 'other', 'c_1': 'native', 'c_2': 'weed'}
#             for classvalue, classdesc in classes.items():
#                 # notify of class currently processing
#                 arcpy.AddMessage(f'> Working on class: {classdesc}.')
#
#                 # create output regression tif and cmatrix
#                 out_fn = f'{dt}_{classvalue}_{classdesc}'.replace('-', '_')
#                 out_frc_tif = os.path.join(fraction_folder, 'frac_' + out_fn + '.tif')
#
#                 # perform regression modelling and prediction
#                 ras_reg = uav_fractions.regress(exp_vars=tmp_exp_vars,
#                                                 sample_points=tmp_smp,
#                                                 classvalue=classvalue,
#                                                 classdesc=classdesc)
#
#                 # apply cubic resampling to smooth pixels out
#                 ras_rsp = arcpy.sa.Resample(raster=ras_reg,
#                                             resampling_type='Cubic',
#                                             input_cellsize=10,
#                                             output_cellsize=2.5)
#
#                 # save regression prediction
#                 ras_rsp.save(out_frc_tif)
#
#                 # convert to netcdf as well
#                 tmp_frc_nc = os.path.join(arcpy.env.scratchFolder, f'tmp_{classvalue}.nc')
#                 shared.single_band_raster_to_xr(in_ras=out_frc_tif,
#                                                 out_nc=tmp_frc_nc)
#
#                 # append to dict
#                 frac_ncs[classvalue] = tmp_frc_nc
#
#             # TODO: combine 3 bands into one netcdf for easier use later
#             # todo
#
#             # delete temp comp projected comp
#             del tmp_s2_cmp
#             del tmp_s2_prj
#             del tmp_exp_vars
#             del ras_reg
#
#             # add successful fractional date to metadata
#             meta_item['fractions'].append(dt)
#
#     except Exception as e:
#         arcpy.AddError('Could not generate fractional map. See messages.')
#         arcpy.AddMessage(str(e))
#         raise  # return
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region UPDATE NEW FRACTIONAL INFO IN METADATA
#
#     arcpy.SetProgressor('default', 'Updating metadata...')
#
#     try:
#         # write json metadata file to project folder top-level
#         with open(in_project_file, 'w') as fp:
#             json.dump(meta, fp)
#
#     except Exception as e:
#         arcpy.AddError('Could not write metadata. See messages.')
#         arcpy.AddMessage(str(e))
#         raise  # return
#
#     # endregion
#
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # region END ENVIRONMENT
#
#     # TODO: enable if move to non-memory temp files
#     # try:
#     #     # drop temp files (free up space)
#     #     arcpy.management.Delete(tmp_comp)
#     #
#     # except Exception as e:
#     #     arcpy.AddWarning('Could not drop temporary files. See messages.')
#     #     arcpy.AddMessage(str(e))
#
#     # free up spatial analyst
#     arcpy.CheckInExtension('Spatial')
#     arcpy.CheckInExtension('ImageAnalyst')  # TODO: remove if wc has no ia
#
#     # set changed env variables back to default
#     arcpy.env.overwriteOutput = False
#     arcpy.env.addOutputsToMap = True
#
#     # endregion
#
#     return
#
#
# # testing
# execute(None)

# old regress method
# def regress_deprecated(
#         in_rois: str,
#         in_classvalue: str,
#         in_class_desc: str,
#         in_exp_vars: list,
#         out_regress_tif: str,
#         out_cmatrix_csv: str,
# ) -> None:
#
#     # convert csv to dbf for function
#     out_cmatrix_dbf = os.path.splitext(out_cmatrix_csv)[0] + '.dbf'
#
#     # perform regression
#     # FIXME: this fails if we run via PyCharm - works ok via toolbox... threading?
#     arcpy.stats.Forest(prediction_type='PREDICT_RASTER',
#                        in_features=in_rois,
#                        variable_predict=in_classvalue,
#                        explanatory_rasters=in_exp_vars,
#                        output_raster=out_regress_tif,
#                        explanatory_rasters_matching=in_exp_vars,
#                        number_of_trees=100,
#                        percentage_for_training=25,
#                        number_validation_runs=5,
#                        output_validation_table=out_cmatrix_dbf)
#
#     # create output confususion matrix as a csv
#     #out_cmx_fn = f'cm_{dt}_{classvalue}_{class_desc}.csv'.replace('-', '_')
#     #out_cmx_csv = os.path.join(fraction_folder, out_cmx_fn)
#
#     # convert dbf to csv
#     arcpy.conversion.ExportTable(in_table=out_cmatrix_dbf,
#                                  out_table=out_cmatrix_csv)
#
#     # delete dbf
#     arcpy.management.Delete(out_cmatrix_dbf)
#
#     # read csv with pandas and get average r-squares
#     avg_r2 = pd.read_csv(out_cmatrix_csv)['R2'].mean().round(3)
#     arcpy.AddMessage(f'> Average R2 for {in_classvalue} ({in_class_desc}): {str(avg_r2)}')
#
#     return

# def regress(
#         exp_vars: str,
#         sample_points: str,
#         classvalue: str,
#         classdesc: str,
# ) -> arcpy.Raster:
#     """
#     Takes a multiband raster of explanatory variables (e.g., Sentinel 2 bands) and
#     a shapefile path of points containing extracted class fraction values to
#     model and performs random forest regression training and prediction.
#
#     :param exp_vars: Path to multiband raster of explanotory variables.
#     :param sample_points: Path to shapefile containing points with fraction values.
#     :param classvalue: Class value in sample points to predict.
#     :param classdesc: Description/label of class value.
#     :return: ArcPy Raster of predoction result.
#     """
#
#     try:
#         # train regression model for current target raster
#         tmp_ecd = os.path.join(arcpy.env.scratchFolder, f'ecd_{classvalue}.ecd')
#         arcpy.ia.TrainRandomTreesRegressionModel(in_rasters=exp_vars,
#                                                  in_target_data=sample_points,
#                                                  out_regression_definition=tmp_ecd,
#                                                  target_value_field=classvalue,
#                                                  max_num_trees=250,
#                                                  max_tree_depth=50,
#                                                  max_samples=-1,
#                                                  percent_testing=20)
#
#         # predict using trained model ecd file
#         ras_reg = arcpy.ia.PredictUsingRegressionModel(in_rasters=exp_vars,
#                                                        in_regression_definition=tmp_ecd)
#
#         # read ecd file (it is just a json file)
#         df = pd.read_json(tmp_ecd)
#
#         # prepare regression r2 metrics and notify
#         r2_train = round(df['Definitions'][0]['Train R2 at train locations'], 2)
#         r2_valid = round(df['Definitions'][0]['Test R2 at train locations'], 2)
#
#         # correct for < 0 models
#         r2_train = 0.0 if r2_train < 0 else r2_train
#         r2_valid = 0.0 if r2_valid < 0 else r2_valid
#
#         # prepare error r2 metrics and notify
#         er_train = round(df['Definitions'][0]['Train Error at train locations'], 2)
#         er_valid = round(df['Definitions'][0]['Test Error at train locations'], 2)
#
#         # notify user
#         arcpy.AddMessage(f'  - R2 Train: {str(r2_train)} / Validation: {str(r2_valid)}')
#         arcpy.AddMessage(f'  - Error Train: {str(er_train)} / Validation: {str(er_valid)}')
#
#     except Exception as e:
#         raise e
#
#     return ras_reg


# def convert_rois_to_rasters_deprecated(
#         rois: str,
#         out_folder: str
# ) -> tuple[str, str, str]:
#     """
#     Takes a filled in region of interest shapefile and
#     converts each fraction class within (0-2) to seperate
#     fraction rasters for use in regression model.
#
#     :param rois: Path to roi shapefile.
#     :param out_folder: Output folder to dump raster tifs.
#     :return: Pathes to each raster.
#     """
#
#     try:
#         # convert other class (c_) rois to a raster
#         out_c_0 = os.path.join(out_folder, 'roi_frac_c_0.tif')
#         arcpy.conversion.PolygonToRaster(in_features=rois,
#                                          value_field='c_0',
#                                          out_rasterdataset=out_c_0,
#                                          cellsize=10)
#
#         # convert native class (c_1) rois to a raster
#         out_c_1 = os.path.join(out_folder, 'roi_frac_c_1.tif')
#         arcpy.conversion.PolygonToRaster(in_features=rois,
#                                          value_field='c_1',
#                                          out_rasterdataset=out_c_1,
#                                          cellsize=10)
#
#         # convert weed class (c_2) rois to a raster
#         out_c_2 = os.path.join(out_folder, 'roi_frac_c_2.tif')
#         arcpy.conversion.PolygonToRaster(in_features=rois,
#                                          value_field='c_2',
#                                          out_rasterdataset=out_c_2,
#                                          cellsize=10)
#
#     except Exception as e:
#         raise e
#
#     return out_c_0, out_c_1, out_c_2

# code that extracts raster band values... bug with extractmult...
# make a copy of template fraction roi centroids
# tmp_smp_var = os.path.join(arcpy.env.scratchFolder, 'tmp_smp_var.shp')
# arcpy.management.CopyFeatures(in_features=tmp_smp,
# out_feature_class=tmp_smp_var)

# extract raster bands (in order) from projected raster at each point
# FIXME: this has a bug in script mode, input raster is broken
# arcpy.sa.ExtractMultiValuesToPoints(in_point_features=tmp_smp_var,
# in_rasters=tmp_s2_prj)

# was used to improve forest results, no longer using
# def random_sample_rois(
#         in_rois: str,
#         out_pnts: str,
#         num_per_roi: int
# ) -> str:
#     """
#     Takes a polygon shapefile of ROIs and generates
#     a set number of points per ROI. Then, extracts
#     the class values aut at each intersecting point.
#
#     :param in_rois: Path to ROI shapefile.
#     :param out_pnts: Path to output random sample shapefile.
#     :return: String to output random sample shapefile.
#     """
#
#     try:
#         # create random samples in roi polygons in memory
#         tmp_rnd_pnt = r'memory\rnd_pnt'
#         arcpy.management.CreateRandomPoints(out_path=r'memory',
#                                             out_name='rnd_pnt',
#                                             constraining_feature_class=in_rois,
#                                             number_of_points_or_field=num_per_roi)
#
#         # extract class values from rois per point
#         arcpy.analysis.PairwiseIntersect(in_features=[tmp_rnd_pnt, in_rois],
#                                          out_feature_class=out_pnts,
#                                          output_type='POINT')
#
#     except Exception as e:
#         raise e
#
#     return out_pnts

# arcgis version of trend rgb, less sophisticated as current
# export current var as netcdf
# tmp_nc = os.path.join(tmp, f'tmp_{var}.nc')
# da.to_netcdf(tmp_nc)
#
# # convert netcdf to crf
# tmp_crf = os.path.join(tmp, f'tmp_{var}.crf')
# shared.netcdf_to_crf(in_nc=tmp_nc,
#                      out_crf=tmp_crf)
#
# # generate linear trend on crf
# ras_tnd = arcpy.ia.GenerateTrendRaster(in_multidimensional_raster=tmp_crf,
#                                        dimension='StdTime',
#                                        line_type='Linear')
#
# # convert trend to rgb composite
# tmp_rgb = arcpy.ia.TrendToRGB(ras_tnd, 'LINEAR')
#
# # save rgb trend raster
# out_rgb = os.path.join(trends_folder, f'tmp_{var}_rgb.crf')
# # tmp_rgb.save(out_rgb)
#
# arcpy.management.CopyRaster(in_raster=tmp_rgb,
#                             out_rasterdataset=out_rgb)
#
# # add result to dictionary
# trend_rgbs[var] = out_rgb


# old optimal class model code
# try:
#     # get result item for best model based on kappa
#     best_rf_model = results[np.argmax([_['kappa'] for _ in results])]
#
#     # show user accuracy of best model
#     arcpy.AddMessage(f"Best Model Iteration: {best_rf_model['cv']}.")
#     arcpy.AddMessage(f"Best Overall Accuracy: {str(np.round(best_rf_model['oa'], 3))}.")
#     arcpy.AddMessage(f"Best Kappa: {str(np.round(best_rf_model['kappa'], 3))}.")
#
# except Exception as e:
#     arcpy.AddError('Could not display overall accuracy metrics. See messages.')
#     arcpy.AddMessage(str(e))
#     return

# todd low mid new raster change method
# arcpy.SetProgressor('default', 'Performing change detection...')
#
#     try:
#         # iter each var...
#         change_rasters = {}
#         for var in ['other', 'native', 'weed']:
#             # get "from", "mid", "to" raster paths
#             from_ras, mid_ras, to_ras = from_map[var], mid_map[var], to_map[var]
#
#             # perform change detection on uav data
#             tmp_epc_chg_from_mid = os.path.join(tmp, f'tmp_epc_chg_from_mid_{var}.tif')
#             tmp_epc_chg_from_to = os.path.join(tmp, f'tmp_cgf_from_to_{var}.tif')
#             change.detect_epoch_change(in_from_ras=from_ras,
#                                        in_mid_ras=mid_ras,
#                                        in_to_ras=to_ras,
#                                        out_from_mid_ras=tmp_epc_chg_from_mid,
#                                        out_from_to_ras=tmp_epc_chg_from_to)
#
#             # threshold "from" to "mid" into zscore where z < -2 or > 2
#             tmp_zsc_from_mid = os.path.join(tmp, f'tmp_zsc_from_mid_{var}.tif')
#             change.threshold_via_zscore(in_ras=tmp_epc_chg_from_mid,
#                                         out_z_ras=tmp_zsc_from_mid)
#
#             # do the same for "from" to "to" rasters
#             tmp_zsc_from_to = os.path.join(tmp, f'tmp_zsc_from_to_{var}.tif')
#             change.threshold_via_zscore(in_ras=tmp_epc_chg_from_to,
#                                         out_z_ras=tmp_zsc_from_to)
#
#             # perform categorical change detection between 0, 1, 2
#             tmp_cat_chg = os.path.join(tmp, f'tmp_cat_chg_from_to_{var}.tif')
#             change.detect_category_change(in_from_ras=tmp_zsc_from_mid,
#                                           in_to_ras=tmp_zsc_from_to,
#                                           out_change_ras=tmp_cat_chg)
#
#             # update class values to real labels
#             change.update_frac_classes(in_ras=tmp_cat_chg)
#
#
#             # TODO: use this instead of above
#             change.detect_diff_change(in_from_ras=from_ras,
#                                       in_to_ras=to_ras,
#                                       out_from_to_ras=tmp_epc_chg_from_to)
#
#             # TODO: update zscore for -1, 1
#             tmp_zsc = arcpy.Raster(tmp_zsc_from_to)
#             ras_pos = arcpy.ia.Con(tmp_zsc == 1, 1, 0)
#             ras_pos.save(os.path.join(tmp, f'{var}_z_pos.tif'))
#
#             ras_neg = arcpy.ia.Con(tmp_zsc == 2, 1, 0)
#             ras_neg.save(os.path.join(tmp, f'{var}_z_neg.tif'))
#
#
#             # TODO: implement in func
#             #for ras in [ras_pos, ras_neg]:
#                 # add field Class_name TEXT
#                 # loop rows and change value to (0) No Change and (1) var name
#                 #
#
#             # append to diff map
#
#
#
#
#             # create map of outputs
#             change_rasters[var] = tmp_cat_chg
#
#             # calc area?
#
#     except Exception as e:
#         arcpy.AddError('Could not perform fractional change detection. See messages.')
#         arcpy.AddMessage(str(e))
#         return

# old categorical frac attribute updater
# def update_frac_classes(
#         in_ras: str
# ) -> None:
#
#     try:
#         # create list of expected fields and extract
#         fields = ['Class_name', 'Class_From', 'Class_To']
#         with arcpy.da.UpdateCursor(in_ras, fields) as cursor:
#             for row in cursor:
#                 if row[0] == '0->0':
#                     row = ['Stable->Stable', 'S', 'S']
#                 elif row[0] == '0->1':
#                     row = ['Stable->Gain', 'S', 'G']
#                 elif row[0] == '0->2':
#                     row = ['Stable->Loss', 'S', 'L']
#                 elif row[0] == '1->0':
#                     row = ['Gain->Stable', 'G', 'S']
#                 elif row[0] == '1->1':
#                     row = ['Gain->Gain', 'G', 'G']
#                 elif row[0] == '1->2':
#                     row = ['Gain->Loss', 'G', 'L']
#                 elif row[0] == '2->0':
#                     row = ['Loss->Stable', 'L', 'S']
#                 elif row[0] == '2->1':
#                     row = ['Loss->Gain', 'L', 'G']
#                 elif row[0] == '2->2':
#                     row = ['Loss->Loss', 'L', 'L']
#
#                 # update cursor
#                 cursor.updateRow(row)
#
#     except Exception as e:
#         raise e
#
#     return

# old roi creator
# get an all-time max of sentinel 2 cube to remove nulls
# tmp_da = ds_lr.max('time', keep_attrs=True)

# export temp max netcdf to scratch
# tmp_max_nc = os.path.join(tmp, 'tmp_max.nc')
# tmp_da.to_netcdf(tmp_max_nc)
# tmp_da.close()

# convert temporary netcdf to a crf
# tmp_max_crf = os.path.join(tmp, 'tmp_max.crf')
# shared.netcdf_to_crf(in_nc=tmp_max_nc,
# out_crf=tmp_max_crf)

# read temp crf
# tmp_max_cmp = arcpy.Raster(tmp_max_crf)

# reproject it to wgs 1984 utm zone 50s using geoprocessor (ia func has issues)
# arcpy.management.ProjectRaster(in_raster=tmp_max_cmp,
# out_raster='tmp_max_prj.crf',
# out_coor_system=arcpy.SpatialReference(32750))

# read projected raster now
# tmp_max_cmp = arcpy.Raster('tmp_max_prj.crf')

# create grid of 10 m rois from crf pixels in scratch
# tmp_rois = os.path.join(tmp, 'tmp_roi.shp')
# tmp_rois = uav_fractions.build_rois_from_raster(in_ras=tmp_max_cmp,
# out_rois=tmp_rois)

# old roi generator
# def calculate_roi_freqs(
#         rois: str,
#         da_hr: xr.DataArray
# ) -> str:
#     """
#     Takes pre-created region of interest polygons and overlaps them with the pixels
#     of a high resolution xarray DataArray. The frequency of all DataArray pixel classes
#     falling within reach region of interest is added to the regoin of interest attrobute
#     table as seperate fields.
#
#     :param rois: Path to region of interest shapefile.
#     :param da_hr: A high-resolution classified raster as a xarray DataArray.
#     :return: Path to output region of interest shapefile.
#     """
#
#     try:
#         # extract all high-res class values within each low res pixel
#         with arcpy.da.UpdateCursor(rois, ['c_0', 'c_1', 'c_2', 'inc', 'SHAPE@']) as cursor:
#             for row in cursor:
#                 # get x and y window by slices for each polygon
#                 # careful, if arcpy rast->nc used, ymax,ymin, if gdal rast->nc, ymin,ymax...
#                 x_slice = slice(row[-1].extent.XMin, row[-1].extent.XMax)
#                 y_slice = slice(row[-1].extent.YMin, row[-1].extent.YMax)
#
#                 # extract window values from high-res
#                 arr = da_hr.sel(x=x_slice, y=y_slice).values
#
#                 # if values exist...
#                 if arr.size != 0 and ~np.all(arr == -999):
#                     # flatten array and remove -999s
#                     arr = arr.flatten()
#                     arr = arr[arr != -999]
#
#                     # get num classes/counts in win, prepare labels, calc freq
#                     classes, counts = np.unique(arr, return_counts=True)
#                     classes = [f'c_{c}' for c in classes]
#                     freqs = (counts / np.sum(counts)).astype('float16')
#
#                     # init fraction map
#                     class_map = {
#                         'c_0': 0.0,
#                         'c_1': 0.0,
#                         'c_2': 0.0,
#                         'inc': 1
#                     }
#
#                     # project existing classes and freqs onto map and update row
#                     class_map.update(dict(zip(classes, freqs)))
#                     row[0:4] = list(class_map.values())
#                 else:
#                     # flag row to be excluded
#                     row[3] = 0
#
#                 # update row
#                 cursor.updateRow(row)
#
#     except Exception as e:
#         raise e
#
#     return rois

# old roi to point funcs
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# region CALCULATE CLASS FREQUENCIES PER ROI

# arcpy.SetProgressor('default', 'Calculating class fractions in training areas...')

# try:
# calculate freq of high-res class pixels per sentinel 2 roi window
# tmp_rois = uav_fractions.calculate_roi_freqs(rois=tmp_rois,
# da_hr=da_hr)

# subset to valid rois (i.e., not all nans) only and save shapefile
# rois = os.path.join(tmp, 'rois.shp')
# arcpy.analysis.Select(in_features=tmp_rois,
# out_feature_class=rois,
# where_clause='inc = 1')

# except Exception as e:
# arcpy.AddError('Could not calculate class fractions. See messages.')
# arcpy.AddMessage(str(e))
# return

# endregion

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# region GENERATE CENTROID OF EACH ROI

# arcpy.SetProgressor('default', 'Generating training area centroids...')

# try:
# convert roi polygons to centroid points and extract class values
# tmp_smp = os.path.join(tmp, 'tmp_smp.shp')
# arcpy.management.FeatureToPoint(in_features=rois,
# out_feature_class=tmp_smp)

# except Exception as e:
# arcpy.AddError('Could not generate centroids. See messages.')
# arcpy.AddMessage(str(e))
# return

# endregion

# old roi to poly and pixel extractor funcs
# # TODO: remove
# def build_rois_from_raster(
#         in_ras: arcpy.Raster,
#         out_rois: str
# ) -> str:
#     """
#     Takes an ArcPy Raster class object and builds regions of interest polygons
#     from its pixels extents. Basically, the rois are the centroid point of each
#     pixel with a square buffer of the pixel size in x and y dimensions. The
#     output is a path to the shapefile of rois. This also creates three fields
#     for classes 0-2.
#
#     :param in_ras: ArcPy Raster.
#     :param out_rois: String to region of interest shapefile path.
#     :return: String to region of interest shapefile path.
#     """
#
#     try:
#         # get a centroid point for each grid cell on sentinel raster
#         tmp_pnts = r'memory\tmp_points'
#         arcpy.conversion.RasterToPoint(in_raster=in_ras,
#                                        out_point_features=tmp_pnts,
#                                        raster_field='Value')
#
#         # create square polygons around points
#         arcpy.analysis.GraphicBuffer(in_features=tmp_pnts,
#                                      out_feature_class=out_rois,
#                                      buffer_distance_or_field='5 Meters')  # half s2 cell size
#
#         # add required fields to envelope shapefile
#         arcpy.management.AddFields(in_table=out_rois,
#                                    field_description="c_0 FLOAT;c_1 FLOAT;c_2 FLOAT;inc SHORT")
#
#     except Exception as e:
#         raise e
#
#     return out_rois
#
#
#
# def extract_pixels_to_points(
#         in_pnts: str,
#         in_ras: str,
#         out_pnts: str
# ) -> str:
#     """
#     Takes points and a raster and extracts the bands
#     out of raster to new fields in points shapefile.
#
#     :param in_pnts: Path to a shapefile of points.
#     :param in_ras: Path to a raster.
#     :param in_ras: Path to a output point shapefile.
#     :return: Path to a output point shapefile.
#     """
#
#     try:
#         # get band values out at each point as new feature
#         tmp_smp_raw = r'memory\tmp_smp_raw'
#         arcpy.sa.Sample(in_rasters=in_ras,
#                         in_location_data=in_pnts,
#                         out_table=tmp_smp_raw,
#                         generate_feature_class='FEATURE_CLASS')
#
#         # join band values back on to fractional samples via spatial proximity
#         arcpy.analysis.SpatialJoin(target_features=in_pnts,
#                                    join_features=tmp_smp_raw,
#                                    out_feature_class=out_pnts)
#
#     except Exception as e:
#         raise e
#
#     return out_pnts

# old netcdf to raster to frac code
# # convert dataset to multiband composite raster and save it (errors otherwise?)
# tmp_s2_cmp = shared.multi_band_xr_to_raster(da=da, out_folder=tmp)
# tmp_s2_cmp.save('tmp_s2_cmb.tif')

# # reproject it to wgs 1984 utm zone 50s using geoprocessor (ia func has issues)
# arcpy.management.ProjectRaster(in_raster='tmp_s2_cmb.tif',
#                                out_raster='tmp_s2_prj.tif',
#                                out_coor_system=arcpy.SpatialReference(32750))

# # copy to new vars raster
# arcpy.management.CopyRaster(in_raster='tmp_s2_prj.tif',
#                             out_rasterdataset='vars.tif')
#
# # extract band values out per point
# tmp_smp_var = os.path.join(tmp, 'tmp_smp_var.shp')
# uav_fractions.extract_pixels_to_points(in_pnts=tmp_smp,
#                                        in_ras='vars.tif',
#                                        out_pnts=tmp_smp_var)

# old regress method
# def regress(
#         in_rois: str,
#         classvalue: str,
#         classdesc: str,
#         out_regress_shp: str,
# ) -> str:
#     """
#
#     :param in_rois:
#     :param classvalue:
#     :param classdesc:
#     :param in_exp_vars:
#     :param out_regress_shp:
#     :return:
#     """
#
#     # TODO: scratch aint good
#     # convert csv to dbf for function
#     tmp_cmat_dbf = os.path.join(arcpy.env.scratchFolder, 'tmp_cmat.dbf')
#
#     # create list of field names for bands 1 to 10
#     #var_names = [f'b{i}_vars' for i in range(1, 11)]  # TODO: remove if no longer using extract multival
#     var_names = [f'vars_{i}' for i in range(0, 10)]
#
#     # train and predict regression
#     arcpy.stats.Forest(prediction_type='PREDICT_FEATURES',
#                        in_features=in_rois,
#                        variable_predict=classvalue,
#                        explanatory_variables=var_names,
#                        features_to_predict=in_rois,
#                        output_features=out_regress_shp,
#                        explanatory_variable_matching=var_names,
#                        number_of_trees=250,
#                        percentage_for_training=10,
#                        output_validation_table=tmp_cmat_dbf,
#                        number_validation_runs=3)
#
#     # convert temporary dbf to temporary csv
#     tmp_cmat_csv = os.path.join(arcpy.env.scratchFolder, 'tmp_cmat.csv')
#     arcpy.conversion.ExportTable(in_table=tmp_cmat_dbf,
#                                  out_table=tmp_cmat_csv)
#
#     # read csv with pandas and get average r-squares
#     med_r2 = pd.read_csv(tmp_cmat_csv)['R2'].median().round(3)
#     arcpy.AddMessage(f'> Median R2 for {classvalue} ({classdesc}): {str(med_r2)}')
#
#     # delete cmatrix dbf and csv
#     arcpy.management.Delete(tmp_cmat_dbf)
#     arcpy.management.Delete(tmp_cmat_csv)
#
#     return

# old regres helper
# def build_frac_map(
#         in_pnts: str,
#         classvalue: str,
#         classdesc: str,
#         out_ras: str,
#         out_nc_dt: str,
#         out_nc: str
# ) -> tuple:
#     """
#     Helper function that builds regression model from
#     points, converts prediction point output to
#     raster, then resamples to higher resolution to
#     improve look of fraction map. Finally, a NetCDF
#     version of the raster is also exported for time-series
#     work. This exists to reduce code in generatefractions
#     geoprocessor.
#
#     :param in_pnts: Path to training shapefile.
#     :param classvalue: Value of class in shapefile to predict.
#     :param classdesc: Label of class in shapefile being predicted.
#     :param out_ras: Path to output raster.
#     :param out_nc_dt: Datetime of current fraction for NetCDF.
#     :param out_nc: Path to output NetCDF.
#     :return:
#     """
#
#     #try:
#         # TODO: remove when happy
#         # train and predict regression model
#         #tmp_prd = os.path.join(arcpy.env.scratchFolder, 'tmp_prd.shp')
#         # tmp_prd = r'memory\tmp_prd'
#         # regress(in_rois=in_pnts,
#         #         classvalue=classvalue,
#         #         classdesc=classdesc,
#         #         out_regress_shp=tmp_prd)
#
#         # TODO: WORKING
#         # tmp_gwr_prd = r'memory\tmp_gwr_prd'
#         # gwregress(in_rois=in_pnts,
#         #           classvalue=classvalue,
#         #           classdesc=classdesc,
#         #           out_regress_shp=tmp_gwr_prd)
#
#
#         # convert points to 10m res pixel raster
#         #tmp_prd_ras = os.path.join(arcpy.env.scratchFolder, 'tmp_prd_ras.tif')
#         # tmp_prd_ras = r'memory\tmp_prd_ras'
#         # arcpy.conversion.PointToRaster(in_features=tmp_prd,
#         #                                value_field='PREDICTED',
#         #                                out_rasterdataset=tmp_prd_ras,
#         #                                cellsize=10.0)
#
#         # apply cubic resampling to smooth pixels out
#         # ras_rsp = arcpy.sa.Resample(raster=tmp_prd_ras,
#         #                             resampling_type='Cubic',
#         #                             input_cellsize=10.0,
#         #                             output_cellsize=2.5)
#
#         # save prediction raster to output file path
#         # ras_rsp.save(out_ras)
#
#         # create a netcdf version of frac tif as well
#         # shared.raster_to_xr(in_ras=out_ras,
#         #                     out_nc=out_nc,
#         #                     epsg=32750,
#         #                     datetime=out_nc_dt,
#         #                     var_names=[classdesc],
#         #                     dtype='float64')
#
#         # # TODO: REMOVE THIS JUST TESTING NOW
#         # # tmp_prd_ras = os.path.join(arcpy.env.scratchFolder, 'tmp_prd_ras.tif')
#         # tmp_prd_gwr_ras = r'memory\tmp_prd_gwr_ras'
#         # arcpy.conversion.PointToRaster(in_features=tmp_gwr_prd,
#         #                                value_field='PREDICTED',
#         #                                out_rasterdataset=tmp_prd_gwr_ras,
#         #                                cellsize=10.0)
#         #
#         # # apply cubic resampling to smooth pixels out
#         # ras_rsp2 = arcpy.sa.Resample(raster=tmp_prd_gwr_ras,
#         #                             resampling_type='Cubic',
#         #                             input_cellsize=10.0,
#         #                             output_cellsize=2.5)
#         #
#         # # save prediction raster to output file path
#         # out_ras_2 = out_ras.split('.')[0] + '_gwr.tif'
#         # ras_rsp2.save(out_ras_2)
#
#         ...
#
#     #except Exception as e:
#         #raise e
#
#     #return out_ras, out_nc

# extract r2 adjusted r2 and aicc from gwr
# # init result dict
# result = {
#     'R2': None,
#     'AdjR2': None,
#     'AICc': None
# }
#
# try:
#     # extract accuracy metrics from messages
#     num_msgs = model.messageCount
#     for i in range(num_msgs):
#         msg = model.getMessage(i)
#         if '- Model Diagnostics -' in msg:
#             for line in msg.split('\n'):
#                 if line.startswith('R2'):
#                     result['R2'] = line.replace('R2', '').strip()
#                 elif line.startswith('AdjR2'):
#                     result['AdjR2'] = line.replace('AdjR2', '').strip()
#                 elif line.startswith('AICc'):
#                     result['AICc'] = line.replace('AICc', '').strip()
#             break
#
#     # notify user of accuracy metrics
#     arcpy.AddMessage(f'> R-Squared for {classdesc}: {result["R2"]}')
#     arcpy.AddMessage(f'> Adj. R-Squared for {classdesc}: {result["AdjR2"]}')
#     arcpy.AddMessage(f'> AICc for {classdesc}: {result["AICc"]}')

# # create dataframe and export to csv
# df = pd.DataFrame.from_dict(result.items())
# df.to_csv(out_accuracy_csv)

# old glcm threaded method
# #num_cpu = 1
#
# #results = []
# # with ThreadPoolExecutor(max_workers=num_cpu) as pool:
# #     futures = []
# #     for metric in metrics:
# #         task = pool.submit(fast_glcm_specify,
# #                            arr,
# #                            metric,  # glcm metric name
# #                            0,       # vmin
# #                            255,     # vmax
# #                            2,       # levels
# #                            5,       # kernel size
# #                            1.0,     # distance
# #                            45.0)    # angle
# #
# #         futures.append(task)
# #         for future in as_completed(futures):
# #             done_glcm, done_metric = future.result()
# #             out_tex = arcpy.NumPyArrayToRaster(in_array=done_glcm,
# #                                                lower_left_corner=ll,
# #                                                x_cell_size=x_size,
# #                                                y_cell_size=y_size)
# #
# #             out_fp = os.path.join(r"C:\Users\Lewis\Desktop\glcm\results", f'tx_{metric}.tif')
# #             out_tex.save(out_fp)
# #             #results.append({'metric': done_metric, 'data': done_glcm})
# #             print(f'Metric: {done_metric} finished.')
#
# # out_ras = arcpy.NumPyArrayToRaster(done_glcm, low_left, cell_size)
#
# # out_fp = os.path.join(r'C:\Users\Lewis\Desktop\New folder', f'tx_{done_metric}.tif')
# # out_ras.save(out_fp)
# # out_ras = None
#
# # import time
# #
# # start = time.time()
#
# for metric in metrics:
#     print(f'Working on {metric}')
#
#     glcm_arr = fast_glcm_specify(arr, metric, 0, 255, 4, 5, 1.0, 45.0)
#     #tex_arr = arcpy.NumPyArrayToRaster(in_array=glcm_arr,
#                                        #lower_left_corner=ll,
#                                        #x_cell_size=x_size,
#                                        #y_cell_size=y_size)
#     #out_fp = os.path.join(r"C:\Users\Lewis\Desktop\glcm\results", f'tx_{metric}.tif')
#     #tex_arr.save(out_fp)
#
#
#
#
#
# # end = time.time()
# # print(end - start)

# old first order textures
# setup neighbourhood object
# win = arcpy.sa.NbrRectangle(5, 5, 'CELL')

# iter each texture and calculate
# for k, v in tx_map.items():
# if k == 'mean':
#     ras = arcpy.sa.FocalStatistics(in_raster=tmp_gry,
#                                    neighborhood=win,
#                                    statistics_type='MEAN')
# elif k == 'max':
#     ras = arcpy.sa.FocalStatistics(in_raster=tmp_gry,
#                                    neighborhood=win,
#                                    statistics_type='MAXIMUM')
# elif k == 'min':
#     ras = arcpy.sa.FocalStatistics(in_raster=tmp_gry,
#                                    neighborhood=win,
#                                    statistics_type='MINIMUM')
# elif k == 'stdev':
#     ras = arcpy.sa.FocalStatistics(in_raster=tmp_gry,
#                                    neighborhood=win,
#                                    statistics_type='STD')
# elif k == 'range':
#     ras = arcpy.sa.FocalStatistics(in_raster=tmp_gry,
#                                    neighborhood=win,
#                                    statistics_type='RANGE')
#
# # save to associated path
# ras.save(v)

# notify user
# arcpy.AddMessage(f'Texture index {k} done.')

# increment progressor
# arcpy.SetProgressorPosition()