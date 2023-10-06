# -*- coding: utf-8 -*-
r""""""
__all__ = ['ClassifyUAVCapture', 'CreateNewSite', 'DetectFractionChange',
           'DetectUAVChange', 'DisplayData', 'GenerateFractions',
           'GenerateTrend', 'IngestNewUAVCapture', 'Testing']
__alias__ = 'WCMonitor'
from arcpy.geoprocessing._base import gptooldoc, gp, gp_fixargs
from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject

# Tools
@gptooldoc('ClassifyUAVCapture_WCMonitor', None)
def ClassifyUAVCapture(in_project_file=None, in_capture_datetime=None, in_include_prior=None, in_roi_feat=None, in_variables=None):
    """ClassifyUAVCapture_WCMonitor(in_project_file, in_capture_datetime, in_include_prior, in_roi_feat, in_variables;in_variables...)

        Classify a UAV image using Random Forest classification. The UAV
        image will be classified into three classes: Native, Weed and Other
        based on field-observed regions of interest (ROIs).

     INPUTS:
      in_project_file (File):
          Every site has a unique meta.json file in its project folder. Find
          and select this file to set the current site containing the UAV image
          you want to classify.
      in_capture_datetime (String):
          Set the UAV image to classify based on the date and time it was
          flown.
      in_include_prior (Boolean):
          Tick this to access previously classified UAV captures from the
          above drop-down. This allows user to re-classify prior classified UAV
          images, if need be.
      in_roi_feat (Feature Layer):
          Regions of interest (ROIs) are required to train and classify the
          UAV image. The ROIs should be provided as a shapefile of polygons
          with at least two columns called Classvalue (Long) and Classname
          (string). These columns must contain the following three classes:
          Classvalue: 0 | Classname: Other   Classvalue: 1 | Classname: Native
          Classvalue: 2 | Classname: Weed   Additional, at least 10 polygons per
          class is required to ensure enough data is available for accuracy
          measurement. We recommend at least 50 per class. <SPAN />
      in_variables (String):
          Set additional variables to be considered in the Random Forest
          classification algorithm. The default variables represent the best
          variables in the model based on our research. Note: the tool will
          always use the raw UAV bands in the classification."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.ClassifyUAVCapture_WCMonitor(*gp_fixargs((in_project_file, in_capture_datetime, in_include_prior, in_roi_feat, in_variables), True)))
        return retval
    except Exception as e:
        raise e

@gptooldoc('CreateNewSite_WCMonitor', None)
def CreateNewSite(in_output_folder=None, in_boundary_feat=None, in_rehab_datetime=None, in_flight_datetime=None, in_blue_band=None, in_green_band=None, in_red_band=None, in_redge_band=None, in_nir_band=None, in_dsm_band=None, in_dtm_band=None):
    """CreateNewSite_WCMonitor(in_output_folder, in_boundary_feat, in_rehab_datetime, in_flight_datetime, in_blue_band, in_green_band, in_red_band, in_redge_band, in_nir_band, in_dsm_band, in_dtm_band)

        Create a new rehabilitation site and associated project folder.

     INPUTS:
      in_output_folder (Folder):
          Folder location of new project folder and files.
      in_boundary_feat (Feature Layer):
          Precise boundary of site's rehabilitation area footprint.
      in_rehab_datetime (Date):
          Start date and time rehabilitation commended.
      in_flight_datetime (Date):
          Start date and time of first UAV image capture.
      in_blue_band (Raster Layer):
          Raster representing the blue reflectance band captured from UAV.
      in_green_band (Raster Layer):
          Raster representing the green reflectance band captured from UAV.
      in_red_band (Raster Layer):
          Raster representing the red reflectance band captured from UAV.
      in_redge_band (Raster Layer):
          Raster representing the red edge reflectance band captured from UAV.
      in_nir_band (Raster Layer):
          Raster representing the near-infrared (NIR) reflectance band
          captured from UAV.
      in_dsm_band (Raster Layer):
          Raster representing the digital surface model (DSM) band captured
          from UAV.
      in_dtm_band (Raster Layer):
          Raster representing the digital terrain model (DTM) band captured
          from UAV."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.CreateNewSite_WCMonitor(*gp_fixargs((in_output_folder, in_boundary_feat, in_rehab_datetime, in_flight_datetime, in_blue_band, in_green_band, in_red_band, in_redge_band, in_nir_band, in_dsm_band, in_dtm_band), True)))
        return retval
    except Exception as e:
        raise e

@gptooldoc('DetectFractionChange_WCMonitor', None)
def DetectFractionChange(in_project_file=None, in_capture_datetime=None, in_s2_from_year=None, in_s2_to_year=None, in_s2_month=None):
    """DetectFractionChange_WCMonitor(in_project_file, in_capture_datetime, in_s2_from_year, in_s2_to_year, in_s2_month)

        Perform a change detection between two fractional cover dates.
        Significant positive and negative changes between the two fraction
        images are then obtained using a global Z-Score method to detect
        outliers.

     INPUTS:
      in_project_file (File):
          Every site has a unique meta.json file in its project folder. Find
          and select this file to set the current site containing the UAV image
          you want to asses fractional change for.
      in_capture_datetime (String):
          Set the UAV image for which fractional cover images have been
          previously generated for, based on the date and time it was flown. If
          fractional cover images have not been generated yet, run the "Generate
          Fractions" tool first.
      in_s2_from_year (Long):
          Set the "baseline" fractional cover image year.
      in_s2_to_year (Long):
          Set the "to" fractional cover image year. This will be compared to
          the "from" or "baseline" fractional cover image.
      in_s2_month (Long):
          Set the month of the fractional cover images to assess change for."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.DetectFractionChange_WCMonitor(*gp_fixargs((in_project_file, in_capture_datetime, in_s2_from_year, in_s2_to_year, in_s2_month), True)))
        return retval
    except Exception as e:
        raise e

@gptooldoc('DetectUAVChange_WCMonitor', None)
def DetectUAVChange(in_project_file=None, in_uav_from_date=None, in_uav_to_date=None, in_use_majority_filter=None, in_use_shrink_filter=None):
    """DetectUAVChange_WCMonitor(in_project_file, in_uav_from_date, in_uav_to_date, in_use_majority_filter, in_use_shrink_filter)

        Perform a categorical change detection between two classified UAV
        images captured at different dates. The output will display where
        Native, Weed and Other classes changed between the two input UAV
        images. Basic summary statistics area also provided in the
        geoprocessing messages panel.

     INPUTS:
      in_project_file (File):
          Every site has a unique meta.json file in its project folder. Find
          and select this file to set the current site containing the UAV image
          you want to generate fraction models for.
      in_uav_from_date (String):
          Set the "baseline" (i.e., "from") UAV capture date and time.
      in_uav_to_date (String):
          Set the "to" UAV capture date and time. This will be compared to
          the "baseline" (i.e., "from") UAV capture image.
      in_use_majority_filter (Boolean):
          Apply a majority filter using a 5x5 focal window to remove isolated
          pixels and classification noise.
      in_use_shrink_filter (Boolean):
          Apply a shrink filter to reduce spatial footprint of pixels by up to
          1 pixel. This algorithm greatly reduces the "salt and pepper" noise
          effect but may also eliminate smaller, valid changes. Recommended if
          you want to focus on larger, more significant changes."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.DetectUAVChange_WCMonitor(*gp_fixargs((in_project_file, in_uav_from_date, in_uav_to_date, in_use_majority_filter, in_use_shrink_filter), True)))
        return retval
    except Exception as e:
        raise e

@gptooldoc('DisplayData_WCMonitor', None)
def DisplayData(in_project_file=None, in_capture_datetime=None, in_layers_to_visualise=None):
    """DisplayData_WCMonitor(in_project_file, in_capture_datetime, in_layers_to_visualise;in_layers_to_visualise...)

     INPUTS:
      in_project_file (File):
          Existing Project File
      in_capture_datetime (String):
          UAV Capture to Visualise
      in_layers_to_visualise (String):
          Layers to Visualise"""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.DisplayData_WCMonitor(*gp_fixargs((in_project_file, in_capture_datetime, in_layers_to_visualise), True)))
        return retval
    except Exception as e:
        raise e

@gptooldoc('GenerateFractions_WCMonitor', None)
def GenerateFractions(in_project_file=None, in_capture_datetime=None):
    """GenerateFractions_WCMonitor(in_project_file, in_capture_datetime)

        Harness Sentinel 2 Analysis Ready Data (ARD) obtained from Digital
        Earth Australia (DEA) to generate monthly fractional cover models of a
        specified site. Geographically-Weighted Regression (GWR) is applied
        to extrapolate UAV image Native, Weed and Other classes onto satellite
        imagery at different dates. This offers a modelled look at how a
        site's natives and weeds may be evolving over time without frequent
        capture of new UAV imagery. The resulting data is to be used in the
        Detect Fractional Change and Generate Trend tools.

     INPUTS:
      in_project_file (File):
          Every site has a unique meta.json file in its project folder. Find
          and select this file to set the current site containing the UAV image
          you want to generate fraction models for.
      in_capture_datetime (String):
          Set the UAV image for which to generate fractional images based on
          the date and time it was flown."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.GenerateFractions_WCMonitor(*gp_fixargs((in_project_file, in_capture_datetime), True)))
        return retval
    except Exception as e:
        raise e

@gptooldoc('GenerateTrend_WCMonitor', None)
def GenerateTrend(in_project_file=None, in_capture_datetime=None, in_rehab_or_capture_month=None, in_manual_year=None, in_manual_month=None, in_export_raw_fractions=None):
    """GenerateTrend_WCMonitor(in_project_file, in_capture_datetime, in_rehab_or_capture_month, in_manual_year, in_manual_month, in_export_raw_fractions)

        Use fractional cover data obtained from the Generate Fractions tool
        to create RGB trend images of native and weeds over time. Trend
        trajectories are built at the same month across years (anniversary
        dates). At least three years required. Colours on the RGB trend image
        correspond to different trend types:   Red: Continuous loss over time.
        Green: Recent gain (last few years) after period of loss. Recent gain
        typically back to original or higher levels. Blue: Continuous gain
        over time. Yellow: Recent gain (last few years) after period of
        loss. Recent gain not up to original levels. Cyan: Continuous gain
        over time, with highest gains recently (last few years). Black: No
        significant gain or loss (stable). Can be stable high or low values.
        White: Not possible. Magenta: Not possible. The brighter the
        colour above, the more significant the trend. <SPAN />

     INPUTS:
      in_project_file (File):
          Every site has a unique meta.json file in its project folder. Find
          and select this file to set the current site containing the UAV image
          you want to RGB trends for.
      in_capture_datetime (String):
          Set the UAV image for which to generate RGB trends based on the
          date and time it was flown.
      in_rehab_or_capture_month (String):
          Set the month to use for trend analysis. The month that
          rehabilitation began at the site is the default. As this can occur
          temporally far away from the month of UAV capture, it is recommended
          to use UAV capture month instead to reduce model error. Use the manual
          option to set a specific trend start year and anniversary month.
      in_manual_year (Long):
          Set a specific year of start trend analysis. Must be 2016 or above.
      in_manual_month (Long):
          Set a specific month for anniversary dates. Trends will be created
          at this month year after year.
      in_export_raw_fractions (Boolean):
          The underlying raw fractional time-series data can be helpful to
          assess trend lines along with the RGB trend images. Enable this to
          export this data along with RGB trends."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.GenerateTrend_WCMonitor(*gp_fixargs((in_project_file, in_capture_datetime, in_rehab_or_capture_month, in_manual_year, in_manual_month, in_export_raw_fractions), True)))
        return retval
    except Exception as e:
        raise e

@gptooldoc('IngestNewUAVCapture_WCMonitor', None)
def IngestNewUAVCapture(in_project_file=None, in_flight_datetime=None, in_blue_band=None, in_green_band=None, in_red_band=None, in_redge_band=None, in_nir_band=None, in_dsm_band=None, in_dtm_band=None):
    """IngestNewUAVCapture_WCMonitor(in_project_file, in_flight_datetime, in_blue_band, in_green_band, in_red_band, in_redge_band, in_nir_band, in_dsm_band, in_dtm_band)

        Add a new UAV capture to an existing site. It is expected this will
        be a UAV capture taken at a similar month to the initial capture but
        at subsequent year(s). Change detection can be undertaken if more than
        one UAV capture exists for a project.

     INPUTS:
      in_project_file (File):
          Every site has a unique meta.json file in its project folder. Find
          and select this file to set the current site to ingest new data into.
      in_flight_datetime (Date):
          Start date and time of new UAV image capture.
      in_blue_band (Raster Layer):
          Raster representing the blue reflectance band captured from UAV.
      in_green_band (Raster Layer):
          Raster representing the green reflectance band captured from UAV.
      in_red_band (Raster Layer):
          Raster representing the red reflectance band captured from UAV.
      in_redge_band (Raster Layer):
          Raster representing the red edge reflectance band captured from
          UAV.
      in_nir_band (Raster Layer):
          Raster representing the near-infrared (NIR) reflectance band
          captured from UAV.
      in_dsm_band (Raster Layer):
          Raster representing the digital surface model (DSM) band captured
          from UAV.
      in_dtm_band (Raster Layer):
          Raster representing the digital terrain model (DTM) band captured
          from UAV."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.IngestNewUAVCapture_WCMonitor(*gp_fixargs((in_project_file, in_flight_datetime, in_blue_band, in_green_band, in_red_band, in_redge_band, in_nir_band, in_dsm_band, in_dtm_band), True)))
        return retval
    except Exception as e:
        raise e

@gptooldoc('Testing_WCMonitor', None)
def Testing(in_folder=None):
    """Testing_WCMonitor(in_folder)

     INPUTS:
      in_folder (Folder):
          Test Data Folder"""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.Testing_WCMonitor(*gp_fixargs((in_folder,), True)))
        return retval
    except Exception as e:
        raise e


# End of generated toolbox code
del gptooldoc, gp, gp_fixargs, convertArcObjectToPythonObject