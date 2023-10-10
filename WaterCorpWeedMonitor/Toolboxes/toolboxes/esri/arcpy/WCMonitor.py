# -*- coding: utf-8 -*-
r""""""
__all__ = ['CalibrateSatellite', 'ClassifyUAVCapture', 'CreateNewSite',
           'DetectFractionChange', 'DetectUAVChange', 'DisplayData',
           'GenerateFractions', 'GenerateNDVI', 'GenerateTrend',
           'IngestNewUAVCapture', 'Testing']
__alias__ = 'WCMonitor'
from arcpy.geoprocessing._base import gptooldoc, gp, gp_fixargs
from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject

# Tools
@gptooldoc('CalibrateSatellite_WCMonitor', None)
def CalibrateSatellite(in_project_file=None, in_operation=None, in_x_shift=None, in_y_shift=None):
    """CalibrateSatellite_WCMonitor(in_project_file, in_operation, in_x_shift, in_y_shift)

        Sentinel 2 imagery obtained from Digital Earth Australia may
        spatially "shift" depending on the bounding box coordinates used to
        obtain the imagery. This could lead to satellite data used in fraction
        analysis to be slightly unaligned from UAV images. This tool can be
        used to calibrate Sentinel 2 data by shifting it in x and y directions
        prior to its use in analyses. Once the satellite is calibrated, these
        shift parameters will be used in all following analyses for the
        selected site. Note: once fraction images have been generated, this
        tool will no longer work - it must be used prior to fractions being
        generated for a site.

     INPUTS:
      in_project_file (File):
          Every site has a unique meta.json file in its project folder. Find
          and select this file to set the current site you want to calibrate.
      in_operation (String):
          Select whether to shift Sentinel 2 data or reset the shift values
          from prior runs.
      in_x_shift (Double):
          Shift the Sentinel 2 image along the x dimension (i.e., west to
          east). Use a negative value to shift west and a positive value to
          shift east. It is recommended no more than 5 metres is used.
      in_y_shift (Double):
          Shift the Sentinel 2 image along the y dimension (i.e., north to
          south). Use a negative value to shift south and a positive value to
          shift north. It is recommended no more than 5 metres is used."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.CalibrateSatellite_WCMonitor(*gp_fixargs((in_project_file, in_operation, in_x_shift, in_y_shift), True)))
        return retval
    except Exception as e:
        raise e

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
def DetectFractionChange(in_project_file=None, in_capture_datetime=None, in_from_year=None, in_manual_from_year=None, in_to_year=None, in_manual_to_year=None, in_month=None, in_manual_month=None, in_z=None):
    """DetectFractionChange_WCMonitor(in_project_file, in_capture_datetime, in_from_year, in_manual_from_year, in_to_year, in_manual_to_year, in_month, in_manual_month, in_z)

        Perform a change detection between two fractional cover dates.
        Gains and losses for natives, weeds and other classes found between
        the two fraction image dates are obtained using a global Z-Score
        method to detect outliers.

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
      in_from_year (String):
          Set the "from" year to use for change detection analysis. The year
          that rehabilitation began at the site is the default. Use the manual
          option to set a specific "from" year.
      in_manual_from_year (Long):
          Set the "from" year manually. Must be &gt;= 2016.
      in_to_year (String):
          Set the "to" year to use for change detection analysis. The current
          year is the default. If the current year does not have the requested
          month below yet, will roll back a year. Use the manual option to set a
          specific "to" year. <SPAN />
      in_manual_to_year (Long):
          Set the "to" year manually. Must be &gt; the "from" year. <SPAN />
      in_month (String):
          Set the month to compare images on during change detection
          analysis. The default is the month rehabilitation commenced. Manually
          override the month using the manual option.
      in_manual_month (Long):
          Set a specific month for change detection.
      in_z (Long):
          Set the z-score threshold to use for separating change into
          "significant" and "insignificant". The z-score value represents
          standard deviation. For example, a z-score of 2 will highlight any
          native or weed change &gt;2 standard deviations of the typical change
          values occurring on the imagery. The higher the z-score value, the
          more significant the change has to be in order to be returned on
          output. Must be between 1 and 3 (inclusive)."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.DetectFractionChange_WCMonitor(*gp_fixargs((in_project_file, in_capture_datetime, in_from_year, in_manual_from_year, in_to_year, in_manual_to_year, in_month, in_manual_month, in_z), True)))
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
          and select this file to set the current site containing the UAV images
          you want to compare.
      in_uav_from_date (String):
          Set the "baseline" (i.e., "from") UAV capture date and time.
      in_uav_to_date (String):
          Set the "to" UAV capture date and time. This will be compared to
          the "baseline" (i.e., "from") UAV capture image.
      in_use_majority_filter (Boolean):
          Apply a majority filter using a 5x5 focal window to remove isolated
          pixels and classification noise.
      in_use_shrink_filter (Boolean):
          Apply a shrink filter to reduce spatial footprint of pixels by up
          to 1 pixel. This algorithm greatly reduces the "salt and pepper" noise
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

        Quickly display various results from previously run tools.

     INPUTS:
      in_project_file (File):
          Every site has a unique meta.json file in its project folder. Find
          and select this file to set the current site containing the UAV image
          you want to display results for.
      in_capture_datetime (String):
          Set the UAV image for which to display previously generated results
          for based on the date and time it was flown.
      in_layers_to_visualise (String):
          Select various results to display. If layer does not exist, user
          will receive a warning."""
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

@gptooldoc('GenerateNDVI_WCMonitor', None)
def GenerateNDVI(in_project_file=None, in_freq=None):
    """GenerateNDVI_WCMonitor(in_project_file, in_freq)

        Generate NDVI time-series data for a site based on previously
        obtained Sentinel 2 data. Run the Generate Fractions tool prior to
        using this to obtain Sentinel 2 data.

     INPUTS:
      in_project_file (File):
          Every site has a unique meta.json file in its project folder. Find
          and select this file to set the current site containing the UAV image
          you want to generate NDVI data for.
      in_freq (String):
          Select the temporal frequency in which NDVI time-series data will
          be presented. Raw NDVI data will be aggregated to this frequency via
          median aggregator."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.GenerateNDVI_WCMonitor(*gp_fixargs((in_project_file, in_freq), True)))
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
          you want to generate RGB trends for.
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

        Run this tool to run various diagnostics on your computer to check
        if the plug-in works. Note: you will need provided test data found on
        the project's GitHub page.

     INPUTS:
      in_folder (Folder):
          Set the folder of the provided test data. Click the Help button to
          open the project's GitHub page and download the test data to run
          diagnotics."""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.Testing_WCMonitor(*gp_fixargs((in_folder,), True)))
        return retval
    except Exception as e:
        raise e


# End of generated toolbox code
del gptooldoc, gp, gp_fixargs, convertArcObjectToPythonObject