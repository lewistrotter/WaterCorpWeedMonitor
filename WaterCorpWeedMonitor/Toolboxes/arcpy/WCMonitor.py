# -*- coding: utf-8 -*-
r""""""
__all__ = ['CreateNewSite', 'IngestNewUAVCapture']
__alias__ = 'WCMonitor'
from arcpy.geoprocessing._base import gptooldoc, gp, gp_fixargs
from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject

# Tools
@gptooldoc('CreateNewSite_WCMonitor', None)
def CreateNewSite(in_output_folder=None, in_flight_datetime=None, in_blue_band=None, in_green_band=None, in_red_band=None, in_redge_band=None, in_nir_band=None, in_dsm_band=None, in_dtm_band=None):
    """CreateNewSite_WCMonitor(in_output_folder, in_flight_datetime, in_blue_band, in_green_band, in_red_band, in_redge_band, in_nir_band, in_dsm_band, in_dtm_band)

     INPUTS:
      in_output_folder (Folder):
          New Project Folder
      in_flight_datetime (Date):
          Flight Date and Time
      in_blue_band (Raster Layer):
          Blue Band
      in_green_band (Raster Layer):
          Green Band
      in_red_band (Raster Layer):
          Red Band
      in_redge_band (Raster Layer):
          Red Edge Band
      in_nir_band (Raster Layer):
          Near-Infrared (NIR) Band
      in_dsm_band (Raster Layer):
          Digital Surface Model (DSM) Band
      in_dtm_band (Raster Layer):
          Digital Terrain Model (DTM) Band"""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.CreateNewSite_WCMonitor(*gp_fixargs((in_output_folder, in_flight_datetime, in_blue_band, in_green_band, in_red_band, in_redge_band, in_nir_band, in_dsm_band, in_dtm_band), True)))
        return retval
    except Exception as e:
        raise e

@gptooldoc('IngestNewUAVCapture_WCMonitor', None)
def IngestNewUAVCapture(in_project_folder=None, in_flight_datetime=None, in_blue_band=None, in_green_band=None, in_red_band=None, in_redge_band=None, in_nir_band=None, in_dsm_band=None, in_dtm_band=None):
    """IngestNewUAVCapture_WCMonitor(in_project_folder, in_flight_datetime, in_blue_band, in_green_band, in_red_band, in_redge_band, in_nir_band, in_dsm_band, in_dtm_band)

     INPUTS:
      in_project_folder (Folder):
          Existing Project Folder
      in_flight_datetime (Date):
          Flight Date and Time
      in_blue_band (Raster Layer):
          Blue Band
      in_green_band (Raster Layer):
          Green Band
      in_red_band (Raster Layer):
          Red Band
      in_redge_band (Raster Layer):
          Red Edge Band
      in_nir_band (Raster Layer):
          Near-Infrared (NIR) Band
      in_dsm_band (Raster Layer):
          Digital Surface Model (DSM) Band
      in_dtm_band (Raster Layer):
          Digital Terrain Model (DTM) Band"""
    from arcpy.geoprocessing._base import gp, gp_fixargs
    from arcpy.arcobjects.arcobjectconversion import convertArcObjectToPythonObject
    try:
        retval = convertArcObjectToPythonObject(gp.IngestNewUAVCapture_WCMonitor(*gp_fixargs((in_project_folder, in_flight_datetime, in_blue_band, in_green_band, in_red_band, in_redge_band, in_nir_band, in_dsm_band, in_dtm_band), True)))
        return retval
    except Exception as e:
        raise e


# End of generated toolbox code
del gptooldoc, gp, gp_fixargs, convertArcObjectToPythonObject