# -*- coding: utf-8 -*-import osimport jsonimport datetimeimport arcpyclass Toolbox(object):    def __init__(self):        self.label = 'WCMonitor'        self.alias = 'WCMonitor'        self.tools = [            CreateNewSite,            IngestNewUAVCapture,            ClassifyUAVCapture,            GenerateFractions,            GenerateTrend,            DetectUAVChange,            DetectFractionChange,            DisplayData,            Testing        ]class CreateNewSite(object):    def __init__(self):        self.label = 'Create New Site'        self.description = 'Create a new rehabilitation site and project data structure.'        self.canRunInBackground = False    def getParameterInfo(self):        params = []        p00 = arcpy.Parameter(displayName='New Project Folder',                              name='in_output_folder',                              datatype='DEFolder',                              parameterType='Required',                              direction='Input')        params.append(p00)        p01 = arcpy.Parameter(displayName='Rehabilitation Boundary',                              name='in_boundary_feat',                              datatype='GPFeatureLayer',                              parameterType='Required',                              direction='Input')        p01.filter.list = ['Polygon']        params.append(p01)        p02 = arcpy.Parameter(displayName='Rehabilitation Date and Time',                              name='in_rehab_datetime',                              datatype='GPDate',                              parameterType='Required',                              direction='Input')        params.append(p02)        p03 = arcpy.Parameter(displayName='Flight Date and Time',                              name='in_flight_datetime',                              datatype='GPDate',                              parameterType='Required',                              direction='Input')        params.append(p03)        p04 = arcpy.Parameter(displayName='Blue Band',                              name='in_blue_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p04)        p05 = arcpy.Parameter(displayName='Green Band',                              name='in_green_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p05)        p06 = arcpy.Parameter(displayName='Red Band',                              name='in_red_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p06)        p07 = arcpy.Parameter(displayName='Red Edge Band',                              name='in_redge_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p07)        p08 = arcpy.Parameter(displayName='Near-Infrared (NIR) Band',                              name='in_nir_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p08)        p09 = arcpy.Parameter(displayName='Digital Surface Model (DSM) Band',                              name='in_dsm_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p09)        p10 = arcpy.Parameter(displayName='Digital Terrain Model (DTM) Band',                              name='in_dtm_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p10)        return params    def isLicensed(self):        if arcpy.CheckProduct('ArcInfo') not in ['AlreadyInitialized', 'Available']:            return False        elif arcpy.CheckExtension('Spatial') != 'Available':            return False        elif arcpy.CheckExtension('ImageAnalyst') != 'Available':            return False        return True    def updateParameters(self, parameters):        return    def updateMessages(self, parameters):        return    def execute(self, parameters, messages):        from geoprocessors import createnewsite        createnewsite.execute(parameters)        return    def postExecute(self, parameters):        returnclass IngestNewUAVCapture(object):    def __init__(self):        self.label = 'Ingest New UAV Capture'        self.description = 'Ingests a new UAV image capture into an existing project.'        self.canRunInBackground = False    def getParameterInfo(self):        params = []        p00 = arcpy.Parameter(displayName='Existing Project File',                              name='in_project_file',                              datatype='DEFile',                              parameterType='Required',                              direction='Input')        p00.filter.list = ['json']        params.append(p00)        p01 = arcpy.Parameter(displayName='Flight Date and Time',                              name='in_flight_datetime',                              datatype='GPDate',                              parameterType='Required',                              direction='Input')        params.append(p01)        p02 = arcpy.Parameter(displayName='Blue Band',                              name='in_blue_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p02)        p03 = arcpy.Parameter(displayName='Green Band',                              name='in_green_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p03)        p04 = arcpy.Parameter(displayName='Red Band',                              name='in_red_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p04)        p05 = arcpy.Parameter(displayName='Red Edge Band',                              name='in_redge_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p05)        p06 = arcpy.Parameter(displayName='Near-Infrared (NIR) Band',                              name='in_nir_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p06)        p07 = arcpy.Parameter(displayName='Digital Surface Model (DSM) Band',                              name='in_dsm_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p07)        p08 = arcpy.Parameter(displayName='Digital Terrain Model (DTM) Band',                              name='in_dtm_band',                              datatype='GPRasterLayer',                              parameterType='Required',                              direction='Input')        params.append(p08)        return params    def isLicensed(self):        if arcpy.CheckProduct('ArcInfo') not in ['AlreadyInitialized', 'Available']:            return False        elif arcpy.CheckExtension('Spatial') != 'Available':            return False        elif arcpy.CheckExtension('ImageAnalyst') != 'Available':            return False        return True    def updateParameters(self, parameters):        return    def updateMessages(self, parameters):        return    def execute(self, parameters, messages):        from geoprocessors import ingestnewuavcapture        ingestnewuavcapture.execute(parameters)        return    def postExecute(self, parameters):        returnclass ClassifyUAVCapture(object):    def __init__(self):        self.label = 'Classify UAV Capture'        self.description = 'Classify UAV capture into Native, Weed or Other classes.'        self.canRunInBackground = False    def getParameterInfo(self):        params = []        p00 = arcpy.Parameter(displayName='Existing Project File',                              name='in_project_file',                              datatype='DEFile',                              parameterType='Required',                              direction='Input')        p00.filter.list = ['json']        params.append(p00)        p01 = arcpy.Parameter(displayName='UAV Capture to Classify',                              name='in_capture_datetime',                              datatype='GPString',                              parameterType='Required',                              direction='Input')        p01.filter.type = 'ValueList'        p01.filter.list = []        p01.value = None        params.append(p01)        p02 = arcpy.Parameter(displayName='Include Prior Classifications',                              name='in_include_prior',                              datatype='GPBoolean',                              parameterType='Required',                              direction='Input')        p02.value = False        params.append(p02)        p03 = arcpy.Parameter(displayName='Field Training Areas',                              name='in_roi_feat',                              datatype='GPFeatureLayer',                              parameterType='Required',                              direction='Input')        p03.filter.list = ['Polygon']        params.append(p03)        p04 = arcpy.Parameter(displayName='Variables',                              name='in_variables',                              datatype='GPString',                              parameterType='Required',                              direction='Input',                              category='Variables',                              multiValue=True)        p04.filter.type = 'ValueList'        p04.filter.list = get_all_classify_variables()        p04.values = get_default_classify_variables()        params.append(p04)        return params    def isLicensed(self):        if arcpy.CheckProduct('ArcInfo') not in ['AlreadyInitialized', 'Available']:            return False        elif arcpy.CheckExtension('Spatial') != 'Available':            return False        elif arcpy.CheckExtension('ImageAnalyst') != 'Available':            return False        return True    def updateParameters(self, parameters):        if not parameters[0].altered:            parameters[1].value = None            parameters[1].filter.list = []        else:            try:                fp = open(parameters[0].valueAsText, 'r')                meta = json.load(fp)                fp.close()                exclude_keys = ['project_name', 'date_created', 'date_rehab']                if not parameters[0].hasBeenValidated or not parameters[2].hasBeenValidated:                    capture_dates = []                    for k, v in meta.items():                        if k not in exclude_keys:                            if parameters[2].value is False:                                if v['classified'] is False:                                    capture_dates.append(v['capture_date'])                            else:                                capture_dates.append(v['capture_date'])                    parameters[1].value = None                    parameters[1].filter.list = capture_dates            except:                parameters[1].value = None                parameters[1].filter.list = []        return    def updateMessages(self, parameters):        return    def execute(self, parameters, messages):        from geoprocessors import classifyuavcapture        classifyuavcapture.execute(parameters)        return    def postExecute(self, parameters):        returnclass GenerateFractions(object):    def __init__(self):        self.label = 'Generate Fractions'        self.description = 'Generate fractional layers of Native, Weed or Other classes.'        self.canRunInBackground = False    def getParameterInfo(self):        params = []        p00 = arcpy.Parameter(displayName='Existing Project File',                              name='in_project_file',                              datatype='DEFile',                              parameterType='Required',                              direction='Input')        p00.filter.list = ['json']        params.append(p00)        p01 = arcpy.Parameter(displayName='UAV Capture to Convert to Fractions',                      name='in_capture_datetime',                      datatype='GPString',                      parameterType='Required',                      direction='Input')        p01.filter.type = 'ValueList'        p01.filter.list = []        p01.value = None        params.append(p01)        return params    def isLicensed(self):        if arcpy.CheckProduct('ArcInfo') not in ['AlreadyInitialized', 'Available']:            return False        elif arcpy.CheckExtension('Spatial') != 'Available':            return False        elif arcpy.CheckExtension('ImageAnalyst') != 'Available':            return False        return True    def updateParameters(self, parameters):        if not parameters[0].altered:            parameters[1].value = None            parameters[1].filter.list = []        else:            try:                fp = open(parameters[0].valueAsText, 'r')                meta = json.load(fp)                fp.close()                exclude_keys = ['project_name', 'date_created', 'date_rehab']                if not parameters[0].hasBeenValidated:                    capture_dates = []                    for k, v in meta.items():                        if k not in exclude_keys:                            if v['classified'] is True:                                capture_dates.append(v['capture_date'])                    parameters[1].value = None                    parameters[1].filter.list = capture_dates            except:                parameters[1].value = None                parameters[1].filter.list = []        return    def updateMessages(self, parameters):        return    def execute(self, parameters, messages):        from geoprocessors import generatefractions        generatefractions.execute(parameters)        return    def postExecute(self, parameters):        returnclass GenerateTrend(object):    def __init__(self):        self.label = 'Generate Trend'        self.description = 'Generates RGB trend of each fraction layer over time.'        self.canRunInBackground = False    def getParameterInfo(self):        params = []        p00 = arcpy.Parameter(displayName='Existing Project File',                              name='in_project_file',                              datatype='DEFile',                              parameterType='Required',                              direction='Input')        p00.filter.list = ['json']        params.append(p00)        p01 = arcpy.Parameter(displayName='UAV Capture of Fractions',                              name='in_capture_datetime',                              datatype='GPString',                              parameterType='Required',                              direction='Input')        p01.filter.type = 'ValueList'        p01.filter.list = []        p01.value = None        params.append(p01)        p02 = arcpy.Parameter(displayName='Month of Trend Analysis',                              name='in_rehab_or_capture_month',                              datatype='GPString',                              parameterType='Required',                              direction='Input')        p02.filter.type = 'ValueList'        p02.filter.list = ['Month of Rehabilitation', 'Month of First UAV Capture', 'Manual']        p02.value = 'Month of Rehabilitation'        params.append(p02)        p03 = arcpy.Parameter(displayName='Specify Year',                              name='in_manual_year',                              datatype='GPLong',                              parameterType='Required',                              direction='Input')        p03.filter.type = 'Range'        p03.filter.list = [2016, 2039]        p03.value = 2018        params.append(p03)        p04 = arcpy.Parameter(displayName='Specify Month',                              name='in_manual_month',                              datatype='GPLong',                              parameterType='Required',                              direction='Input')        p04.filter.type = 'Range'        p04.filter.list = [1, 12]        p04.value = 6        params.append(p04)        p05 = arcpy.Parameter(displayName='Export Raw Fractions',                              name='in_export_raw_fractions',                              datatype='GPBoolean',                              parameterType='Required',                              direction='Input')        p05.value = False        params.append(p05)        return params    def isLicensed(self):        if arcpy.CheckProduct('ArcInfo') not in ['AlreadyInitialized', 'Available']:            return False        elif arcpy.CheckExtension('Spatial') != 'Available':            return False        elif arcpy.CheckExtension('ImageAnalyst') != 'Available':            return False        return True    def updateParameters(self, parameters):        if not parameters[0].altered:            parameters[1].value = None            parameters[1].filter.list = []        else:            try:                fp = open(parameters[0].valueAsText, 'r')                meta = json.load(fp)                fp.close()                exclude_keys = ['project_name', 'date_created', 'date_rehab']                if not parameters[0].hasBeenValidated:                    capture_dates = []                    for k, v in meta.items():                        if k not in exclude_keys:                            if v['classified'] is True:                                if len(v['fractions']) > 0:                                    capture_dates.append(v['capture_date'])                    parameters[1].value = None                    parameters[1].filter.list = capture_dates            except:                parameters[1].value = None                parameters[1].filter.list = []        if parameters[2].value == 'Manual':            parameters[3].enabled = True            parameters[4].enabled = True        else:            parameters[3].enabled = False            parameters[4].enabled = False        return    def updateMessages(self, parameters):        return    def execute(self, parameters, messages):        from geoprocessors import generatetrend        generatetrend.execute(parameters)        return    def postExecute(self, parameters):        returnclass DetectUAVChange(object):    def __init__(self):        self.label = 'Detect UAV Change'        self.description = 'Detect "from-to" class changes between UAV captures.'        self.canRunInBackground = False    def getParameterInfo(self):        params = []        p00 = arcpy.Parameter(displayName='Existing Project File',                              name='in_project_file',                              datatype='DEFile',                              parameterType='Required',                              direction='Input')        p00.filter.list = ['json']        params.append(p00)        p01 = arcpy.Parameter(displayName='From Date',                              name='in_uav_from_date',                              datatype='GPString',                              parameterType='Required',                              direction='Input')        p01.filter.type = 'ValueList'        p01.filter.list = []        p01.value = None        params.append(p01)        p02 = arcpy.Parameter(displayName='To Date',                              name='in_uav_to_date',                              datatype='GPString',                              parameterType='Required',                              direction='Input')        p02.filter.type = 'ValueList'        p02.filter.list = []        p02.value = None        params.append(p02)        p03 = arcpy.Parameter(displayName='Apply Majority Filter',                              name='in_use_majority_filter',                              datatype='GPBoolean',                              parameterType='Required',                              category='Noise Reduction Options',                              direction='Input')        p03.value = True        params.append(p03)        p04 = arcpy.Parameter(displayName='Apply Shrink Filter',                              name='in_use_shrink_filter',                              datatype='GPBoolean',                              parameterType='Required',                              category='Noise Reduction Options',                              direction='Input')        p04.value = False        params.append(p04)        return params    def isLicensed(self):        if arcpy.CheckProduct('ArcInfo') not in ['AlreadyInitialized', 'Available']:            return False        elif arcpy.CheckExtension('Spatial') != 'Available':            return False        elif arcpy.CheckExtension('ImageAnalyst') != 'Available':            return False        return True    def updateParameters(self, parameters):        if not parameters[0].altered:            parameters[1].value = None            parameters[1].filter.list = []        else:            try:                fp = open(parameters[0].valueAsText, 'r')                meta = json.load(fp)                fp.close()                exclude_keys = ['project_name', 'date_created', 'date_rehab']                if not parameters[0].hasBeenValidated:                    from_capture_dates = []                    to_capture_dates = []                    for k, v in meta.items():                        if k not in exclude_keys:                            if v['classified'] is True:                                from_capture_dates.append(v['capture_date'])                                to_capture_dates.append(v['capture_date'])                    parameters[1].value = None                    parameters[1].filter.list = []                    if len(from_capture_dates) > 0:                        parameters[1].filter.list = from_capture_dates                        parameters[1].value = from_capture_dates[0]                    parameters[2].value = None                    parameters[2].filter.list = []                    if len(to_capture_dates) > 1:                        parameters[2].filter.list = to_capture_dates                        parameters[2].value = to_capture_dates[-1]            except:                parameters[1].value = None                parameters[1].filter.list = []                parameters[2].value = None                parameters[2].filter.list = []        return    def updateMessages(self, parameters):        return    def execute(self, parameters, messages):        from geoprocessors import detectuavchange        detectuavchange.execute(parameters)        return    def postExecute(self, parameters):        returnclass DetectFractionChange(object):    def __init__(self):        self.label = 'Detect Fraction Change'        self.description = 'Detect gain and loss change per fraction class.'        self.canRunInBackground = False    def getParameterInfo(self):        params = []        p00 = arcpy.Parameter(displayName='Existing Project File',                              name='in_project_file',                              datatype='DEFile',                              parameterType='Required',                              direction='Input')        p00.filter.list = ['json']        params.append(p00)        p01 = arcpy.Parameter(displayName='UAV Capture of Fractions',                              name='in_capture_datetime',                              datatype='GPString',                              parameterType='Required',                              direction='Input')        p01.filter.type = 'ValueList'        p01.filter.list = []        p01.value = None        params.append(p01)        p02 = arcpy.Parameter(displayName='From Year',                              name='in_from_year',                              datatype='GPString',                              parameterType='Required',                              direction='Input')        p02.filter.type = 'ValueList'        p02.filter.list = ['Year of Rehabilitation', 'Manual']        p02.value = 'Year of Rehabilitation'        params.append(p02)        p03 = arcpy.Parameter(displayName='Specify From Year',                              name='in_manual_from_year',                              datatype='GPLong',                              parameterType='Required',                              direction='Input')        p03.filter.type = 'Range'        p03.filter.list = [2016, 2039]        p03.value = 2018        params.append(p03)        p04 = arcpy.Parameter(displayName='To Year',                              name='in_to_year',                              datatype='GPString',                              parameterType='Required',                              direction='Input')        p04.filter.type = 'ValueList'        p04.filter.list = ['Current Year', 'Manual']        p04.value = 'Current Year'        params.append(p04)        p05 = arcpy.Parameter(displayName='Specify To Year',                              name='in_manual_to_year',                              datatype='GPLong',                              parameterType='Required',                              direction='Input')        p05.filter.type = 'Range'        p05.filter.list = [2016, 2039]        p05.value = datetime.datetime.now().year        params.append(p05)        p06 = arcpy.Parameter(displayName='Analysis Month ',                              name='in_month',                              datatype='GPString',                              parameterType='Required',                              direction='Input')        p06.filter.type = 'ValueList'        p06.filter.list = ['Month of Rehabilitation', 'Month of First UAV Capture', 'Manual']        p06.value = 'Month of Rehabilitation'        params.append(p06)        p07 = arcpy.Parameter(displayName='Specific Month',                              name='in_manual_month',                              datatype='GPLong',                              parameterType='Required',                              direction='Input')        p07.filter.type = 'Range'        p07.filter.list = [1, 12]        p07.value = 6        params.append(p07)        p08 = arcpy.Parameter(displayName='Z-score Threshold',                              name='in_z',                              datatype='GPLong',                              parameterType='Required',                              category='Statistical Threshold',                              direction='Input')        p08.filter.type = 'Range'        p08.filter.list = [1, 3]        p08.value = 2        params.append(p08)        return params    def isLicensed(self):        if arcpy.CheckProduct('ArcInfo') not in ['AlreadyInitialized', 'Available']:            return False        elif arcpy.CheckExtension('Spatial') != 'Available':            return False        elif arcpy.CheckExtension('ImageAnalyst') != 'Available':            return False        return True    def updateParameters(self, parameters):        if not parameters[0].altered:            parameters[1].value = None            parameters[1].filter.list = []        else:            try:                fp = open(parameters[0].valueAsText, 'r')                meta = json.load(fp)                fp.close()                exclude_keys = ['project_name', 'date_created', 'date_rehab']                if not parameters[0].hasBeenValidated:                    capture_dates = []                    for k, v in meta.items():                        if k not in exclude_keys:                            if v['classified'] is True:                                if len(v['fractions']) > 0:                                    capture_dates.append(v['capture_date'])                    parameters[1].value = None                    parameters[1].filter.list = capture_dates            except:                parameters[1].value = None                parameters[1].filter.list = []        if parameters[2].value == 'Manual':            parameters[3].enabled = True        else:            parameters[3].enabled = False        if parameters[4].value == 'Manual':            parameters[5].enabled = True        else:            parameters[5].enabled = False        if parameters[6].value == 'Manual':            parameters[7].enabled = True        else:            parameters[7].enabled = False        return    def updateMessages(self, parameters):        return    def execute(self, parameters, messages):        from geoprocessors import detectfracchange        detectfracchange.execute(parameters)        return    def postExecute(self, parameters):        returnclass DisplayData(object):    def __init__(self):        self.label = 'Display Data'        self.description = 'Displays previously created data on the currently active map.'        self.canRunInBackground = False    def getParameterInfo(self):        params = []        p00 = arcpy.Parameter(displayName='Existing Project File',                              name='in_project_file',                              datatype='DEFile',                              parameterType='Required',                              direction='Input')        p00.filter.list = ['json']        params.append(p00)        p01 = arcpy.Parameter(displayName='UAV Capture to Visualise',                              name='in_capture_datetime',                              datatype='GPString',                              parameterType='Required',                              direction='Input')        p01.filter.type = 'ValueList'        p01.filter.list = []        p01.value = None        params.append(p01)        p02 = arcpy.Parameter(displayName='Layers to Visualise',                              name='in_layers_to_visualise',                              datatype='GPString',                              parameterType='Required',                              direction='Input',                              multiValue=True)        p02.filter.type = 'ValueList'        lyrs = ['UAV RGB', 'UAV NDVI', 'UAV Classified', 'S2 NDVI', 'S2 Fractions']        p02.filter.list = lyrs        p02.value = lyrs        params.append(p02)        return params    def isLicensed(self):        if arcpy.CheckProduct('ArcInfo') not in ['AlreadyInitialized', 'Available']:            return False        elif arcpy.CheckExtension('Spatial') != 'Available':            return False        elif arcpy.CheckExtension('ImageAnalyst') != 'Available':            return False        return True    def updateParameters(self, parameters):        if not parameters[0].altered:            parameters[1].value = None            parameters[1].filter.list = []        else:            try:                fp = open(parameters[0].valueAsText, 'r')                meta = json.load(fp)                fp.close()                exclude_keys = ['project_name', 'date_created', 'date_rehab']                if not parameters[0].hasBeenValidated:                    capture_dates = []                    for k, v in meta.items():                        if k not in exclude_keys:                            capture_dates.append(v['capture_date'])                    parameters[1].value = None                    parameters[1].filter.list = capture_dates            except:                parameters[1].value = None                parameters[1].filter.list = []    def updateMessages(self, parameters):        return    def execute(self, parameters, messages):        from geoprocessors import displaydata        displaydata.execute(parameters)        return    def postExecute(self, parameters):        returnclass Testing(object):    def __init__(self):        self.label = 'Tests (Development)'        self.description = 'Run tests to see if WaterCorp tool works.'        self.canRunInBackground = False    def getParameterInfo(self):        params = []        p00 = arcpy.Parameter(displayName='Test Data Folder',                              name='in_folder',                              datatype='DEFolder',                              parameterType='Required',                              direction='Input')        params.append(p00)        return params    def isLicensed(self):        if arcpy.CheckProduct('ArcInfo') not in ['AlreadyInitialized', 'Available']:            return False        elif arcpy.CheckExtension('Spatial') != 'Available':            return False        elif arcpy.CheckExtension('ImageAnalyst') != 'Available':            return False        return True    def updateParameters(self, parameters):        return    def updateMessages(self, parameters):        return    def execute(self, parameters, messages):        from geoprocessors import testing        testing.execute(parameters)        return    def postExecute(self, parameters):        returndef get_all_classify_variables():    variables = [        'NDVI',        'NDREI',        'NGRDI',        'RGBVI',        'OSAVI',        'Mean',        'Minimum',        'Maximum',        'StanDev',        'Range',        'Skew',        'Kurtosis',        #'Contrast',        #'Dissimilarity',        'Entropy',        #'Homogeneity',        #'SecondMoment',        'Variance',        'CHM'        ]    return variablesdef get_default_classify_variables():    variables = [        'NDVI',        'NDREI',        'NGRDI',        'RGBVI',        'OSAVI',        #'Mean',        #'Minimum',        #'Maximum',        'StanDev',        #'Range',        #'Skew',        #'Kurtosis',        'Entropy',        #'Variance',        'CHM'        ]    return variables