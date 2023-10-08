
def execute(
        parameters
):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region IMPORTS

    import os
    import warnings
    import shutil
    import arcpy

    from scripts import tests

    # set data overwrites and mapping
    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = False

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region WARNINGS

    # disable warnings
    warnings.filterwarnings('ignore')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region EXTRACT PARAMETERS

    # inputs from arcgis pro ui
    in_folder = parameters[0].valueAsText

    # inputs for testing only
    #in_folder = r'D:\Work\Curtin\Water Corp Project - General\Testing\Test data'

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK DATA FOLDER CORRECT

    arcpy.SetProgressor('default', 'Checking test data folder...')

    # check if input test folder exists
    if not os.path.exists(in_folder):
        arcpy.AddError('Test data folder does not exist.')
        return

    # create test areas folders
    cb_folder = os.path.join(in_folder, 'city_beach')

    # check if it contains three folders
    folders = [cb_folder]
    for folder in folders:
        if not os.path.exists(in_folder):
            arcpy.AddError(f'Test data folder does not exist: {folder}.')
            return

    # create input folder
    in_folder = os.path.join(cb_folder, 'inputs')

    # create project folder
    project_folder = os.path.join(cb_folder, 'project')

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK LICENSES

    arcpy.SetProgressor('default', 'Checking licenses...')

    try:
        # test if advanced license can be obtained
        tests.test_license()
        arcpy.AddMessage('Licenses passed test.')

    except Exception as e:
        arcpy.AddError('Licenses failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK EXTENSIONS

    arcpy.SetProgressor('default', 'Checking extensions...')

    try:
        # test if spatial/image analyst extensions can be obtained
        tests.test_extensions()
        arcpy.AddMessage('Extensions passed test.')

    except Exception as e:
        arcpy.AddError('Extensions failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region CHECK WEB ACCESS

    arcpy.SetProgressor('default', 'Checking web access...')

    # create temp folder if doesnt exist
    tmp_folder = os.path.join(cb_folder, 'tmp')
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    try:
        # test if dea can be reached and s2 data cant be obtained
        tests.test_web(tmp_folder)
        arcpy.AddMessage('Web access passed test.')

    except Exception as e:
        arcpy.AddError('Web access failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region RUN CREATE NEW SITE TOOL

    arcpy.SetProgressor('default', 'Checking create new site tool...')

    try:
        # delete project folder if already exists
        shutil.rmtree(project_folder)

    except Exception as e:
        arcpy.AddError('Could not delete previous project folder. See messages.')
        arcpy.AddMessage(str(e))

    # recreate it
    os.mkdir(project_folder)

    try:
        # test if dea can be reached and s2 data cant be obtained
        tests.test_createnewsite(in_folder, project_folder)
        arcpy.AddMessage('Create new site tool passed test.')

    except Exception as e:
        arcpy.AddError('Create new site tool failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region RUN INGEST NEW UAV CAPTURE TOOL

    arcpy.SetProgressor('default', 'Checking ingest new UAV capture tool...')

    try:
        tests.test_ingestnewuavcapture(in_folder, project_folder)
        arcpy.AddMessage('Ingest new UAV capture tool passed test.')

    except Exception as e:
        arcpy.AddError('Ingest new UAV capture tool failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region RUN CLASSIFY UAV CAPTURE TOOL (ONE)

    arcpy.SetProgressor('default', 'Checking classify UAV capture tool (one)...')

    try:
        in_date = '2022-02-02 10:30:15'
        tests.test_classifyuavcapture(in_folder, in_date, project_folder)
        arcpy.AddMessage('Classify UAV capture tool passed test.')

    except Exception as e:
        arcpy.AddError('Classify UAV capture tool failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region RUN CLASSIFY UAV CAPTURE TOOL (TWO)

    arcpy.SetProgressor('default', 'Checking classify UAV capture tool (two)...')

    try:
        in_date = '2023-02-02 10:30:15'
        tests.test_classifyuavcapture(in_folder, in_date, project_folder)
        arcpy.AddMessage('Classify UAV capture tool passed test.')

    except Exception as e:
        arcpy.AddError('Classify UAV capture tool failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region RUN GENERATE FRACTIONS TOOL

    arcpy.SetProgressor('default', 'Checking generate fractions tool...')

    try:
        tests.test_generatefractions(in_folder, project_folder)
        arcpy.AddMessage('Generate Fractions tool passed test.')

    except Exception as e:
        arcpy.AddError('Generate Fractions tool failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region RUN GENERATE TREND TOOL

    arcpy.SetProgressor('default', 'Checking generate trend tool...')

    try:
        tests.test_generatetrend(in_folder, project_folder)
        arcpy.AddMessage('Generate Trend tool passed test.')

    except Exception as e:
        arcpy.AddError('Generate Trend tool failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region RUN DETECT UAV CHANGE TOOL

    arcpy.SetProgressor('default', 'Checking detect UAV change tool...')

    try:
        tests.test_detectuavchange(in_folder, project_folder)
        arcpy.AddMessage('Detect UAV Change tool passed test.')

    except Exception as e:
        arcpy.AddError('Detect UAV Change tool failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region RUN DETECT FRACTION CHANGE TOOL

    arcpy.SetProgressor('default', 'Checking detect fraction change tool...')

    try:
        tests.test_detectfractionchange(in_folder, project_folder)
        arcpy.AddMessage('Detect Fraction Change tool passed test.')

    except Exception as e:
        arcpy.AddError('Detect Fraction Change tool failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # region RUN GENERATE NDVI TOOL

    arcpy.SetProgressor('default', 'Checking generate NDVI tool...')

    try:
        tests.test_generatendvi(in_folder, project_folder)
        arcpy.AddMessage('Generate NDVI tool passed test.')

    except Exception as e:
        arcpy.AddError('Generate NDVI tool failed test. See messages.')
        arcpy.AddMessage(str(e))

    # endregion

# testing
#execute(None)
