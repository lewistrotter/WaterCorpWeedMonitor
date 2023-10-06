
import pandas as pd
import arcpy


def detect_category_change(
        in_from_ras: str,
        in_to_ras: str,
        out_change_ras: str
) -> str:
    """
    Computes a change raster for UAV data. Must include a
    from and to raster of same area. The output are categorical
    changes of each class e.g., Native - Weed. Saves output
    to designated output location.

    :param in_from_ras: Path to a UAV classified raster.
    :param in_to_ras: Path to a UAV classified raster.
    :param out_change_ras: Output path to a change raster.
    :return: Path of output change raster.
    """

    try:
        # read from and to uav raster in
        tmp_from = arcpy.Raster(in_from_ras)
        tmp_to = arcpy.Raster(in_to_ras)

        # calculate change from baseline to latest uav capture
        out_ras = arcpy.ia.ComputeChangeRaster(from_raster=tmp_from,
                                               to_raster=tmp_to,
                                               compute_change_method='CATEGORICAL_DIFFERENCE',
                                               filter_method='CHANGED_PIXELS_ONLY',
                                               define_transition_colors='AVERAGE')

        # save out raster
        out_ras.save(out_change_ras)

    except Exception as e:
        raise e

    return out_change_ras


def extract_uav_change_attrs(
        in_ras: str
) -> list:
    """
    Takes a UAV change raster with table and extracts all
    row information as a list of dictionaries with field names and
    field values as key:value pairs.
    :param in_ras: Path to a UAV change raster.
    :return: List of dictionaries.
    """

    try:
        # init list to hold dictionaries
        tbl_rows = []

        # create list of expected fields and extract
        fields = ['Value', 'Classvalue', 'Class_name', 'Class_From', 'Class_To']
        with arcpy.da.SearchCursor(in_ras, fields) as cursor:
            for row in cursor:
                # build dict based on row
                data = {
                    'Value': row[0],
                    'Classvalue': row[1],
                    'Class_name': row[2],
                    'Class_From': row[3],
                    'Class_To': row[4]
                }

                # append to list
                tbl_rows.append(data)

    except Exception as e:
        raise e

    # check if we got something, error otherwise
    if len(tbl_rows) == 0:
        raise ValueError('No rows in raster table.')

    return tbl_rows


def apply_majority_filter(
    in_ras: str,
    out_ras: str
):
    """
    Takes a classified UAV raster and applies majority filter
    to it to remove speckles.

    :param in_ras: Path to classified UAV raster.
    :param out_ras: Path to output clean classified UAV raster.
    :return: Path of output clean classified UAV raster.
    """

    try:
        # apply a 5x5 majority focal window filter to remove noise
        win = arcpy.sa.NbrRectangle(5, 5, 'CELL')
        tmp_maj = arcpy.sa.FocalStatistics(in_ras, win, 'MAJORITY')

        # save cleaned classified raster
        tmp_maj.save(out_ras)

    except Exception as e:
        raise e

    return out_ras


def apply_shrink_filter(
        in_ras: str,
        chg_attrs: list,
        out_ras: str
):
    """
    Takes a classified UAV raster and applies pixel shrink
    to remove small isolated pixels and more noise.

    :param in_ras: Path to classified UAV raster.
    :param chg_attrs: List of raster value and change name dicts.
    :param out_ras: Path to output clean classified UAV raster.
    :return: Path of output clean classified UAV raster.
    """

    try:
        # extract all non "no change" classes
        shrink_classes = []
        for item in chg_attrs:
            if item['Class_name'] != 'No Change':
                shrink_classes.append(item['Value'])

        if len(shrink_classes) > 0:
            # shrink filtered raster by 1 pixel if we have no change
            tmp_shk = arcpy.sa.Shrink(in_raster=in_ras,
                                      number_cells=1,
                                      zone_values=shrink_classes,
                                      shrink_method='MORPHOLOGICAL')

            # save cleaned shrunk raster
            tmp_shk.save(out_ras)

    except Exception as e:
        raise e

    return out_ras


def append_uav_attrs(
        in_ras: str,
        in_attrs: list
) -> str:
    """
    Append original class columns back on to
    raster attribute table and update rows
    to contain original values. This is needed
    as majority filter strips class information.

    :param in_ras: Path to classified raster.
    :param in_attrs: List of dictionaries containing field, value pairs.
    :return: Path to the input raster.
    """

    # set expected fields with types
    fields = [
        ['Value', 'LONG'],
        ['Classvalue', 'LONG'],
        ['Class_name', 'TEXT'],
        ['Class_From', 'TEXT'],
        ['Class_To', 'TEXT']
    ]

    try:
        # add new fields to raster
        arcpy.management.AddFields(in_table=in_ras,
                                   field_description=fields)

        # iter raster attribute table...
        fields = [f[0] for f in fields]
        with arcpy.da.UpdateCursor(in_ras, fields) as cursor:
            for row in cursor:
                for item in in_attrs:
                    # update row with associated data
                    if row[0] == item['Value']:
                        row[1] = item['Classvalue']
                        row[2] = item['Class_name']
                        row[3] = item['Class_From']
                        row[4] = item['Class_To']

                        # update row and move to next row
                        cursor.updateRow(row)
                        break

    except Exception as e:
        raise e

    return in_ras


def calc_uav_change_areas(
        in_ras: str,
        in_boundary: str,
        out_csv: str
) -> str:
    """
    Calculates the area (hectares) of each
    raster class occurring within a given polygon
    boundary.

    :param in_ras: Path to a classified raster.
    :param in_boundary: Path to a boundary shapefile.
    :param out_csv: Path to output csv of area calculations.
    :return: Path to output csv file.
    """

    try:
        # get number of pixels per class
        arcpy.ia.SummarizeCategoricalRaster(in_raster=in_ras,
                                            out_table=out_csv,
                                            aoi=in_boundary,
                                            aoi_id_field='SID')

        # extract raster cell x, y size
        desc = arcpy.Describe(in_ras)
        band = desc.children[0]
        x_res, y_res = band.meanCellWidth, band.meanCellHeight

        # get cell area and convert to hectares
        cell_area_ha = (x_res * y_res) * 0.0001

        # read as dataframe
        df = pd.read_csv(out_csv)

        # set header
        arcpy.AddMessage('Area (Hectares) per Change Class')

        for field in list(df):
            if field != 'SID':
                # get clean hectraes value
                value_ha = float(df[field] * cell_area_ha)
                value_ha = round(value_ha, 8)

                # assign to field
                df[field] = value_ha

                # get clean field name
                clean_field = field.replace('__', '->').replace('_', ' ')

                # update field name in dataframe
                df = df.rename(columns={field: clean_field})

                # notify user
                arcpy.AddMessage(f'{clean_field}: {value_ha}')

        # export dataframe
        df.to_csv(out_csv)

    except Exception as e:
        raise e

    return out_csv


def detect_epoch_change(
        in_from_ras: str,
        in_mid_ras: str,
        in_to_ras: str,
        out_from_mid_ras: str,
        out_from_to_ras: str,
) -> tuple:
    """

    :param in_from_ras:
    :param in_mid_ras:
    :param in_to_ras:
    :param out_from_mid_ras:
    :param out_from_to_ras:
    :return:
    """

    try:
        # read "from" and "to" fraction rasters
        tmp_from = arcpy.Raster(in_from_ras)
        tmp_mid = arcpy.Raster(in_mid_ras)
        tmp_to = arcpy.Raster(in_to_ras)

        # calculate change "from" to "mid" for fraction rasters
        chg_from_mid = arcpy.ia.ComputeChangeRaster(from_raster=tmp_from,
                                                    to_raster=tmp_mid,
                                                    compute_change_method='DIFFERENCE',
                                                    filter_method='ALL')

        # save it
        chg_from_mid.save(out_from_mid_ras)

        # same again but for "from" to "to" fraction rasters
        chg_from_to = arcpy.ia.ComputeChangeRaster(from_raster=tmp_from,
                                                   to_raster=tmp_to,
                                                   compute_change_method='DIFFERENCE',
                                                   filter_method='ALL')

        # save it
        chg_from_to.save(out_from_to_ras)

    except Exception as e:
        raise e

    return out_from_mid_ras, out_from_to_ras


def validate_frac_dates(
        dates: list,
        from_year: int,
        to_year: int,
        month: int
) -> tuple[int, int, int]:
    """

    :param dates:
    :param from_date:
    :param to_date:
    :return:
    """

    # convert "from" and "to" years and month to string dates
    from_date = f'{from_year}-{str(month).zfill(2)}'
    to_date = f'{to_year}-{str(month).zfill(2)}'

    # check if both in dates, all good if so
    if from_date in dates and to_date in dates:
        return from_year, to_year, month

    # error if issue is with "from" date, cant do much
    if from_date not in dates:
        raise ValueError(f'"From" date {from_date} not in NetCDF. Change "from" year or month.')

    # now try "to" date
    if to_date not in dates:
        # create new date string with roll back year
        new_to_date = f'{to_year - 1}-{str(month).zfill(2)}'

        # roll back a year if missing and warn, else error
        if new_to_date not in dates:
            raise ValueError(f'"To" date {to_date} not in NetCDF. Change "to" year or month.')
        else:
            arcpy.AddWarning(f'"To" date {to_date} was not in NetCDF. Rolled back a year ({to_year - 1}).')
            return from_year, to_year - 1, month

    return from_year, to_year, month


def threshold_via_zscore(
        in_ras: str,
        z: int,
        out_z_pos_ras: str,
        out_z_neg_ras: str
) -> tuple:
    """

    :param in_ras:
    :param z:
    :param out_z_pos_ras:
    :param out_z_neg_ras:
    :return:
    """

    try:
        # read raster
        ras = arcpy.Raster(in_ras)

        # extract stats, mean and stdv
        stats = ras.getStatistics()[0]
        mean = stats['mean']
        stdv = stats['standardDeviation']

        # convert to zscores
        tmp_zsc = (ras - mean) / stdv

        # threshold zscore into pos raster
        ras_pos = arcpy.ia.Con(tmp_zsc > z, 1, 0)
        ras_pos.save(out_z_pos_ras)

        # threshold zscore into neg raster
        ras_neg = arcpy.ia.Con(tmp_zsc < -z, 1, 0)
        ras_neg.save(out_z_neg_ras)

    except Exception as e:
        raise e

    return out_z_pos_ras, out_z_neg_ras


def update_frac_classes(
        in_ras: str
) -> None:
    """

    :param in_ras:
    :return:
    """

    try:
        # get raster fields
        ras_fields = [f.name for f in arcpy.ListFields(in_ras)]

        # check if direction is pos or neg
        direction = None
        for field in ras_fields:
            if 'native' in field:
                if 'p_' in field:
                    direction = 'p_'
                elif 'n_' in field:
                    direction = 'n_'

        # check we got something back
        if direction is None:
            raise ValueError('Could not determine direction from class fields.')

        # prepare expected field names
        fields = ['other', 'native', 'weed']
        fields = [direction + f for f in fields]

        # now we know direction, check all fields in raster
        for field in fields:
            if field not in ras_fields:
                raise ValueError(f'Could not find field {field} in change raster.')

        # add class name field
        arcpy.management.AddField(in_table=in_ras,
                                  field_name='Class_name',
                                  field_type='TEXT')

        # add to end of expected fields
        fields = fields + ['Class_name']

        # update row values based on value
        with arcpy.da.UpdateCursor(in_ras, fields) as cursor:
            for row in cursor:
                if row[0:3] == [0, 0, 0]:
                    row[3] = 'No Change'
                elif row[0:3] == [1, 0, 0]:
                    row[3] = 'Other'
                elif row[0:3] == [0, 1, 0]:
                    row[3] = 'Native'
                elif row[0:3] == [0, 0, 1]:
                    row[3] = 'Weed'
                elif row[0:3] == [1, 1, 0]:
                    row[3] = 'Native & Other'
                elif row[0:3] == [1, 0, 1]:
                    row[3] = 'Weed & Other'
                elif row[0:3] == [0, 1, 1]:
                    row[3] = 'Native & Weed'
                elif row[0:3] == [1, 1, 1]:
                    row[3] = 'Native & Weed & Other'

                # update cursor
                cursor.updateRow(row)

    except Exception as e:
        raise e

    return


def detect_diff_change(
        in_from_ras: str,
        in_to_ras: str,
        out_from_to_ras: str,
) -> str:
    """

    :param in_from_ras:
    :param in_to_ras:
    :param out_from_to_ras:
    :return:
    """

    try:
        # read "from" and "to" fraction rasters
        tmp_from = arcpy.Raster(in_from_ras)
        tmp_to = arcpy.Raster(in_to_ras)

        # calculate change "from" to "to" fraction rasters
        chg_from_to = arcpy.ia.ComputeChangeRaster(from_raster=tmp_from,
                                                   to_raster=tmp_to,
                                                   compute_change_method='DIFFERENCE',
                                                   filter_method='ALL')

        # save it
        chg_from_to.save(out_from_to_ras)

    except Exception as e:
        raise e

    return out_from_to_ras


def calc_frac_change_areas(
        in_ras: str,
        in_boundary: str,
        out_csv: str,
) -> str:
    """

    :param in_ras:
    :param in_boundary:
    :param out_csv:
    :return:
    """

    try:
        # get number of pixels per class for gain
        arcpy.ia.SummarizeCategoricalRaster(in_raster=in_ras,
                                            out_table=out_csv,
                                            aoi=in_boundary,
                                            aoi_id_field='SID')

        # extract raster cell x, y size
        desc = arcpy.Describe(in_ras)
        band = desc.children[0]
        x_res, y_res = band.meanCellWidth, band.meanCellHeight

        # get cell area and convert to hectares
        cell_area_ha = (x_res * y_res) * 0.0001

        # read as dataframe
        df = pd.read_csv(out_csv)

        # set header
        if '_gain_' in in_ras:
            arcpy.AddMessage('Area (Hectares) gained.')
        elif '_loss_' in in_ras:
            arcpy.AddMessage('Area (Hectares) lost.')
        else:
            raise ValueError('Not a gain/loss raster.')

        # for each field in dataframe...
        for field in list(df):
            if field not in ['SID']:
                # get clean hectraes value
                value_ha = float(df[field] * cell_area_ha)
                value_ha = round(value_ha, 8)

                # assign to field
                df[field] = value_ha

                # get clean field name
                clean_field = field.replace('___', ' & ').replace('_', ' ')

                # update field name in dataframe
                df = df.rename(columns={field: clean_field})

                # notify user
                arcpy.AddMessage(f'{clean_field}: {value_ha}')

        # export dataframe
        df.to_csv(out_csv)

    except Exception as e:
        raise e

    return out_csv
