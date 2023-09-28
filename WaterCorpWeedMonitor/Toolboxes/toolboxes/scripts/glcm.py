
import os
import arcpy
import numpy as np
import warnings

from scipy import ndimage

warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

def build_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):
    '''
    Parameters
    ----------
    img: array_like, shape=(h,w), dtype=np.uint8
        input image
    vmin: int
        minimum value of input image
    vmax: int
        maximum value of input image
    levels: int
        number of grey-levels of GLCM
    kernel_size: int
        Patch size to calculate GLCM around the target pixel
    distance: float
        pixel pair distance offsets [pixel] (1.0, 2.0, and etc.)
    angle: float
        pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)

    Returns
    -------
    Grey-level co-occurrence matrix for each pixels
    shape = (levels, levels, h, w)
    '''

    mi, ma = vmin, vmax
    ks = kernel_size
    h, w = img.shape

    # create sequence of values 0-255 for n levels
    # reclassify raster values depending on where they fall in above sequence (start at 1 to n)
    # converts to matrix
    # calls calc_texture

    # digitize
    bins = np.linspace(mi, ma + 1, levels + 1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance * np.cos(np.deg2rad(angle))
    dy = distance * np.sin(np.deg2rad(-angle))

    # scipy slightly different inputs, modified x and y order and sign
    mat = np.array([[1.0, 0.0, dy], [0.0, 1.0, dx]], dtype=np.float32)
    gl2 = ndimage.affine_transform(gl1, mat, mode='nearest')

    # make glcm
    glcm = np.zeros([levels, levels, h, w], dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1 == i) & (gl2 == j))
            glcm[i, j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i, j] = ndimage.convolve(glcm[i, j], kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def get_glcm_mean(glcm_arr, h, w, levels):
    arr = glcm_arr.copy()
    mean = np.zeros([h, w], dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += arr[i, j] * i / levels ** 2

    return mean


def get_glcm_std(glcm_arr, h, w, levels):
    arr = glcm_arr.copy()
    mean = np.zeros([h, w], dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += arr[i, j] * i / levels ** 2

    std2 = np.zeros([h, w], dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            std2 += (arr[i, j] * i - mean) ** 2

    std = np.sqrt(std2)
    return std


def get_glcm_contrast(glcm_arr, h, w, levels):
    arr = glcm_arr.copy()
    cont = np.zeros([h, w], dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            cont += arr[i, j] * (i - j) ** 2

    return cont


def get_glcm_dissimilarity(glcm_arr, h, w, levels):
    arr = glcm_arr.copy()
    diss = np.zeros([h, w], dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            diss += arr[i, j] * np.abs(i - j)

    return diss


def get_glcm_homogeneity(glcm_arr, h, w, levels):
    arr = glcm_arr.copy()
    homo = np.zeros([h, w], dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            homo += arr[i, j] / (1. + (i - j) ** 2)

    return homo


def get_glcm_secmoment(glcm_arr, h, w, levels):
    arr = glcm_arr.copy()
    asm = np.zeros([h, w], dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            asm += arr[i, j] ** 2

    return asm


def get_glcm_energy(glcm_arr, h, w, levels):
    arr = glcm_arr.copy()
    asm = np.zeros([h, w], dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            asm += arr[i, j] ** 2

    ene = np.sqrt(asm)
    return ene


def get_glcm_max(glcm_arr):
    arr = glcm_arr.copy()
    _max = np.max(arr, axis=(0, 1))

    return _max


def get_glcm_entropy(glcm_arr, ks=5):
    arr = glcm_arr.copy()
    pnorm = arr / np.sum(arr, axis=(0, 1)) + 1. / ks ** 2
    ent = np.sum(-pnorm * np.log(pnorm), axis=(0, 1))

    return ent


def get_specific_glcm(glcm_arr, metric, h, w, levels=8, ks=5):

    if metric == 'mean':
        result_arr = get_glcm_mean(glcm_arr, h, w, levels)
    elif metric == 'variance':
        result_arr = get_glcm_std(glcm_arr, h, w, levels)
    elif metric == 'contrast':
        result_arr = get_glcm_contrast(glcm_arr, h, w, levels)
    elif metric == 'dissimilarity':
        result_arr = get_glcm_dissimilarity(glcm_arr, h, w, levels)
    elif metric == 'homogeneity':
        result_arr = get_glcm_homogeneity(glcm_arr, h, w, levels)
    elif metric == 'secondmoment':
        result_arr = get_glcm_secmoment(glcm_arr, h, w, levels)
    elif metric == 'energy':
        result_arr = get_glcm_energy(glcm_arr, h, w, levels)
    elif metric == 'maximum':
        result_arr = get_glcm_max(glcm_arr)
    elif metric == 'entropy':
        result_arr = get_glcm_entropy(glcm_arr, ks)
    else:
        raise ValueError('Metric not supported.')

    return result_arr


def quick_glcm(
        in_gray_ras: str,
        textures_map: dict,
        levels=18,
        kernel_size=5,
        v_block_size=50
):
    """

    :param in_gray_ras:
    :param textures_map:
    :param levels:
    :param kernel_size:
    :param v_block_size:
    :return:
    """
    
    try:
        # read grayscale raster
        ras = arcpy.Raster(in_gray_ras)

        # get raster info for re-building later
        desc = arcpy.Describe(ras)
        srs = desc.SpatialReference
        x_size = ras.meanCellWidth
        y_size = ras.meanCellHeight
        lowleft = arcpy.Point(ras.extent.XMin, ras.extent.YMin)

        # convert raster to array
        arr = arcpy.RasterToNumPyArray(ras)

        # get h, w of array
        h, w = arr.shape

        # copy texture map keys with empty arrays
        results = {}
        for var in textures_map:
            results[var] = []

        i = 0
        items = []
        for y_s in range(0, h, v_block_size):
            # get right win extent + kernel size buffer
            y_e = y_s + v_block_size + kernel_size

            # left win extent + kernel size buffer, if not first element
            if i != 0:
                y_s = y_s - kernel_size

            # slice array to subset size vertically
            y_slice = slice(y_s, y_e)
            tmp_arr = arr[y_slice, :]

            # append to arrs array and increment
            items.append({i: tmp_arr})
            i += 1

        # set progressor
        arcpy.SetProgressor('step', 'Calculating GLCM textures...', 0, len(items), 1)

        for item in items:
            i = list(item.keys())[0]
            tmp_arr = list(item.values())[0]

            # generate gray co-occurrence matrix
            glcm_arr = build_glcm(tmp_arr,
                                  vmin=0,
                                  vmax=255,
                                  levels=levels,
                                  kernel_size=kernel_size,
                                  distance=1.0,
                                  angle=45.0)

            # iterate texture metrics and create raster
            for metric in textures_map:

                # get h, w of current array (can change)
                h, w = tmp_arr.shape

                # calculate texture metric
                tex_arr = get_specific_glcm(glcm_arr,
                                            metric=metric,
                                            h=h,
                                            w=w,
                                            levels=levels,
                                            ks=kernel_size)

                # remove buffer area depending on block
                if i == 0:
                    tex_arr = tex_arr[:-kernel_size]
                elif i == len(items) - 1:
                    tex_arr = tex_arr[kernel_size:]
                else:
                    tex_arr = tex_arr[kernel_size:-kernel_size]

                # append to current metric map list
                results[metric].append(tex_arr)

            # increment progressor
            arcpy.SetProgressorPosition(i)

        # reset progressor
        arcpy.ResetProgressor()

        # iter each metric...
        for metric in results:
            # combine list of arrays into single 2d arrays
            results[metric] = np.concatenate(results[metric])

            # convert back to raster with new values
            tmp_tex = arcpy.NumPyArrayToRaster(in_array=results[metric],
                                               lower_left_corner=lowleft,
                                               x_cell_size=x_size,
                                               y_cell_size=y_size)

            # get path for current metric and save raster
            out_fp = textures_map[metric]
            #out_fp = os.path.join(r"C:\Users\Lewis\Desktop\new_glcm\output", f'tx_{metric}_gray16_levels_18.tif')
            tmp_tex.save(out_fp)

            # TODO: 1 pixel off...

            # ensure projection is defined
            arcpy.DefineProjection_management(out_fp, srs)



    except Exception as e:
        raise e

    return

