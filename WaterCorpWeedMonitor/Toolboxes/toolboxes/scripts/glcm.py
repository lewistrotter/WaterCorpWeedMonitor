
import os
import arcpy
import numpy as np
#import matplotlib.pyplot as plt

from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


# TODO: update code

def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):
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


# def fast_glcm_mean(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
#     '''
#     calc glcm mean
#     '''
#     h, w = img.shape
#     glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
#     mean = np.zeros((h, w), dtype=np.float32)
#     for i in range(levels):
#         for j in range(levels):
#             mean += glcm[i, j] * i / (levels) ** 2
#
#     return mean
#
#
# def fast_glcm_std(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
#     '''
#     calc glcm std
#     '''
#     h, w = img.shape
#     glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
#     mean = np.zeros((h, w), dtype=np.float32)
#     for i in range(levels):
#         for j in range(levels):
#             mean += glcm[i, j] * i / (levels) ** 2
#
#     std2 = np.zeros((h, w), dtype=np.float32)
#     for i in range(levels):
#         for j in range(levels):
#             std2 += (glcm[i, j] * i - mean) ** 2
#
#     std = np.sqrt(std2)
#     return std
#
#
# def fast_glcm_contrast(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
#     '''
#     calc glcm contrast
#     '''
#     h, w = img.shape
#     glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
#     cont = np.zeros((h, w), dtype=np.float32)
#     for i in range(levels):
#         for j in range(levels):
#             cont += glcm[i, j] * (i - j) ** 2
#
#     return cont
#
#
# def fast_glcm_dissimilarity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
#     '''
#     calc glcm dissimilarity
#     '''
#     h, w = img.shape
#     glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
#     diss = np.zeros((h, w), dtype=np.float32)
#     for i in range(levels):
#         for j in range(levels):
#             diss += glcm[i, j] * np.abs(i - j)
#
#     return diss


# def fast_glcm_homogeneity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
#     '''
#     calc glcm homogeneity
#     '''
#     h, w = img.shape
#     glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
#     homo = np.zeros((h, w), dtype=np.float32)
#     for i in range(levels):
#         for j in range(levels):
#             homo += glcm[i, j] / (1. + (i - j) ** 2)
#
#     return homo


# def fast_glcm_secmoment(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
#     '''
#     calc glcm asm
#     '''
#     h, w = img.shape
#     glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
#     asm = np.zeros((h, w), dtype=np.float32)
#     for i in range(levels):
#         for j in range(levels):
#             asm += glcm[i, j] ** 2
#
#     return asm


# def fast_glcm_energy(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
#     '''
#     calc glcm energy
#     '''
#     h, w = img.shape
#     glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
#     asm = np.zeros((h, w), dtype=np.float32)
#     for i in range(levels):
#         for j in range(levels):
#             asm += glcm[i, j] ** 2
#
#     ene = np.sqrt(asm)
#     return ene


# def fast_glcm_max(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
#     '''
#     calc glcm max
#     '''
#     glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
#     max_ = np.max(glcm, axis=(0, 1))
#     return max_


# def fast_glcm_entropy(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
#     '''
#     calc glcm entropy
#     '''
#     glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
#     pnorm = glcm / np.sum(glcm, axis=(0, 1)) + 1. / ks ** 2
#     ent = np.sum(-pnorm * np.log(pnorm), axis=(0, 1))
#     return ent


# def fast_glcm_specify(img, metric='mean', vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
#     '''
#     calc specified glcm metrics
#     '''
#
#     if metric == 'mean':
#         glcm = fast_glcm_mean(img, vmin, vmax, levels, ks, distance, angle)
#     elif metric == 'std':
#         glcm = fast_glcm_std(img, vmin, vmax, levels, ks, distance, angle)
#     elif metric == 'contrast':
#         glcm = fast_glcm_contrast(img, vmin, vmax, levels, ks, distance, angle)
#     elif metric == 'dissimilarity':
#         glcm = fast_glcm_dissimilarity(img, vmin, vmax, levels, ks, distance, angle)
#     elif metric == 'homogeneity':
#         glcm = fast_glcm_homogeneity(img, vmin, vmax, levels, ks, distance, angle)
#     elif metric == 'secmoment':
#         glcm = fast_glcm_secmoment(img, vmin, vmax, levels, ks, distance, angle)
#     elif metric == 'energy':
#         glcm = fast_glcm_energy(img, vmin, vmax, levels, ks, distance, angle)
#     elif metric == 'maximum':
#         glcm = fast_glcm_max(img, vmin, vmax, levels, ks, distance, angle)
#     elif metric == 'entropy':
#         glcm = fast_glcm_entropy(img, vmin, vmax, levels, ks, distance, angle)
#     else:
#         raise ValueError('Metric not supported.')
#
#     return glcm, metric




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
    elif metric == 'std':
        result_arr = get_glcm_std(glcm_arr, h, w, levels)
    elif metric == 'contrast':
        result_arr = get_glcm_contrast(glcm_arr, h, w, levels)
    elif metric == 'dissimilarity':
        result_arr = get_glcm_dissimilarity(glcm_arr, h, w, levels)
    elif metric == 'homogeneity':
        result_arr = get_glcm_homogeneity(glcm_arr, h, w, levels)
    elif metric == 'secmoment':
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



# create a glcm object
# then pass it in to each func so we dont have to rebuild it each time
# just do a copy in each metric func

# testing
in_ras = arcpy.Raster(r'C:\Users\Lewis\Desktop\new_glcm\gray_subset.tif')

# set mask
# ...

desc = arcpy.Describe(in_ras)
srs = desc.SpatialReference
x_size = in_ras.meanCellWidth
y_size = in_ras.meanCellHeight
ll = arcpy.Point(in_ras.extent.XMin, in_ras.extent.YMin)

in_arr = arcpy.RasterToNumPyArray(in_ras)

h, w = in_arr.shape

metrics = [
    'mean',
    'std',
    'contrast',
    'dissimilarity',
    'homogeneity',
    'secmoment',
    'energy',
    'maximum',
    'entropy'
]

# set levels and kernel size
# levels = 16
# kernel_size = 5
# #
# # generate gray co-occurrence matrix
# glcm_arr = fast_glcm(in_arr,
#                      vmin=0,
#                      vmax=255,
#                      levels=levels,
#                      kernel_size=kernel_size,
#                      distance=1.0,
#                      angle=45.0)
#
# # iterate metrics and create raster
# for metric in metrics:
#     print(f'Processing {metric}...')
#
#     # calculate metric
#     tmp_arr = get_specific_glcm(glcm_arr, metric, h, w, levels=levels, ks=kernel_size)
#
#     # convert back to raster
#     tmp_tex = arcpy.NumPyArrayToRaster(in_array=tmp_arr,
#                                        lower_left_corner=ll,
#                                        x_cell_size=x_size,
#                                        y_cell_size=y_size)
#
#     # add spatial reference system back on
#     arcpy.DefineProjection_management(tmp_tex, srs)
#
#     # export
#     arcpy.env.overwriteOutput = True
#     out_fp = os.path.join(r"C:\Users\Lewis\Desktop\new_glcm\output", f'tx_{metric}_16.tif')
#     tmp_tex.save(out_fp)




#stdv_arr = _fast_glcm_std(glcm_arr, mean_arr, h, w, levels=4)
#cont_arr = _fast_glcm_contrast(glcm_arr, h, w, levels=4)

#glcm_arr = fast_glcm_specify(arr, metric, 0, 255, 4, 5, 1.0, 45.0)
#tex_arr = arcpy.NumPyArrayToRaster(in_array=glcm_arr,
                                   #lower_left_corner=ll,
                                   #x_cell_size=x_size,
                                   #y_cell_size=y_size)

#out_fp = os.path.join(r"C:\Users\Lewis\Desktop\glcm\results", f'tx_mean.tif')
#tmp_tex.save(out_fp)

















# gry = r"C:\Users\Lewis\Desktop\glcm\gry.tif"
#
# in_ras = arcpy.Raster(gry)
#
# desc = arcpy.Describe(in_ras)
# srs = desc.SpatialReference
# extent = desc.Extent
# x_size = in_ras.meanCellWidth
# y_size = in_ras.meanCellHeight
# ll = arcpy.Point(in_ras.extent.XMin, in_ras.extent.YMin)
# nd = desc.noDataValue
#
# arr = arcpy.RasterToNumPyArray(in_ras)
#
# metrics = [
#     'mean',
#     'std',
#     'contrast',
#     'dissimilarity',
#     'homogeneity',
#     'secmoment',
#     'energy',
#     'maximum',
#     'entropy'
# ]
#
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


# not focalwindow has a ia equivalent!

# https://www.l3harrisgeospatial.com/docs/backgroundtexturemetrics.html
# 1st order metrics
# range is just range in focal window 5x5 on non-sliced raster
#out_raster = arcpy.ia.FocalStatistics("gry.tif", "Rectangle 5 5 CELL", "RANGE", "DATA", 90); out_raster.save(r"D:\Work\Curtin\Water Corp Project - General\Development\ArcGIS Pro\Development Project\Development Project.gdb\FocalSt_gry3")

# mean exists, but we do 2nd order (just use 16 classes)
#

# variance... not sure how to calc?

# entropy... how?
#from scipy.special import entr
#entr(kernel_window).sum(axis=1)

# skewness... how?
# scipy.stats.skew
# skew([1, 2, 3, 4, 5])

#with RasterCellIterator({'rasters':[myRas1, outRas1], 'padding': 2}) as rci_padded:
    #for i,j in rci_padded:
        #outRas1[i,j] = (myRas1[i-2,j-2] + myRas1[i-2, j] + myRas1[i-2, j+2] + \
                      #myRas1[i,j-2] + myRas1[i, j] + myRas1[i, j+2] + \
                      #myRas1[i+2,j-2] + myRas1[i+2, j] + myRas1[i+2, j+2]) / 9
#outRas1.save()

# std dev
# this comes out the same as range

# max and min are good

# use a high pass filter

# 2nd order metrics

# use the Laplacian  5x5


# get grayscale raster and slice into 16
#out_raster = arcpy.sa.Slice("gry.tif", 16, "EQUAL_INTERVAL", 1, None, None); out_raster.save(r"D:\Work\Curtin\Water Corp Project - General\Development\ArcGIS Pro\Development Project\Development Project.gdb\Slice_gry2")

# use np linespace to figure out values at 16 intervals and get ceil of array
# labels = np.range(1, 16 + 1)
# values = np.floor(np.linspace(0, 255, 16))

# reclassy into the 16 linspace values
#out_raster = arcpy.sa.Reclassify("Slice_gry2", "Value", "1 10;2 24;3 35;4 4;5 5;6 6;7 7;8 8;9 9;10 10;11 11;12 12;13 13;14 14;15 15;16 16", "DATA"); out_raster.save(r"D:\Work\Curtin\Water Corp Project - General\Development\ArcGIS Pro\Development Project\Development Project.gdb\Reclass_Slic1")




# mean is focalwin 5x5 on sliced 16 class raster
#out_raster = arcpy.ia.FocalStatistics("Slice_gry2", "Rectangle 5 5 CELL", "MEAN", "DATA", 90); out_raster.save(r"D:\Work\Curtin\Water Corp Project - General\Development\ArcGIS Pro\Development Project\Development Project.gdb\FocalSt_Slic3")



# mean


