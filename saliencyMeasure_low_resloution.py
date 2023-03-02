#! /usr/bin/env python3


from functools import partial
import numpy as np
from numpy import random
from skimage import exposure
from skimage import img_as_float
from skimage.transform import resize
from constants import *

import re, os, glob

EPSILON = np.finfo('float').eps





def normalize(x, method='standard', axis=None):
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def match_hist(image, cdf, bin_centers, nbins=256):
    image = img_as_float(image)
    old_cdf, old_bin = exposure.cumulative_distribution(image,
                                                        nbins)  # Unlike [1], we didn't add small positive number to the histogram
    new_bin = np.interp(old_cdf, cdf, bin_centers)
    out = np.interp(image.ravel(), old_bin, new_bin)
    return out.reshape(image.shape)


def KLD(p, q):
    p = normalize(p, method='sum')
    q = normalize(q, method='sum')
    return np.sum(np.where(p != 0, p * np.log((p + EPSILON) / (q + EPSILON)), 0))


def AUC_Judd(saliency_map, fixation_map, jitter=False):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0;
    tp[-1] = 1
    fp[0] = 0;
    fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)  # Total number of saliency map values above threshold
        tp[k + 1] = (k + 1) / float(n_fix)  # Ratio saliency map values at fixation locations above threshold
        fp[k + 1] = (above_th - k - 1) / float(n_pixels - n_fix)  # Ratio other saliency map values above threshold
    return np.trapz(tp, fp)  # y, x


def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r]  # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:, rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds) + 2)
        fp = np.zeros(len(thresholds) + 2)
        tp[0] = 0;
        tp[-1] = 1
        fp[0] = 0;
        fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k + 1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k + 1] = np.sum(S_rand[:, rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc)  # Average across random splits


def NSS(saliency_map, fixation_map):
    s_map = np.array(saliency_map, copy=False)
    f_map = np.array(fixation_map, copy=False) > 0.5
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape)
    # Normalize saliency map to have zero mean and unit std
    s_map = normalize(s_map, method='standard')
    # Mean saliency value at fixation locations
    return np.mean(s_map[f_map])


def CC(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3,
                      mode='constant')  # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='standard')
    map2 = normalize(map2, method='standard')
    # Compute correlation coefficient
    return np.corrcoef(map1.ravel(), map2.ravel())[0, 1]


def SIM(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3,
                      mode='constant')  # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have values between [0,1] and sum up to 1
    map1 = normalize(map1, method='range')
    map2 = normalize(map2, method='range')
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


#### METRICS --

#  Name: func, symmetric?, second map should be saliency or fixation?
metrics = {
    "AUC_Judd": [AUC_Judd, False, 'fix'],  # Binary fixation map
    "AUC_Borji": [AUC_Borji, False, 'fix'],  #  Binary fixation map
    "NSS": [NSS, False, 'fix'],  #  Binary fixation map
    "CC": [CC, False, 'sal'],  #  Saliency map
    "SIM": [SIM, False, 'sal'],  #  Saliency map
    "KLD": [KLD, False, 'sal']}  #  Saliency map

#  Possible float precision of bin files
dtypes = {16: np.float16,
          32: np.float32,
          64: np.float64}

get_binsalmap_infoRE = re.compile("(\w+_\d{1,2})_(\d+)x(\d+)_(\d+)b")


def get_binsalmap_info(filename):
    name, width, height, dtype = get_binsalmap_infoRE.findall(filename.split(os.sep)[-1])[0]
    width, height, dtype = int(width), int(height), int(dtype)
    return name, width, height


def getSimVal(salmap1, salmap2, fixmap1=None, fixmap2=None):
    values = []

    for metric in keys_order:

        func = metrics[metric][0]
        sim = metrics[metric][1]
        compType = metrics[metric][2]

        if not sim:
            if compType == "fix" and not "NoneType" in [type(fixmap1), type(fixmap2)]:
                # x = func(salmap1, fixmap2)
                # y = func(salmap2, fixmap1)
                # print(x)
                # print(y)
                m = (func(salmap2, fixmap1))
            else:
                m = (func(salmap1, salmap2)
                     + func(salmap2, salmap1)) / 2
        else:
            m = func(salmap1, salmap2)
        values.append(m)
    return values


def uniformSphereSampling(N):
    gr = (1 + np.sqrt(5)) / 2
    ga = 2 * np.pi * (1 - 1 / gr)

    ix = iy = np.arange(N)

    lat = np.arccos(1 - 2 * ix / (N - 1))
    lon = iy * ga
    lon %= 2 * np.pi

    return np.concatenate([lat[:, None], lon[:, None]], axis=1)


if __name__ == "__main__":
    from time import time

    t1 = time()
    #  Similarité metrics to compute and output to file
    keys_order = ['AUC_Judd', 'NSS', 'CC', 'SIM', 'KLD']
    # keys_order = ['AUC_Borji', 'NSS', 'CC', 'SIM', 'KLD']


    OUT_PATH = "./generator_output/"
    SM_PATH = pathToValSaliencyMap
    SP_PATH = pathToValFixationMap



    SAMPLING_TYPE = [  #  Different sampling method to apply to saliency maps
        "Sphere_9999999",  # Too many points
        "Sphere_1256637",  # 100,000 points per steradian
        "Sphere_10000",  # 10,000
        "Sin",  # Sin(height)
        "Equi"  # None
    ]
    SAMPLING_TYPE = SAMPLING_TYPE[-2]  #  Sin weighting by default
    print("SAMPLING_TYPE: ", SAMPLING_TYPE)

    #  Path to vieo saliency maps we wish to compare
    list_img = [k.split('/')[-1].split('P')[1].split('.')[0] for k in glob.glob(os.path.join(OUT_PATH, '*bin'))]

    temp = np.zeros(10)

    std = np.zeros((5, 25))

    for i in range(len(list_img)):
        salmap1_path = SM_PATH + 'HEsalmap_' + list_img[i] + '.bin'
        salmap2_path = OUT_PATH + 'P' + list_img[i] + '.bin'

        scanpath1_path = SP_PATH + 'HEscanpath_' + list_img[i] + '.bin'


        width = 320
        height = 160


        if SAMPLING_TYPE.split("_")[0] == "Sphere":
            print(int(SAMPLING_TYPE.split("_")[1]))
            unifS = uniformSphereSampling(int(SAMPLING_TYPE.split("_")[1]))
            unifS[:, 0] = unifS[:, 0] / np.pi * (height - 1)
            unifS[:, 1] = unifS[:, 1] / (2 * np.pi) * (width - 1)
            unifS = unifS.astype(int)
        elif SAMPLING_TYPE == "Sin":
            VerticalWeighting = np.sin(np.linspace(0, np.pi, height))  #  latitude weighting
        # plt.plot(np.arange(height), VerticalWeighting);plt.show()

        salmap1_file = open(salmap1_path, "rb")
        salmap2_file = open(salmap2_path, "rb")
        fixmap1_file = open(scanpath1_path, "rb")
        #  Load from raw data
        salmap1 = np.fromfile(salmap1_file, count=width * height, dtype=np.float32)
        salmap2 = np.fromfile(salmap2_file, count=width * height, dtype=np.float32)
        fixmap1 = np.fromfile(fixmap1_file, count=width * height, dtype=np.float32)
        salmap1 = salmap1.reshape([height, width])
        salmap2 = salmap2.reshape([height, width])
        fixmap1 = fixmap1.reshape([height, width])
        #  LOAD SM TO np.array
        salmap1_file.close()
        salmap2_file.close()
        fixmap1_file.close()


        #  Apply uniform sphere sampling if specified
        if SAMPLING_TYPE.split("_")[0] == "Sphere":
            salmap1 = salmap1[unifS[:, 0], unifS[:, 1]]
            salmap2 = salmap2[unifS[:, 0], unifS[:, 1]]

            fixmap1 = fixmap1[unifS[:, 0], unifS[:, 1]]
        # Weight saliency maps vertically if specified
        elif SAMPLING_TYPE == "Sin":
            salmap1 = salmap1 * VerticalWeighting[:, None] + EPSILON
            salmap2 = salmap2 * VerticalWeighting[:, None] + EPSILON

        salmap1 = normalize(salmap1, method='sum')
        salmap2 = normalize(salmap2, method='sum')

        # Compute similarity metrics
        values = getSimVal(salmap1, salmap2,
                           fixmap1, fixmap1)
        # Outputs results
        print(i)
        for iVal, val in enumerate(values):
            print("{}, {}".format(keys_order[iVal], val))



