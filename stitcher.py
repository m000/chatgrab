#!/usr/bin/env python3

import json
import operator
import logging
import sys
import numpy
import math
import itertools
from pathlib import Path
from functools import reduce
from PIL import Image, ImageStat, ImageChops

CONFIG = 'stitcher.json'
logger = logging.getLogger()

def image_stats(img, cmpband, cmph):
    w, h = img.size
    if h < cmph:
        logger.critical("Height less than the comparison band height: %d < %d", h, cmph)
        return None

    stats = {'t': {}, 'b': {}}
    #for hoff in range(0, cmph):
    for hoff in range(0, int(h/2)):
        bandt_coords = [0, hoff, w, hoff+cmpband]
        bandt = img.crop(bandt_coords)
        stats['t'][hoff] = band_stats(bandt)
        #print('t', hoff, bandt_coords, stats['t'][hoff])

        bandb_coords = [0, h-cmpband-hoff, w, h-hoff]
        bandb = img.crop(bandb_coords)
        stats['b'][hoff] = band_stats(bandb)
        #print('b', hoff, bandt_coords, stats['b'][hoff])
    return stats

def band_stats(band):
    imgstat = ImageStat.Stat(band)
    return {
        'stddev': numpy.array(imgstat.stddev),
        'histogram': band.histogram(),
    }

def analyze(stats, stats_prev, stddev_distance_threshold, hrms_threshold):
    # calculate crop pairs
    crop_pairs = []

    for t, st in stats['t'].items():
        for b, sb in stats_prev['b'].items():
            # first compare the distance of the stddev vectors - fast & approximate
            stddev_distance = numpy.linalg.norm(st['stddev']-sb['stddev'])
            if stddev_distance > stddev_distance_threshold:
                continue

            # if previous test is passing, compare histogram rms - slow & more accurate
            h = st['histogram']
            hprev = sb['histogram']
            hrms = math.sqrt(reduce(operator.add, map(lambda a,b: (a-b)**2, h, hprev))/len(h))
            if hrms > hrms_threshold:
                continue

            crop_pairs.append((t, b, hrms))
    #return crop_pairs

    # consolidate crop pairs
    seq_start = None
    seq_len = 1
    seq_start_hrms = None
    crop_pairs_filtered = []
    for t, b, hrms in crop_pairs:
        if seq_start is not None and seq_start[0]+seq_len == t and seq_start[1]-seq_len == b: # and seq_start_hrms == hrms:
            logging.debug('Dropping match pair: %s', (t, b, hrms))
            seq_len += 1
        else:
            seq_start = (t, b, hrms)
            seq_len = 1
            seq_start_hrms = hrms
            crop_pairs_filtered.append(seq_start)
            #print((hrms, t, b))
   
    logging.info('Crop pairs [0:10]: %s', crop_pairs_filtered[0:10])
    return crop_pairs_filtered

def imgcrop(img, l=0, t=0, r=0, b=0, save_as=None, show=False):
    w, h = img.size
    cropped = img.crop([l, t, w-r, h-b])
    if show:
        cropped.show()
    if save_as is not None:
        cropped.save(save_as)
    return cropped

def vconcat(images, save_as=None, show=False):
    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths)
    max_height = sum(heights)
    hoff = 0
    img_new = Image.new('RGB', (total_width, max_height))
    for img in images:
        imgw, imgh = img.size
        img_new.paste(img, (0, hoff))
        hoff += imgh
    if show:
        img_new.show()
    if save_as is not None:
        img_new.save(save_as)
    return img_new

def process_batch(batch, batchn, config):
    cmpband, cmph = config['band_parameters']
    stddev_distance_threshold, hrms_threshold = config['cmp_parameters']
    cimg_prev = None
    cimg_stats_prev = None
    stitched = None
    for f in batch:
        img = Image.open(f);
        logger.info("Cropping %s on %s", f, config['initial_crop'])
        cimg = imgcrop(img, *config['initial_crop'], show=True)

        cimg_stats = image_stats(cimg, cmpband, cmph)
        if cimg_prev is not None:
            crop_pairs = analyze(cimg_stats, cimg_stats_prev, stddev_distance_threshold, hrms_threshold)
            if crop_pairs:
                t, b, stddev_distance_threshold = crop_pairs[0]
            else:
                logger.info("No crop pairs for %s", f)
                t, b, stddev_distance_threshold = (0, 0, 0)
            c1 = imgcrop(stitched, b=b+cmpband)
            c2 = imgcrop(cimg, t=t)
            stitched = vconcat([c1, c2])
        else:
            stitched = cimg

        cimg_prev = cimg
        cimg_stats_prev = cimg_stats

    stitched.save(config['output_pattern'] % batchn)

if __name__ == '__main__':
    # setup logging
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)-20s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    with open(CONFIG, encoding='utf-8') as config_file:
        config = json.loads(config_file.read())

    infiles = Path('.').glob(config['input_pattern'])
    if 'reverse_order' in config and config['reverse_order'] is True:
        infiles = itertools.chain(reversed(list(infiles)))
    batch_size = config['batch_size']
    cmpband, cmph = config['band_parameters']
    stddev_distance_threshold, hrms_threshold = config['cmp_parameters']
    logger.info("batch_size=%d cmpband=%d cmph=%d, stddev_distance_threshold=%f hrms_threshold=%f", batch_size, cmpband, cmph, stddev_distance_threshold, hrms_threshold)

    batchn = 0
    batch = []
    batch_keep = []
    batch_add = []
    while True:
        if not batch_keep:
            batch = list(itertools.islice(infiles, batch_size))
            print('a', batch)
        else:
            # overlap with previous batch by one
            batch_add = list(itertools.islice(infiles, batch_size-1))
            batch = batch_keep[-1:] + batch_add
            print('b', batch_add)

        if batch_keep and not batch_add: break

        #if not batch_keep and len(batch) < batch_size: brea
        logger.info("Processing batch %03d: %s", batchn, batch)
        process_batch(batch, batchn, config)

        batch_keep = batch[-1:]
        batchn += 1
