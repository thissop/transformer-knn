#NOTE(Jesse): The premise of this script is to train a CNN with the provided prepared training data.
# The inputs are assumed to have been produced using the stage_1 script

training_data_fp = "/home/tkiker/code/GitHub/transformer-knn/data"
training_data_glob_match = "/annotation_*.png" #NOTE(Jesse): This string is appended to training_data_fp and is globbed to locate all matching files under the fp directory.

model_weights_fp = None #NOTE(Jesse): Set to a weights training file for post-train

unet_weights_fn_template = "unet_{}.weights.h5"

a100_partition = False

epochs = 10
steps_per_epoch = 500

evaluation_count_target = 2 ** 12 #NOTE(Jesse): Number of evaluation steps between train epochs. 

#training_data_glob_match = "/*/*_annotation_and_boundary_*.tif"

##
# Do Not Adjust global variables declared below this comment!
##
from os import environ
environ['OPENBLAS_NUM_THREADS'] = '2' #NOTE(Jesse): Insanely, _importing numpy_ will spawn a threadpool of num_cpu threads and this is the 'preferred' mechanism to limit the thread count.

from unet.config import unet_config
from predict_utilities import standardize_inplace

from shutil import copy
from os import remove

from numpy import dot, newaxis, zeros, float32, uint16, uint8, uint64, isnan, concatenate, array_equal
from numpy.random import default_rng
from gc import collect

global_local_tmp_dir = environ.get("LOCAL_TMPDIR")
if global_local_tmp_dir is None:
    global_local_tmp_dir = environ.get("TSE_TMPDIR")

model_training_percentage = 100
model_number = 1

#NOTE(Jesse): The global_ objects below are for multiplrocessing pool workers

global_unet_context = None
global_batch_queue_depth = 2 #NOTE(Jesse): Do not touch

import faulthandler
faulthandler.enable()

from osgeo import gdal
gdal.UseExceptions()
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")
gdal.SetConfigOption("GDAL_CACHEMAX", "0")

#NOTE(Jesse): For debug printing
#job_host = environ.get("HOST")
#if job_host is not None:
#    print(job_host)

user_id = environ.get("USER")
try:
    sm_suffix = "_" + user_id + "_" + environ["SLURM_JOB_ID"]
    sm_suffix += ("_" + environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in environ else ""
except Exception as e:
    sm_suffix = "_local"

if global_local_tmp_dir is not None:
    global_local_tmp_dir += '/' + sm_suffix

def init_load_raster_global_variables(in_unet_context):
    global global_unet_context
    global_unet_context = in_unet_context

def process_load_raster_and_anno_files(in_anno_boun_fp):
    pan_fp = in_anno_boun_fp.replace("annotation_", "pan_")
    ndvi_fp = in_anno_boun_fp.replace("annotation_", "ndvi_")
    anno_boun_fp = in_anno_boun_fp
    if global_local_tmp_dir:
        #NOTE(Jesse): Use the directory and filename as a kind of namespace when copying to local tempdir
        # otherwise, other jobs on the same node might be loading / removing a file with the same name
        # from a different input source (same tile and same name but different date)
        tmp_in_anno_boun_fn = in_anno_boun_fp.replace('/', '_')
        tmp_pan_fn = pan_fp.replace('/', '_')
        tmp_ndvi_fn = ndvi_fp.replace('/', '_')

        anno_boun_fp = copy(in_anno_boun_fp, global_local_tmp_dir + tmp_in_anno_boun_fn)
        pan_fp = copy(pan_fp, global_local_tmp_dir + tmp_pan_fn)
        ndvi_fp = copy(ndvi_fp, global_local_tmp_dir + tmp_ndvi_fn)

    pan_ds = gdal.Open(pan_fp)
    assert pan_ds.RasterCount == 1

    ndvi_ds = gdal.Open(ndvi_fp)
    assert ndvi_ds.RasterCount == 1

    raster_xy = 128
    if pan_ds.RasterXSize < raster_xy or pan_ds.RasterYSize < raster_xy:
        print(f"SKIP: {pan_fp} {pan_ds.RasterXSize, pan_ds.RasterYSize}")
        return None, None, in_anno_boun_fp, None

    for pan_i in range(pan_ds.RasterCount):
        assert pan_ds.GetRasterBand(pan_i + 1).DataType == gdal.GDT_Float32
        assert ndvi_ds.GetRasterBand(pan_i + 1).DataType == gdal.GDT_Float32

        assert pan_ds.RasterXSize == ndvi_ds.RasterXSize, (pan_ds.RasterXSize, ndvi_ds.RasterXSize)
        assert pan_ds.RasterYSize == ndvi_ds.RasterYSize, (pan_ds.RasterYSize, ndvi_ds.RasterYSize)
    
    raster_xyc_size = (pan_ds.RasterYSize, pan_ds.RasterXSize, 2)
    raster =    zeros(raster_xyc_size, dtype=float32)
    anno_boun = zeros((raster_xyc_size[0], raster_xyc_size[1], 1), dtype=uint8)

    pan_ds.GetRasterBand(1).ReadAsArray(buf_obj=raster[..., 0])
    ndvi_ds.GetRasterBand(1).ReadAsArray(buf_obj=raster[..., 1])

    pan_ds = None
    ndvi_ds = None

    ab_ds = gdal.Open(anno_boun_fp)
    assert ab_ds.RasterCount == 1
    assert ab_ds.RasterYSize == raster_xyc_size[0]
    assert ab_ds.RasterXSize == raster_xyc_size[1]
    assert ab_ds.GetRasterBand(1).DataType in (gdal.GDT_UInt16, gdal.GDT_Int16), ab_ds.GetRasterBand(1).DataType

    anno_boun_tmp = zeros((raster_xyc_size[0], raster_xyc_size[1]), dtype=uint8)
    ab_ds.ReadAsArray(buf_obj=anno_boun_tmp)
    ab_ds = None

    anno_boun[anno_boun_tmp == 1] = 1
    anno_boun[anno_boun_tmp == 10] = 10

    assert not isnan(anno_boun).any(), in_anno_boun_fp
    assert not isnan(raster).any(), pan_fp

    is_empty = (anno_boun == 1).sum(dtype=uint64) == 0

    if global_local_tmp_dir:
        remove(anno_boun_fp)
        remove(pan_fp)
        remove(ndvi_fp)

    return raster, anno_boun, in_anno_boun_fp, is_empty


def rene_val_uniform_patch_generator(batch_size, validation_patches, anno_patches, empty_patches):
    rng = default_rng()
    randint = rng.integers

    raster_xy_size = 128
    batch_xy_size = raster_xy_size
    rnd_count = 128

    empty_patches_this_idx = randint(0, 10, rnd_count, dtype=uint8)
    maximum_empty_patch_regions_count_per_map = max(1, batch_size // 16)

    raster_batch = zeros((batch_size, batch_xy_size, batch_xy_size, 2), dtype=float32)
    anno_batch = zeros((batch_size, batch_xy_size, batch_xy_size, 1), dtype=uint8)
    anno_b2 = zeros((batch_size, batch_xy_size // 2, batch_xy_size // 2, 1), dtype=uint8)
    anno_b4 = zeros((batch_size, batch_xy_size // 4, batch_xy_size // 4, 1), dtype=uint8)
    anno_b8 = zeros((batch_size, batch_xy_size // 8, batch_xy_size // 8, 1), dtype=uint8)

    empty_anno_bound = zeros((batch_xy_size, batch_xy_size, 1), dtype=uint8)

    rnd_idx = 0
    while True:
        batch_idx = 0
        empty_patch_regions_count = 0

        while batch_idx < batch_size:
            choice_raster_patches = validation_patches
            #choice_fps = raster_fps

            empty_patch_this_idx = False
            if len(empty_patches) > 0:
                empty_patch_this_idx = empty_patches_this_idx[rnd_idx] == 0 
                if empty_patch_this_idx:
                    if empty_patch_regions_count >= maximum_empty_patch_regions_count_per_map:
                        empty_patch_this_idx = False
                    else:
                        choice_raster_patches = empty_patches
                        #choice_fps = raster_empty_fps

                        empty_patch_regions_count += 1

            skipped = 0
            while True:
                patch_idx = randint(0, len(choice_raster_patches))
                raster = choice_raster_patches[patch_idx]
                
                y = randint(0, raster.shape[0] - batch_xy_size, dtype=uint16)
                x = randint(0, raster.shape[1] - batch_xy_size, dtype=uint16)

                anno_boun_patch = empty_anno_bound if empty_patch_this_idx else anno_patches[patch_idx][y:y + batch_xy_size, x:x + batch_xy_size]
                raster_patch = raster[y:y + batch_xy_size, x:x + batch_xy_size]

                rnd_idx += 1
                if rnd_idx == rnd_count:
                    empty_patches_this_idx[:] = randint(0, 10, rnd_count, dtype=uint8)
                    rnd_idx = 0

                if not empty_patch_this_idx:
                    if (anno_boun_patch == 1).sum(dtype=uint64) == 0:
                        if skipped > 6:
                            break

                        skipped += 1
                        continue

                break

            #assert not any(isnan(batch_pan_ndvi)), tf
            #assert not any(isnan(batch_anno_boun)), tf

            raster_batch[batch_idx] = raster_patch
            #standardize_inplace(raster_batch[batch_idx])
            anno_batch[batch_idx] = anno_boun_patch
            batch_idx += 1

        standardize_inplace(raster_batch)

        anno_b1 = anno_batch
        anno_b2[:] = anno_b1[..., ::2, ::2, :]
        anno_b2[anno_b2 == 10] = 0

        anno_b4[:] = anno_b2[..., ::2, ::2, :]
        anno_b8[:] = anno_b4[..., ::2, ::2, :]
        sb = [anno_b8, anno_b4, anno_b2, anno_b1]

        yield raster_batch, sb#anno_b1


def rene_uniform_with_data_aug_random_patch_generator(batch_size, patches, anno_patches, empty_patches):
    from cv2 import INTER_NEAREST, INTER_AREA, warpAffine#,INTER_LINEAR, INTER_MAX

    rng = default_rng()
    randint = rng.integers
    random = rng.random
    std_nrm = rng.standard_normal

    raster_xy_size = 128
    batch_xy_size = raster_xy_size

    rnd_count = 128

    empty_patches_this_idx = randint(0, 10, rnd_count, dtype=uint8)
    patch_indices = randint(0, len(patches), rnd_count, dtype=uint16)
    empty_patch_indices = randint(0, len(empty_patches), rnd_count, dtype=uint16)
    flip_xy_yes_no = randint(0, 2, (rnd_count, 2), dtype=uint8)
    standardize_patch_yes_no = randint(0, 4, rnd_count, dtype=uint8)
    roll_offsets = randint(0, batch_xy_size, (rnd_count, 2), dtype=uint16)
    roll_xy_yes_no = randint(0, 2, rnd_count, dtype=uint8)
    rotate_yes_no = randint(0, 3, rnd_count, dtype=uint8)
    scale_yes_no = randint(0, 3, rnd_count, dtype=uint8)
    skew_yes_no = randint(0, 3, rnd_count, dtype=uint8)
    affine_yes_no = randint(0, 3, rnd_count, dtype=uint8)

    xy_scale = random((rnd_count, 2), dtype=float32)
    xy_skew = random((rnd_count, 2), dtype=float32)

    raster_batch = zeros((batch_size, batch_xy_size, batch_xy_size, 2), dtype=float32)
    anno_boun_batch = zeros((batch_size, batch_xy_size, batch_xy_size, 1), dtype=uint8)
    anno_b2 = zeros((batch_size, batch_xy_size // 2, batch_xy_size // 2, 1), dtype=uint8)
    anno_b4 = zeros((batch_size, batch_xy_size // 4, batch_xy_size // 4, 1), dtype=uint8)
    anno_b8 = zeros((batch_size, batch_xy_size // 8, batch_xy_size // 8, 1), dtype=uint8)

    batch_gaussian_noise = std_nrm(raster_batch.shape, dtype=float32)
    
    random_buffer = zeros((batch_size, 1, 1, 1), dtype=float32)
    random(out=random_buffer, dtype=float32)

    rotation_matrix = zeros((3, 3), dtype=float32)
    scale_matrix = zeros((3, 3), dtype=float32)
    skew_matrix = zeros((3, 3), dtype=float32)
    transform_matrix = zeros((3, 3), dtype=float32)
    identity_matrix = zeros((3, 3), dtype=float32)

    identity_matrix[0, 0] = 1
    identity_matrix[1, 1] = 1
    identity_matrix[2, 2] = 1

    scale_matrix[:] = identity_matrix
    rotation_matrix[:] = identity_matrix
    skew_matrix[:] = identity_matrix
    transform_matrix[:] = identity_matrix

    empty_anno_bound = zeros((batch_xy_size, batch_xy_size, 1), dtype=uint8)

    maximum_empty_patch_regions_count_per_map = max(1, batch_size // 16)

    half_raster_xy_size = raster_xy_size // 2

    repeat_patched_count = 1

    rnd_idx = 0
    while True:
        empty_patch_regions_count = 0
        batch_idx = 0

        collect()

        while batch_idx < batch_size:
            select_new_patch = batch_idx % max(1, repeat_patched_count) == 0
            if select_new_patch:
                patch_idx = patch_indices[rnd_idx]
                raster = patches[patch_idx]
                anno_boun = anno_patches[patch_idx]
            
                empty_patch_this_idx = False
                if len(empty_patches) > 0:
                    empty_patch_this_idx = empty_patches_this_idx[rnd_idx] == 0 
                    if empty_patch_this_idx:
                        if empty_patch_regions_count >= maximum_empty_patch_regions_count_per_map:
                            empty_patch_this_idx = False
                        else:
                            #choice_fps = raster_empty_fps

                            empty_patch_regions_count += repeat_patched_count

                            patch_idx = empty_patch_indices[rnd_idx]
                            raster = empty_patches[patch_idx]
                            anno_boun = empty_anno_bound

            attempt_count = 0
            while True:
                y = randint(0, raster.shape[0] - batch_xy_size, dtype=uint16)
                x = randint(0, raster.shape[1] - batch_xy_size, dtype=uint16)
                #assert x + batch_xy_size < raster.shape[1]

                affine = affine_yes_no[rnd_idx] == 0
                if affine:
                    skew = False and skew_yes_no[rnd_idx] == 0
                    if skew: #NOTE(Jesse): Busted?
                        x_skew, y_skew = (2 * xy_skew[rnd_idx] - 1) * 0.15

                        skew_matrix[0, 1] = y_skew
                        skew_matrix[1, 0] = x_skew

                        #NOTE(Jesse): Variable skewing is a more general form of rotation, where each dimension is "rotated" separately, and scaled.
                        # For this reason rotation and skewing are not used together.
                        y = uint16(half_raster_xy_size * 0.25)
                        x = uint16(half_raster_xy_size * 0.25)

                        transform_matrix[:] = dot(transform_matrix, skew_matrix)

                    x_scale, y_scale = 1, 1
                    scale = scale_yes_no[rnd_idx] == 0
                    if (not skew) and scale:
                        x_scale, y_scale = 1 + .2 * xy_scale[rnd_idx]

                        #if x_scale < 1:
                        #    x = uint16(x * x_scale)

                        #if y_scale < 1:
                        #    y = uint16(y * y_scale)
                        
                        scale_matrix[0, 0] = x_scale
                        scale_matrix[1, 1] = y_scale

                        transform_matrix[:] = dot(transform_matrix, scale_matrix)

                    warp = not array_equal(transform_matrix, identity_matrix)
                    if warp:
                        warp_xy = (raster.shape[1], raster.shape[0]) #NOTE(Jesse): warp swaps X and Y indices (x, y), not numpy (y, x)
                        raster_warp = warpAffine(raster, transform_matrix[:2], warp_xy, flags=INTER_AREA)
                        anno_boun_warp = warpAffine(anno_boun, transform_matrix[:2], warp_xy, flags=INTER_NEAREST) if not empty_patch_this_idx else empty_anno_bound

                        transform_matrix[:] = identity_matrix

                        if len(raster_warp.shape) < 3:
                            raster_warp = raster_warp[..., newaxis]

                        if len(anno_boun_warp.shape) < 3:
                            anno_boun_warp = anno_boun_warp[..., newaxis]

                        anno_boun_warp[anno_boun_warp > 1] = 10

                        raster = raster_warp
                        anno_boun = anno_boun_warp

                anno_boun_patch = anno_boun[y:y + batch_xy_size, x:x + batch_xy_size] if not empty_patch_this_idx else empty_anno_bound
                raster_patch = raster[y:y + batch_xy_size, x:x + batch_xy_size]
                #assert raster_patch.shape == (raster_xy_size, raster_xy_size, 2)
                
                if False and (raster_patch[..., 0] == 0).sum() > 1000:
                    x = x//2
                    y = y//2

                    anno_boun_patch = anno_boun[y:y + batch_xy_size, x:x + batch_xy_size]
                    raster_patch = raster[y:y + batch_xy_size, x:x + batch_xy_size]

                rnd_idx += 1
                if rnd_idx == rnd_count:
                    empty_patches_this_idx[:] = randint(0, 10, rnd_count, dtype=uint8)

                    roll_offsets[:] = randint(0, batch_xy_size, (rnd_count, 2), dtype=uint16)

                    standardize_patch_yes_no[:] = randint(0, 4, rnd_count, dtype=uint8)
                    roll_xy_yes_no[:] = randint(0, 2, rnd_count, dtype=uint8)
                    flip_xy_yes_no[:] = randint(0, 2, (rnd_count, 2), dtype=uint8)
                    rotate_yes_no[:] = randint(0, 3, rnd_count, dtype=uint8)
                    affine_yes_no[:] = randint(0, 4, rnd_count, dtype=uint8)
                    skew_yes_no[:] = randint(0, 3, rnd_count, dtype=uint8)
                    scale_yes_no[:] = randint(0, 3, rnd_count, dtype=uint8)

                    xy_skew[:] = random((rnd_count, 2), dtype=float32)
                    xy_scale[:] = random((rnd_count, 2), dtype=float32)

                    patch_indices[:] = randint(0, len(patches), rnd_count, dtype=uint16)
                    if empty_patch_indices is not None:
                        empty_patch_indices[:] = randint(0, len(empty_patches), rnd_count, dtype=uint16)

                    rnd_idx = 0

                if not empty_patch_this_idx:
                    #assert anno_boun[..., 0].sum() > 0
                    if (anno_boun_patch == 1).sum(dtype=uint64) == 0:
                        if attempt_count > 6:
                            break

                        patch_idx = randint(0, len(patches), dtype=uint16)
                        raster = patches[patch_idx]
                        anno_boun = anno_patches[patch_idx]

                        attempt_count += 1

                        continue

                if (raster_patch.max() == raster_patch.min() == 0) or ((raster_patch[..., 0] == 0).sum() / raster_xy_size**2) > 0.1:
                    patch_idx = randint(0, len(patches), dtype=uint16)
                    raster = patches[patch_idx]
                    anno_boun = anno_patches[patch_idx]
                    empty_patch_this_idx = False

                    continue

                break

            #assert not any(isnan(batch_pan_ndvi)), tf
            #assert not any(isnan(batch_anno_boun)), tf

            raster_batch[batch_idx] = raster_patch
            #if standardize_patch_yes_no[rnd_idx] > 0:
            #standardize_inplace(raster_batch[batch_idx])
            anno_boun_batch[batch_idx] = anno_boun_patch

            #if roll_xy_yes_no[rnd_idx] == 0:
            #    raster_patch = roll(raster_patch, roll_offsets[rnd_idx], (0,1))
            #    anno_boun_patch = roll(anno_boun_patch, roll_offsets[rnd_idx], (0,1))

            if flip_xy_yes_no[rnd_idx, 0] == 1:
                raster_batch[batch_idx] = raster_batch[batch_idx, ::-1, :]
                anno_boun_batch[batch_idx] = anno_boun_batch[batch_idx, ::-1, :]

            if flip_xy_yes_no[rnd_idx, 1] == 1:
                raster_batch[batch_idx] = raster_batch[batch_idx, :, ::-1]
                anno_boun_batch[batch_idx] = anno_boun_batch[batch_idx, :, ::-1]

            batch_idx += 1

        if False and randint(0, 2, 1, dtype=uint8) == 0:
            no_data_mask = raster_batch[..., 0] == 0

            raster_batch[:] += (batch_gaussian_noise * 0.1)
            
            std_nrm(out=batch_gaussian_noise, dtype=float32)
            
            raster_batch[:] += raster_batch * random_buffer * batch_gaussian_noise * 0.01

            random(out=random_buffer, dtype=float32)
            std_nrm(out=batch_gaussian_noise, dtype=float32)

            raster_batch[no_data_mask] = 0

        standardize_inplace(raster_batch)

        anno_b1 = anno_boun_batch
        anno_b2[:] = anno_b1[..., ::2, ::2, :]
        anno_b2[anno_b2 == 10] = 0

        anno_b4[:] = anno_b2[..., ::2, ::2, :]
        anno_b8[:] = anno_b4[..., ::2, ::2, :]
        ab = [anno_b8, anno_b4, anno_b2, anno_b1]

        yield raster_batch, ab


if __name__ == "__main__":
    from multiprocessing import Pool, set_start_method
    set_start_method("spawn")

    def main():
        from time import time
        start = time() / 60

        from os import remove, mkdir
        from shutil import move
        from os.path import join, isdir, isfile, normpath
        from datetime import date
        from copy import deepcopy

        global model_training_percentage, unet_context
        global global_unet_context
        global training_data_fp, model_weights_fp

        model_training_ratio = model_training_percentage / 100
        assert 0 < model_training_ratio <= 1

        if global_local_tmp_dir:
            if not isdir(global_local_tmp_dir):
                mkdir(global_local_tmp_dir)

        #NOTE(Jesse): Early failure for bad inputs.
        training_data_fp = normpath(training_data_fp)
        assert isdir(training_data_fp), training_data_fp

        model_task_name = training_data_fp.split('/')[-1]

        if model_weights_fp:
            model_weights_fp = normpath(model_weights_fp)
            assert isfile(model_weights_fp), model_weights_fp

        #from json import dump
        from numpy.random import Generator, PCG64DXSM
        from glob import glob
        from gc import collect

        rng = Generator(PCG64DXSM())

        rng_seed = int(rng.integers(1_000_000_000))
        if unet_context.is_deterministic:
            rng_seed = 1

        rng = Generator(PCG64DXSM(seed=rng_seed))

        training_files = glob(training_data_fp + training_data_glob_match)
        training_files_count = len(training_files)
        assert training_files_count > 0

        #NOTE(Jesse): Sort then shuffle gives us a deterministic lever to pull.
        training_files.sort()
        rng.shuffle(training_files)

        #training_files = training_files[:256]
        #training_files_count = 256

        print(f"Loading {training_files_count} items of training data from match: {training_data_fp + training_data_glob_match}")

        frames_count = 0
        empty_frames_count = 0
        training_fps = None

        with Pool(initializer=init_load_raster_global_variables, initargs=(unet_context,)) as p:
            try:
                out_data = p.map(process_load_raster_and_anno_files, training_files)
            except Exception as e:
                print(e)
                return

            training_fps = [deepcopy(b[2]) for b in out_data if not b[3] and b[0] is not None]
            rasters = [b[0] for b in out_data if not b[3] and b[0] is not None]
            anno_bouns = [b[1] for b in out_data if not b[3] and b[0] is not None]

            empty_rasters = []

            training_fps_empty = [deepcopy(b[2]) for b in out_data if b[3] and b[0] is not None]
            empty_frames_count = len(training_fps_empty)
            if empty_frames_count > 0:
                empty_rasters = [b[0] for b in out_data if b[3] and b[0] is not None]
                training_fps += training_fps_empty

        frames_count = len(rasters) - empty_frames_count

        print(f"Found {frames_count} labeled regions and {empty_frames_count} empty regions")

        batch_size = unet_context.batch_size
        batch_input_shape = unet_context.batch_input_shape

        collect()

        training_ratio = 0.9 * model_training_ratio
        validation_ratio = 1 - training_ratio
        assert .999 < training_ratio + validation_ratio <= 1

        training_frames_count = int((frames_count * training_ratio))
        training_empty_frames_count = int((empty_frames_count * training_ratio))

        validation_frames_count = int((frames_count * validation_ratio))
        validation_empty_frames_count = int((empty_frames_count * validation_ratio))

        print(f"Training ratio: {int(training_ratio * 100)}%, Validation ratio: {int(validation_ratio * 100)}%")
        print(f"Training label count: {training_frames_count}, Validation label count: {validation_frames_count}")
        print(f"Empty Training label count: {training_empty_frames_count}, Empty Validation label count: {validation_empty_frames_count}")

        training_raster_frames = rasters[: training_frames_count]
        training_anno_boun_frames = anno_bouns[: training_frames_count]

        validation_raster_frames = rasters[training_frames_count :][: validation_frames_count]
        validation_anno_boun_frames = anno_bouns[training_frames_count :][: validation_frames_count]

        training_raster_empty_frames = empty_rasters[: training_empty_frames_count]
        validation_raster_empty_frames = empty_rasters[training_empty_frames_count: ]

        assert frames_count - 1 <= len(training_raster_frames) + len(validation_raster_frames) <= frames_count + 1
        assert empty_frames_count -1 <= len(training_raster_empty_frames) + len(validation_raster_empty_frames) <= empty_frames_count + 1
        
        training_raster_fps = training_fps[: training_frames_count]
        training_raster_empty_fps = training_fps[training_frames_count + validation_frames_count :][: training_empty_frames_count]

        val_gen = rene_val_uniform_patch_generator(batch_size, validation_raster_frames, validation_anno_boun_frames, validation_raster_empty_frames)
        train_gen = rene_uniform_with_data_aug_random_patch_generator(batch_size, training_raster_frames, training_anno_boun_frames, training_raster_empty_frames)

        debug = False
        if debug:
            from unet.visualize import display_images
            for rb, sb in train_gen:
                up2 = sb[2].repeat(2, axis=1).repeat(2, axis=2)
                up2[up2 == 10] = 1

                up1 = sb[1].repeat(4, axis=1).repeat(4, axis=2)
                up1[up1 == 10] = 1

                up0 = sb[0].repeat(8, axis=1).repeat(8, axis=2)
                up0[up0 == 10] = 1

                display_images(concatenate((rb, sb[3], up2, up1, up0), axis=-1), training_data_fp)

        print("Loading Model API")
        environ["KERAS_BACKEND"] = "jax"

        #NOTE(Jesse): JAX uses these TF_ env variables
        environ["TF_GPU_THREAD_MODE"] = "gpu_private" #NOTE(Jesse): Seperate I/O and Compute CPU thread scheduling.
        environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

        #from keras.callbacks import ModelCheckpoint#, TensorBoard
        from unet.loss import focal_tversky, tversky, accuracy, dice_coef
        import keras as K
        from keras.optimizers import AdamW, SGD, Nadam#, Adadelta, schedules

        UNet_version = unet_context.UNet_version
        raster_bands = unet_context.raster_bands

        K.utils.set_random_seed(rng_seed)
        data_parallel = K.distribution.DataParallel()
        K.distribution.set_distribution(data_parallel)

        if UNet_version == 1:
            from unet.UNet import UNet_v1 as UNet
        elif UNet_version == 2:
            from unet.UNet import UNet_v2 as UNet
        elif UNet_version == 3:
            from unet.AttnUNetMultiInput import attn_reg_train as UNet
            from unet.AttnUNetMultiInput import attn_reg
            
        model = UNet(batch_input_shape, weights_file=model_weights_fp) # e8: final_accuracy: 0.9235 - final_dice_coef: 0.7211 - final_loss: 0.3695 

        base_lr = 0.01
        lr = unet_context.batch_size * base_lr
        cn = 0.125
        cv = 0.01
        wd = 0.01
        #adaDelta = Adadelta(learning_rate=1.0, clipnorm=cn, clipvalue=cv, weight_decay=wd)
        sgd = SGD(learning_rate=lr, momentum=0.9, nesterov=True)#, clipnorm=cn)# clipvalue=cv, weight_decay=wd)
        #nadam = Nadam(learning_rate=lr)#, clipnorm=cn, weight_decay=wd)
        #adamw = AdamW(learning_rate=lr, weight_decay=wd)
        opt = sgd

        v3_prune_str = "_train"
        unet_weights_fn = unet_weights_fn_template.format("_".join(raster_bands) + "_" + date.today().strftime("%d-%b-%Y") + "_v-" + f"{UNet_version}" + f"_model_number-{model_number}" + f"_label_percentage-{model_training_percentage}" + f"_epoch-{epochs}" + f"_steps-{steps_per_epoch}" + "_opt-" + opt.name + "_" + model_task_name + (v3_prune_str if UNet_version == 3 else ""))
        print(unet_weights_fn)

        post_train = False
        if model_weights_fp:
            post_train = True
        else:
            weights_tmp_dir = training_data_fp
            if a100_partition:
                weights_tmp_dir = environ["TSE_TMPDIR"]
                assert isdir(weights_tmp_dir)

            model_weights_fp = join(weights_tmp_dir, unet_weights_fn)

        loss_cfg = tversky
        loss_str_match = "loss"
        m = [dice_coef, accuracy]
        if UNet_version == 3:
            m = [m]
            m *= 4

            loss_cfg = {
                'pred1':focal_tversky,
                'pred2':focal_tversky,
                'pred3':focal_tversky,
                'final': tversky
            }

            loss_str_match = "final_loss"

        print("Start training")
        evaluation_steps = 32
        model_fit = model.fit
        model_evaluate = model.evaluate

        lr_half_count = 0
        previous_loss = 1
        failed_count = 0
        epoch_idx = 0 if not post_train else epochs

        model.compile(optimizer=opt, loss=loss_cfg, metrics=m, jit_compile=True)

        while epoch_idx < epochs:
            collect()
            print(f"Begin training epoch {epoch_idx + 1}.") 
            epoch_idx += 1

            train_val_loss_and_metrics = model_fit(train_gen, shuffle=False, epochs=1, steps_per_epoch=steps_per_epoch).history#, callbacks=cb).history
                
            #print(train_val_loss_and_metrics)
            if isnan(train_val_loss_and_metrics[loss_str_match]):
                print("[ERROR] nan detected during training.  Weights file removed.")
                remove(model_weights_fp)

                return

            print("Eval")
            eval_val_loss_and_metrics = model_evaluate(val_gen, return_dict=True, steps=evaluation_steps)
            print(eval_val_loss_and_metrics)

            val_loss_this_epoch = eval_val_loss_and_metrics[loss_str_match]
            if isnan(val_loss_this_epoch) or (val_loss_this_epoch > previous_loss):
                print(f"Training epoch resulted in worse loss {val_loss_this_epoch} than before {previous_loss}.")

                failed_count += 1
                if failed_count >= 2:
                    print("Halving learning rate.")
                    lr_half_count += 1
                    opt = SGD(learning_rate=lr * 0.5**lr_half_count, momentum=0.9, nesterov=True, clipnorm=cn)
                    #opt = Nadam(learning_rate=lr * 0.5**lr_half_count)#, clipnorm=cn, weight_decay=wd)
                    model.compile(optimizer=opt, loss=loss_cfg, metrics=m, jit_compile=True)

                if failed_count >= 10:
                    print("Early stop training.")
                    break

                continue

            if val_loss_this_epoch < previous_loss:
                print(f"Loss reduced by {previous_loss - val_loss_this_epoch}! Now: {val_loss_this_epoch}")
                previous_loss = val_loss_this_epoch
                failed_count = 0

                model.save_weights(model_weights_fp)

                continue

        if not post_train and a100_partition:
            if isfile(join(training_data_fp, unet_weights_fn)):
                remove(join(training_data_fp, unet_weights_fn))

            move(model_weights_fp, training_data_fp)
            model_weights_fp = join(training_data_fp, unet_weights_fn)

        if UNet_version == 3:
            #NOTE(Jesse): The attn_reg model for training has extra outputs and weights that are not necessary for inferrence and incur a substantial performance cost.
            # So we "prune" them here.
            #
            # Also, TF / Keras globally namespace layer names, so if two identical models are created, they _do not_ share the same layer names
            # if layer names are not explicit provided.  This is stupid and causes all this dumb code to exist for no reason.
            # These models have over a hundred layers and they continue to grow so I don't think it's reasonable to solve it on a per
            # layer basis
            K.utils.clear_session(free_memory=True)
            
            K.utils.set_random_seed(rng_seed)
            model = UNet(batch_input_shape, weights_file=model_weights_fp)

            K.utils.set_random_seed(rng_seed)
            dst_model = attn_reg(batch_input_shape)

            blacklisted_lyrs = ("pred1", "pred2", "pred3", "conv6", "conv7", "conv8")
            base_idx = 0
            for lyr_idx, dst_lyr in enumerate(dst_model.layers):
                while True:
                    src_lyr = model.get_layer(index=lyr_idx + base_idx)
                    for bl_lyrn in blacklisted_lyrs:
                        if src_lyr.name.startswith(bl_lyrn):
                            base_idx += 1
                            src_lyr = model.get_layer(index=lyr_idx + base_idx)
                            break
                    else:
                        break

                src_lyr_wghts = src_lyr.get_weights()
                if len(src_lyr_wghts) == 0:
                    continue

                assert src_lyr.output.shape == dst_lyr.output.shape
                
                dst_lyr.set_weights(src_lyr_wghts)

            unet_weights_fp = join(training_data_fp, unet_weights_fn.replace(v3_prune_str, ""))
            dst_model.save_weights(unet_weights_fp)
            dst_model = None

        model = None

        stop = time() / 60
        print(f"Took {stop - start} minutes to train {unet_weights_fn}.")

        #predict = model.predict_on_batch
        #seterr(all='ignore')
        #with open(join(training_data_fp, 'training_session.json'), 'w') as f:
        #    dump({
        #    'training': calc_stats_for_training_data(predict, training_raster_frames, training_raster_empty_frames, training_anno_boun_frames, training_fps[: training_frames_count], "training"),
        #    'validation': calc_stats_for_training_data(predict, validation_raster_frames, validation_raster_empty_frames, validation_anno_boun_frames, training_fps[training_frames_count:][:validation_frames_count], "validation"),
        #    #'test': calc_stats_for_training_data(predict, test_frames + test_empty_frames, "test"),
        #    }, f)

    unet_context = unet_config()
    unet_context.UNet_version = 3

    from sys import argv
    argc = len(argv)
    if argc >= 3:
        training_data_fp = argv[1]
        training_data_glob_match = argv[2]

        if argc >= 5:
            model_number = int(argv[3])
            model_training_percentage = int(argv[4])

            if argc >= 6:
                unet_context.set_raster_bands(tuple(argv[5:]))

    main()
