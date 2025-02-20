if __name__ == "__main__":
    exit()

from os import environ
environ['OPENBLAS_NUM_THREADS'] = '1' #NOTE(Jesse): Insanely, _importing numpy_ will spawn a threadpool of num_cpu threads and this is the 'preferred' mechanism to limit the thread count.

from numpy import nan, float32, uint8, zeros, nanmean, mean, std, nanstd, ceil, squeeze, maximum, seterr, isnan
from gc import collect
seterr(all='raise')

def standardize_inplace(i):
    assert i.dtype == float32, i.dtype
    
    has_nans = i == 0
    no_data_count = has_nans.sum()

    if no_data_count == 0:
        for idx in range(i.shape[-1]):
            f = i[..., idx]
            f[:] = (f - mean(f)) / (std(f) + 1e-9) #NOTE(Jesse): There are some blocks of raster texels which are same valued and NOT no-data.
    else:
        if no_data_count < i.size:
            i[has_nans] = nan

            for idx in range(i.shape[-1]):
                f = i[..., idx]
                f[:] = (f - nanmean(f)) / (nanstd(f) + 1e-9) #NOTE(Jesse): There are some blocks of raster texels which are same valued and NOT no-data.

        i[has_nans] = 0

    #assert not isnan(i).any()

def predict(batch_predict, raster, unet_ctx):
    batch_size = unet_ctx.batch_size
    batch_input_shape = unet_ctx.batch_input_shape
    #unet_version = unet_ctx.UNet_version

    shape_xy = batch_input_shape[0]
    step_xy = unet_ctx.step_xy

    batch = zeros((batch_size, *batch_input_shape), dtype=float32)
    end_patch = zeros(batch_input_shape, dtype=float32)

    if len(raster.shape) != 3:
        raster = raster.reshape((*raster.shape, 1))

    tile_y = raster.shape[0]
    tile_x = raster.shape[1]
    out_predictions = zeros((tile_y, tile_x), dtype=uint8)

    #TODO(Jesse): Under estimate because each batch is comprised of a window size of shape_xy^2, but they overlap each other
    # by 16 meters worth of pixels, so it's not just a matter of carving the tile into independent blocks as done here.
    est_total_batch_count = ceil((tile_y * tile_x) / (step_xy * step_xy * batch_size))

    y0_out = x0_out = batch_idx = 0
    y1_out = x1_out = shape_xy
    raster_xy_shape = (shape_xy, shape_xy)
    batch_count = 0
    last_percent_done = 0
    for y0 in range(0, tile_y, step_xy):
        y1 = min(y0 + shape_xy, tile_y)

        collect()
        for x0 in range(0, tile_x, step_xy):
            x1 = min(x0 + shape_xy, tile_x)

            r = raster[y0:y1, x0:x1] #NOTE(Jesse): Assumes channels are Y,X,N where N is bands.
            if r.shape[:2] != raster_xy_shape:
                #NOTE(Jesse): For some reason resize is acting strangely, so reimplement logic ourselves.

                end_patch.fill(0)
                end_patch[:r.shape[0], :r.shape[1]] = r
                r = end_patch

            batch[batch_idx] = r

            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0

                for b in batch:
                    standardize_inplace(b)

                predictions = squeeze(batch_predict(batch)) * 100

                batch_count += 1
                percent_done = int((batch_count / est_total_batch_count) * 100)
                if percent_done > last_percent_done:
                    last_percent_done = percent_done

                    if percent_done % 10 == 0:
                        print(f"{percent_done}% done", end=", ", flush=True)

                for p in predictions:
                    p = p.astype(uint8)
                    op = out_predictions[y0_out:y1_out, x0_out:x1_out]

                    if p.shape != op.shape: #NOTE(Jesse): Handle fractional step patches
                        p = p[:op.shape[0], :op.shape[1]]

                    op[:] = maximum(op, p)
                    p = None

                    x0_out += step_xy
                    x1_out = min(x0_out + shape_xy, tile_x)
                    assert x0_out != x1_out
                    if x0_out >= tile_x:
                        x0_out = 0
                        x1_out = shape_xy

                        y0_out += step_xy
                        y1_out = min(y0_out + shape_xy, tile_y)
                        assert y1_out != y0_out

    if batch_idx > 0: #NOTE(Jesse): Process partial final 'batch'
        #NOTE(Jesse): Simple, deterministic, "best effort" filling of partial batch to fill in the remaining batch with locally relevant pixel data
        for _ in range(batch_idx * 2 + (batch_size - batch_idx)): #NOTE(Jesse): "rewind" back through the tile
            x0 -= step_xy
            if x0 < 0:
                x0 = tile_x - shape_xy
                y0 -= step_xy
                if y0 < 0:
                    y0 = 0
                    x0 = 0
                    break

        for b_idx in range(batch_idx, batch_size, 1):
            r = raster[y0:y0 + shape_xy, x0:x0 + shape_xy]
            if r.shape[:2] != raster_xy_shape:
                end_patch.fill(0)
                end_patch[:r.shape[0], :r.shape[1]] = r
                r = end_patch

            batch[b_idx] = r

            x0 += step_xy
            if x0 >= tile_x:
                x0 = 0
                y0 += step_xy
                if y0 + step_xy > tile_y:
                    if b_idx < batch_size - 1:
                        batch[b_idx + 1:] = 0 #NOTE(Jesse): At this point, give up and set the remaining patches to no data.  If this happens, the tile input size is likely too small wrt batch size.
                    break

        #NOTE(Jesse): Process the whole batch as opposed to only the relevant parts to prevent CUDA recompiling a whole new UNET
        # Then use only the relevant outputs.
        #if unet_version == 3:
        #    standardize_inplace(batch)
        #else:
        for b in batch:
            standardize_inplace(b)

        predictions = squeeze(batch_predict(batch))[:batch_idx] * 100

        for p in predictions:
            p = p.astype(uint8)
            op = out_predictions[y0_out:y1_out, x0_out:x1_out]

            if p.shape != op.shape: #NOTE(Jesse): Handle fractional step patches
                p = p[:op.shape[0], :op.shape[1]]

            op[:] = maximum(op, p)
            p = None

            x0_out += step_xy
            x1_out = min(x0_out + shape_xy, tile_x)
            assert x0_out != x1_out
            if x0_out >= tile_x:
                x0_out = 0
                x1_out = shape_xy

                y0_out += step_xy
                y1_out = min(y0_out + shape_xy, tile_y)
                assert y1_out != y0_out

    return out_predictions


def DEPRECATED_calc_stats_for_training_data(predict_on_batch, rasters, rasters_empty, anno_bounds, training_fps, group: str):
    from osgeo import gdal
    from unet.config import raster_shape
    from numpy import bool_, where, uint64
    from os import isfile, remove

    sensitivity = lambda tp, fn: tp / (tp + fn)
    specificity = lambda tn, fp: tn / (tn + fp)
    precision   = lambda tp, fp: tp / (tp + fp)
    accuracy    = lambda tp, tn, fp, fn: (tp + tn) / (tp + tn + fp + fn)
    miss_rate   = lambda fn, tp: fn / (fn + tp)
    fall_out    = lambda fp, tn: fp / (fp + tn)

    def tversky_np(y_true, y_pred, alpha=0.6, smooth=10):
        #NOTE(Jesse): The choice of 10 as the smooth parameter is to prevent floating point inprecision from losing it altogether if it's too low.
        # At 64 batches * 256 * 256 pixels ~= 4 million, the smallest delta in value is on the order of 1, so this gives us some headroom.
        # At the cost of lengthing the smoothing.
        y_t = y_true == 1
        y_w = y_true == 10
        y_w[y_w == 0] = 1

        p0 = y_pred  # proba that voxels are class i
        p1 = 1 - y_pred  # proba that voxels are not class i
        g0 = y_t
        g1 = 1 - y_t

        tp = sum(p0 * g0)
        #tn = sum(p1 * g1)

        fp = alpha * sum(y_w * p0 * g1)
        fn = (1.0 - alpha) * sum(y_w * p1 * g0)

        numerator = tp
        denominator = tp + fp + fn
        
        score = (numerator + smooth) / (denominator + smooth)
        return 1.0 - score

    def focal_tversky_np(y_true, y_pred, alpha):
        loss = tversky_np(y_true, y_pred, alpha)
        return loss**0.75

    nn_mem_ds = gdal.GetDriverByName("MEM").Create('', ysize=raster_shape[0], xsize=raster_shape[1], bands=1, eType=gdal.GDT_Byte)
    assert nn_mem_ds

    nn_mem_ds.GetRasterBand(1).SetNoDataValue(0)

    metrics = [None] * rasters.shape[0]
    for m_i in range(rasters.shape[0]):
        raster, anno_bound, fp = rasters[m_i], anno_bounds[m_i], training_fps[m_i]
        anno = anno_bound[..., 0].astype(bool_)

        fqfp = fp
        pred = predict(predict_on_batch, raster)
        assert pred.shape == raster.shape[:-1]

        pred_01 = where(pred > 50, 1, 0).astype(bool_)

        pred_p = pred_01
        pred_n = ~pred_01

        anno_p = anno
        anno_n = ~anno

        tp = sum(pred_p & anno_p, dtype=uint64)
        tn = sum(pred_n & anno_n, dtype=uint64)

        fp = sum(pred_p & anno_n, dtype=uint64)
        fn = sum(pred_n & anno_p, dtype=uint64)

        t_loss = tversky_np(anno_bound, pred_01)
        acc = accuracy(tp, tn, fp, fn)
        spec = specificity(tn, fp)
        sen = sensitivity(tp, fn)
        prec = precision(tp, fp)
        mr = miss_rate(fn, tp)
        fo = fall_out(fp, tn)

        balanced_accuracy = (sen + spec) * 0.5

        print(fqfp, t_loss)

        ds = gdal.Open(fqfp)
        assert ds, fqfp

        nn_mem_ds.SetGeoTransform(ds.GetGeoTransform())
        nn_mem_ds.SetProjection(ds.GetProjection())
        nn_mem_ds.GetRasterBand(1).WriteArray(pred_01)
        ds = None

        fqfp_split = fqfp.split("/")
        base_fp = "/".join(fqfp_split[:-1])
        file_name = fqfp_split[-1].replace("_annotation_and_boundary", "")

        #NOTE(Jesse): Clear old group entries from old runs.
        groups = ("train", "validate", "test")
        for g in groups:
            g_fp = base_fp + f"/{file_name}_{g}.txt"
            if isfile(g_fp):
                remove(g_fp)

        open(base_fp + f"/{file_name}_{group}.txt", 'a').close()

        seg_fp = base_fp + f'/{file_name}_NN_classification.tif'
        gtiff_create_options = ['COMPRESS=DEFLATE', 'Tiled=YES', 'NUM_THREADS=ALL_CPUS', 'SPARSE_OK=True']
        nn_disk_ds = gdal.GetDriverByName('GTiff').CreateCopy(seg_fp, nn_mem_ds, options=gtiff_create_options)
        nn_disk_ds = None

        metrics[m_i] = {"src raster filepath": fqfp, 
                    "seg fp": seg_fp,
                    "true positives": int(tp),
                    "false positives": int(fp),
                    "true negatives": int(tn),
                    "false negatives": int(fn),
                    "tversky loss": float(t_loss) if not isnan(t_loss) else -1, 
                    "accuracy": float(acc) if not isnan(acc) else -1, 
                    "specificity": float(spec) if not isnan(spec) else -1, 
                    "sensitivity": float(sen) if not isnan(sen) else -1, 
                    "precision": float(prec) if not isnan(prec) else -1, 
                    "miss rate": float(mr) if not isnan(mr) else -1, 
                    "fall out": float(fo) if not isnan(fo) else -1, 
                    "balanced accuray": float(balanced_accuracy) if not isnan(balanced_accuracy) else -1}

    nn_mem_ds = None
    return metrics