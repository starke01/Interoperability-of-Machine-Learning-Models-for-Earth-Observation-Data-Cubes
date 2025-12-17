import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report)
import time
import joblib
import onnxruntime as ort
import torch
import math
from prediction import(class_dict_)



def write_prediction(parcels, dict_classes, height, width, transform, crs, out_path, nodata = -1,):
    shapes = []
    for _, row in parcels.iterrows():
        fid = int(row["fid"])
        cls = dict_classes.get(fid, None)
        if cls is None:
            continue
        shapes.append((row.geometry, int(cls)))


    out_rast = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype="int16",
        all_touched=False,
    )


    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="int16",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_rast, 1)

    return out_rast



def rasterize_ground_truth(gt_data, class_field, class_to_idx, out_shape, transform, crs_da, nodata =  -1):


    ground_truth = gpd.read_file(gt_data)
    if str(ground_truth.crs) != str(crs_da):
        ground_truth = ground_truth.to_crs(crs_da)

    features_to_rasterize = []
    for _, row in ground_truth.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue

        gt_class = row.get(class_field, None)
        if gt_class is None:
            val = nodata
        else:
            key = str(gt_class)
            if key in class_to_idx:
                val = class_to_idx[key]
            else:
                val = nodata


        features_to_rasterize.append((geometry, int(val)))

    gt = rasterize(shapes = features_to_rasterize, out_shape = out_shape, transform = transform, fill = nodata, dtype = "int16", all_touched = False)
    gt = gt.astype(np.int16)
    return gt





def raster_grid(data, resolution):
    minx, miny, maxx, maxy = data.total_bounds
    width  = math.ceil((maxx - minx) / resolution)
    height = math.ceil((maxy - miny) / resolution)
    transform = from_origin(minx, maxy, resolution, resolution)
    crs = data.crs

    return height, width, transform, crs



def json_load(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_model_artifacts(path, classes_path, scale_path):
    data = json_load(path)

    classes_data = pd.read_parquet(classes_path)
    scaler_data = pd.read_parquet(scale_path)

    label_serier = classes_data["label"]
    label_string = label_serier.astype(str)
    classes = label_string.tolist()
    scaler_data_indexed = scaler_data.set_index("band")
    scaler_df = scaler_data_indexed.loc[data["band_order"]]



    mean = scaler_df["mean"].to_numpy(np.float32)
    scale = scaler_df["scale"].to_numpy(np.float32)
    band_order = list(data["band_order"])
    time_steps = int(data.get("time_steps", 12))

    return {
        "classes": classes,
        "band_order": band_order,
        "mean": mean,
        "scale": scale,
        "T": time_steps,
    }


def load_svm_artifacts(classes_path, features_path, model_config_path):
    classes_data = pd.read_parquet(classes_path)
    
    class_s = classes_data["label"]
    classes =  class_s.astype(str)
    classes = classes.tolist()

    features_p = json_load(features_path)
    config_m = json_load(model_config_path)

    feature_order = features_p["features"]


    return {"classes": classes, "feature_order": feature_order, "config": config_m}





def eval_prediction(pred_vec, valid_mask_vec, ground_truth_vec, classes, nodata = -1):

    mask_valid_gt_and_pred = (valid_mask_vec) & (ground_truth_vec != nodata)


    ground_truth_pred = ground_truth_vec[mask_valid_gt_and_pred].astype(int)

    prediction = pred_vec[mask_valid_gt_and_pred].astype(int)

    all_labels = list(range(len(classes)))

    acc = float(accuracy_score(ground_truth_pred, prediction))
    bacc = float(balanced_accuracy_score(ground_truth_pred, prediction))
    mf1 = float(f1_score(ground_truth_pred, prediction, average = "macro", zero_division = 0))
    confusion_matrix_ = confusion_matrix(ground_truth_pred, prediction, labels = all_labels)


    report = classification_report(ground_truth_pred, prediction, labels = all_labels, target_names = classes, zero_division = 0, output_dict = True)

    stats_correct_pre(confusion_matrix_, classes, level_name="pixel-level")

    mask_valid_gt_and_pred_sum = mask_valid_gt_and_pred.sum()
    mask_valid_gt_and_pred_sum = int(mask_valid_gt_and_pred_sum)

    results = {
        "n_eval": mask_valid_gt_and_pred_sum,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": mf1,
        "confusion_matrix": confusion_matrix_,
        "classification_report": report,
    }

    return results




def stats_correct_pre(confusion_matrix_, classes, level_name: str = "pixel-level"):
 
    total = confusion_matrix_.sum()
    correct = np.trace(confusion_matrix_)
    wrong = total - correct

    if total == 0:
        raise ValueError("total empty")


    correct_p = 100* correct / total
    wrong_p   = 100 * wrong   / total
    print("Total", total, "Correct", correct_p, "Wrong", wrong_p)

    print(f"\n Overall correctness ({level_name})")
   
    p_class_t = confusion_matrix_.sum(axis=1)
    p_class_c = np.diag(confusion_matrix_)
    p_class_w = p_class_t - p_class_c
    p_class_acc = np.divide(
        p_class_c,
        p_class_t,
        out=np.zeros_like(p_class_c, dtype=float),
        where=p_class_t != 0,
    )

    df_stats = pd.DataFrame({
        "class": classes,
        "total": p_class_t.astype(int),
        "correct": p_class_c.astype(int),
        "wrong": p_class_w.astype(int),
        "acc": p_class_acc,
    })

    print(f"\n  Per-class correctness ({level_name}) ")
    print(df_stats.to_string(index=False))





def _ort_session(model_path):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def _load_torchscript(model_path):
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model

def calc_time(t1, t0):
    time_t = t1-t0
    return time_t


def predict_onnx(model_path, data, batch_size, julia_layout=False):

    sess = _ort_session(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name



    #warum up
    if julia_layout:
        data_a = data[0].transpose(1, 0)
        data_t = data_a[None, ...].astype(np.float32)
        for _ in range(3):
            _ = sess.run([output_name], {input_name: data_t})
    else:
        b_number = min(data.shape[0], batch_size)
        data_b = data[:b_number].astype(np.float32)
        for _ in range(3):
            _ = sess.run([output_name], {input_name: data_b})

    preds_v = []
    N = data.shape[0]
    t0 = time.perf_counter()

    if julia_layout:
        print("Julia onnx")
        for i in range(N):
            data_ten = data[i].transpose(1, 0)[None, ...]
            data_ten = data_ten.astype(np.float32)  
            out = sess.run([output_name], {input_name: data_ten})[0]
            scores = np.argmax(out, axis=1)  # (1,)
            first = scores[0]
            pred = int(first)
            preds_v.append(pred)
        t1 = time.perf_counter()
        time_t = calc_time(t1, t0)
        print("onnx inference-loop total  " + str(round(time_t, 4)) + " s")
        preds_v = np.array(preds_v, dtype=np.int32)
        return preds_v
    else:
        for s in range(0, N, batch_size):
            he = s + batch_size
            b_step = min(he, N)
            data_ten = data[s:b_step]
            data_ten = data_ten.astype(np.float32)  
            out = sess.run([output_name], {input_name: data_ten})[0]
            argmax_per_row = np.argmax(out, axis=1)    
            pred = argmax_per_row.astype(np.int32)       
            preds_v.append(pred)
        t1 = time.perf_counter()
        time_t = calc_time(t1,t0)
        print("onnx inference-loop total = " + str(round(time_t, 4)) + " s")
        preds_v = np.concatenate(preds_v, axis=0)
        return preds_v





def predict_onnx_svm(model_path, data, batch_size):
    sess = _ort_session(model_path)
    in_name = sess.get_inputs()[0].name


    #warm up
    if data.shape[0] > 0:
        b_number = min(data.shape[0], batch_size)
        data_b = data[:b_number].astype(np.float32)
        for _ in range(3):
            _ = sess.run(None, {in_name: data_b})

    preds_v = []
    N = data.shape[0]

    t0 = time.perf_counter()
    for s in range(0, N, batch_size):
        he = s + batch_size
        b_step = min(he, N)
        data_arr = data[s:b_step].astype(np.float32)
        outputs = sess.run(None, {in_name: data_arr})
        labels0 = outputs[0]                 
        labels_arr = np.array(labels0)          
        labels_r = labels_arr.ravel()            
        labels  = labels_r.astype(np.int32)   
        preds_v.append(labels)
    t1 = time.perf_counter()
    time_t = calc_time(t1, t0) 
    print(" svm-onnx inference-loop total" + str(round(time_t, 4)) +  "s")
    preds_v = np.concatenate(preds_v, axis=0)
    return preds_v





def predict_torchscript(model_path, data, batch_size):
    model = _load_torchscript(model_path)

    # warm up
    b_number = min(data.shape[0], batch_size)
    if b_number > 0:
        x_ten = torch.from_numpy(data[:b_number]).float()
        with torch.no_grad():
            for _ in range(3):
                model(x_ten)


    preds_v = []
    N = data.shape[0]

    t0 = time.perf_counter()
    for s in range(0, N, batch_size):
        he = s + batch_size
        batch_step = min(he, N)
        chunk = data[s:batch_step]                 
        tensor = torch.from_numpy(chunk)       
        data_numpy = tensor.float()           

        with torch.no_grad():
            logp = model(data_numpy)

        pred_t = torch.argmax(logp, dim=1)   
        pred_cpu = pred_t.cpu()             
        pred_numpy = pred_cpu.numpy()        
        pred_numpy = pred_numpy.astype(np.int32)  
        preds_v.append(pred_numpy)

    t1 = time.perf_counter()
    time_t = calc_time(t1, t0)
    print(" torch inference-loop total  " + str(round(time_t, 4)) + " s")

    pred_vec = np.concatenate(preds_v, axis=0)
    return pred_vec




def predict_svm_native_in_batches(model, data, batch_size = 1024):
   
    N = data.shape[0]
    #warmp up
    b_number = min(N, batch_size)
    
    xb_w = data[:b_number]  
    for _ in range(3):
        _ = model.predict(xb_w)

    preds_v = []
    t0 = time.perf_counter()
    for s in range(0, N, batch_size):
        he = s + batch_size
        b_step = min(he, N)
        data_arr = data[s:b_step]  
        pred = model.predict(data_arr).astype(np.int16) 
        preds_v.append(pred)
    t1 = time.perf_counter()
    time_t = calc_time(t1,t0)

    print(" svm-native inference-loop total " + str(round(time_t, 4)) + " s")

    preds_v = np.concatenate(preds_v, axis=0)
    return preds_v 







def compare_torch_vs_onnx(onnx_path, torch_data, T, D, julia_layout = False, batch = None):

    mod = _load_torchscript(torch_data)
    tensor_data = torch.randn(batch, T, D, dtype=torch.float32)

    with torch.no_grad():
        torch_pre = mod(tensor_data).cpu()
        torch_pre = torch_pre.numpy()

    sess = _ort_session(onnx_path)
    input_name = sess.get_inputs()[0].name

    data_ten = tensor_data.numpy().astype(np.float32)

    if julia_layout:
        data_ten = data_ten.transpose(0, 2, 1)

    pre_onnx = sess.run(None, {input_name: data_ten})[0]  

  
    difference_raw = np.abs(torch_pre - pre_onnx)
    maximal_difference_raw = float(difference_raw.max())
    mean_difference_raw = float(difference_raw.mean())

    max_r = round(maximal_difference_raw, 3)
    mean_r = round(mean_difference_raw, 3)
    print(" ONNX vs Torch — max=" + str(max_r) + ", mean=" + str(mean_r))

 
    pred_torch = np.argmax(torch_pre, axis=1)
    pred_onnx  = np.argmax(pre_onnx,  axis=1)

    number_total = pred_torch.shape[0]
    number_equal = int((pred_torch == pred_onnx).sum())
    proc = 100.0 * number_equal / number_total 
    proc_r = round(proc, 2)
    print("argmax compare " + str(number_equal) + " / " + str(number_total) +  " identical prediction (" + str(proc_r) + "%)")


    result = {
        "max_diff_logprobs": maximal_difference_raw,
        "mean_diff_logprobs": mean_difference_raw,
        "n_equal": number_equal,
        "n_total": number_total,
        "y_torch_logprobs": torch_pre,
        "y_onnx_logprobs": pre_onnx,
    }


    return result





def compare_svm_native_vs_onnx_scores(data, svm_pipeline_pkl, svm_onnx_path, n_samples, batch_size):

    N = data.shape[0]
    num = min(N, n_samples)
    data_arr = data[:num].astype(np.float32, copy=False)

    model = joblib.load(svm_pipeline_pkl)
    scores_native = model.decision_function(data_arr)

    if scores_native.ndim == 1:
        scores_native = scores_native.reshape(-1, 1)

    sess = _ort_session(svm_onnx_path)
    in_name = sess.get_inputs()[0].name

    prediction_vector = []
    for s in range(0, num, batch_size):
        he = s + batch_size
        batch_step = min(he, num)
        batch_dat = data_arr[s:batch_step]
        outputs = sess.run(None, {in_name: batch_dat})
        pred = outputs[1]
        prediction_vector.append(pred)

    scores_onnx = np.concatenate(prediction_vector, axis=0)

    if scores_onnx.ndim == 1:
        scores_onnx = scores_onnx.reshape(-1, 1)

    row_pred = min(scores_native.shape[1], scores_onnx.shape[1])
    pred_row_native = scores_native[:, :row_pred]
    pred_raw_onnx = scores_onnx[:, :row_pred]

    difference = np.abs(pred_row_native - pred_raw_onnx)
    maximal_difference = float(difference.max())
    mean_difference = float(difference.mean())

    max_r = round(maximal_difference, 3)
    mean_r = round(mean_difference, 3)

    print("svm max" + str(max_r) + ", mean=" + str(mean_r))

    pred_native = np.argmax(pred_row_native, axis=1)
    pred_onnx = np.argmax(pred_raw_onnx, axis=1)

    number_total = pred_native.shape[0]
    number_equal = int((pred_native == pred_onnx).sum())
    if number_total > 0:
        procent_equal = 100.0 * number_equal / number_total
    else:
        procent_equal = 0.0

    procent_r = round(procent_equal, 2)
    print("argmax on row data scores from svm. Not the Labels!!!" + str(number_equal) + " / " + str(number_total) + " identische Vorhersagen (" + str(procent_r) + "%)")

    result = {
        "max_diff_scores": maximal_difference,
        "mean_diff_scores": mean_difference,
        "n_equal": number_equal,
        "n_total": number_total,
        "scores_native": scores_native,
        "scores_onnx": scores_onnx,
        "pred_native": pred_native,
        "pred_onnx": pred_onnx,
    }

    return result




def compare_external(local, external_data, local_pred_col = None):

    externel_data = pd.read_parquet(external_data)

    if externel_data.empty:
        raise ValueError("externel data is empty")
    

    column_ = ["fid", local_pred_col]
    external_ = ["fid", "pred"]
    local_small = local[column_]
    local_small = local_small.rename(columns={local_pred_col: "pred"})

    external_small = externel_data[external_]
    external_small = external_small.rename(columns={"pred": "pred_outside"})

    compare = pd.merge(local_small, external_small, on="fid", how="inner", copy=False)

    if len(compare) == 0:
        raise ValueError("no machting data")

    compare["diff"] = compare["pred"] != compare["pred_outside"]

    number_difference = int(compare["diff"].sum())
    number_total  = int(len(compare))

    if number_total == 0:
        raise ValueError("total == 0")
    
    difference_procent  = 100 * number_difference / number_total

    print("Compare", number_difference, "diff from", number_total, "in %", round(difference_procent, 2))

    if number_difference > 0:
        print(compare.loc[compare["diff"], ["fid", "pred", "pred_outside"]])

    return compare




def parcel_level_report(data, preds, classes):

    class_to_id = class_dict_(classes)


    classes_data = data["class_name"].astype(str)
    y_true = classes_data.map(class_to_id)   

    mask_data = y_true.notna()
    mask_data_numpy = mask_data.to_numpy()

    y_true = y_true[mask_data].astype(int)
    y_true = y_true.to_numpy()


    y_pred = preds[mask_data_numpy]

    all_labels = list(range(len(classes)))
    confusion_matrix_ = confusion_matrix(y_true, y_pred, labels=all_labels)

    stats_correct_pre(confusion_matrix_, classes, level_name="parcel-level")

    report = classification_report(y_true, y_pred, labels = all_labels, target_names = classes, digits = 4, zero_division=0)
    print("Parcel-level classification report:\n")
    print(report)





def write_overlay_tif(prediction_array, ground_truth_array, transform, crs, out_path, nodata= None):
    height, weight = prediction_array.shape

    red = np.full((height, weight), 255, dtype=np.uint8)
    green = np.full((height, weight), 255, dtype=np.uint8)
    blue = np.full((height, weight), 255, dtype=np.uint8)

    valid_pred = (prediction_array != nodata)
    valid_gt   = (ground_truth_array   != nodata)
    valid_both = valid_pred & valid_gt

    correct_mask = valid_both & (prediction_array == ground_truth_array)
    wrong_mask   = valid_both & (prediction_array != ground_truth_array)

    valid_sum = int(valid_gt.sum())
    valid_pred_sum = int(valid_pred.sum())
    valid_both_sum = int(valid_both.sum())
    correct_mask_sum = int(correct_mask.sum())
    wrong_mask_sum = int(wrong_mask.sum())

    print(" GT-Pixel:", valid_sum)
    print("Pred-Pixel:", valid_pred_sum)
    print("Vergleichs-Pixel:", valid_both_sum)
    print("Korrekte Pixel :", correct_mask_sum)
    print("Falsche Pixel :", wrong_mask_sum)

    red[wrong_mask] = 255
    green[wrong_mask] = 0
    blue[wrong_mask] = 0
    

    red[correct_mask] = 0
    green[correct_mask] = 0
    blue[correct_mask] = 255



    profile_raster = dict(
        driver="GTiff",
        height=height,
        width=weight,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(out_path, "w", **profile_raster) as rast:
        rast.write(red, 1)
        rast.write(green, 2)
        rast.write(blue, 3)


