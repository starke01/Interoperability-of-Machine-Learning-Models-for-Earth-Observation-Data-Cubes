import os, warnings, math
import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

import time
import joblib


from prediction_helper import (write_prediction, load_model_artifacts, load_svm_artifacts, 
                               eval_prediction, compare_svm_native_vs_onnx_scores, 
                               predict_onnx, predict_onnx_svm, predict_torchscript, 
                               predict_svm_native_in_batches, compare_torch_vs_onnx, compare_external, parcel_level_report, write_overlay_tif, 
                               rasterize_ground_truth, raster_grid)

#config 

inference_data_file = "..inference_data.."
shp_path = "..ground_truth_geojson.."


##tempcnn
model_config_path = "..model_config.."
model_pt_path     = "..pt_model.."
classes_py_tempcnn = "..classes_python.."
scaler = "..scaler_python.."
onnx_model_path = "..onnx_model_path.."

##svm
svm_pipeline_pkl = "..svm_pipline.."
svm_onnx_path = "..svm_onnx.."
svm_artifacts_dir = "..artifacts_directory.."
classes_svm_py = "..classes_svm.."
features_svm = "..features_svm.."
model_config_svm = "..model_config_json.."

##outputs
out_tif = "..output_tif.."
overlay_tif = "..overlay_tif.."
prediction_parquet_path = "..prediction_for_safe"


#externel
external_prediction_path = "..external_data_for_comparsion"

out_dir = os.path.dirname(out_tif) if os.path.dirname(out_tif) else "."
os.makedirs(out_dir, exist_ok=True)



# config: what soudl be used
use_onnx = False          
use_onnx_julia_format = False   
use_onnx_svm = False       
use_native_svm = False 


####
def parameter():
    return {"target_epsg" : 3857,
            "resolution_m" : 20, 
            "batch_size"   : 1024,
            "nodata_value": -1, 
            "gt_class"  : "class_name", 
            "n_samples": 2000}





def read_aoi_geojson(path_aoi, target_epsg):
    aoi_data = gpd.read_file(path_aoi)
    if aoi_data.crs.to_epsg() != target_epsg:
        aoi_data = aoi_data.to_crs(target_epsg)
    if "fid" not in aoi_data.columns:
        aoi_data = aoi_data.reset_index(drop=True)
        length_aoi = len(aoi_data)
        aoi_data["fid"] = np.arange(1, length_aoi + 1, dtype=int)
    return aoi_data




def class_dict_ (classes):
    class_index = list(enumerate(classes))
    class_dict = {}
    for i, label in class_index:
        label_str = str(label)
        values = i
        class_dict[label_str] = values
    return class_dict 



def main():

    # AOI einlesen
    target_epsg = parameter()["target_epsg"]
    resolution_m = parameter()["resolution_m"]
    nodata_value = parameter()["nodata_value"]
    batch_size = parameter()["batch_size"]
    gt_class = parameter()["gt_class"]
    n_samples = parameter()["n_samples"]

    aoi = read_aoi_geojson(shp_path, target_epsg)



    if use_onnx_svm:
        art_svm = load_svm_artifacts(classes_svm_py, features_svm, model_config_svm)
        feature_order = art_svm["feature_order"]
        classes = art_svm["classes"]


        inference_data = pd.read_parquet(inference_data_file)

        features = inference_data[feature_order]
        featues_numeric = features.apply(pd.to_numeric)
        features_float32 = featues_numeric.astype(np.float32)
        data_array = features_float32.values


        compare_svm_native_vs_onnx_scores(data_array, svm_pipeline_pkl = svm_pipeline_pkl, svm_onnx_path = svm_onnx_path, n_samples = n_samples, batch_size = batch_size)


        preds = predict_onnx_svm(svm_onnx_path, data_array, batch_size=batch_size)

        parcel_level_report(inference_data, preds, classes)

        pred_labels = []
        for i in preds:
            pred_labels.append(classes[i])

        df_prediction = pd.DataFrame({
            "fid": inference_data["fid"].to_numpy(int),
            "pred": preds,
            "pred_label": pred_labels,
        })



        df_prediction.to_parquet(prediction_parquet_path, index=False)


        height, width, transform, crs = raster_grid(aoi, resolution_m)

        fid_s = df_prediction["fid"].astype(int)
        cls_s = df_prediction["pred"].astype(int)
        pair = zip(fid_s, cls_s)
        dict_pair = dict(pair)


        prediction_array = write_prediction(aoi, dict_pair, height, width, transform, crs, out_tif, nodata=nodata_value)


        class_dict = class_dict_(classes)

        ground_truth_array = rasterize_ground_truth(shp_path, gt_class, class_dict, (height, width), transform, crs, nodata=nodata_value)

        pred_vec = prediction_array.reshape(-1)
        gt_vec = ground_truth_array.reshape(-1)
        valid_mask_vec = (pred_vec != nodata_value)
        eval_res = eval_prediction(pred_vec, valid_mask_vec, gt_vec, classes, nodata=nodata_value)

        print(
            "acc = " + str(round(eval_res["accuracy"], 4)),
            "bal_acc = " + str(round(eval_res["balanced_accuracy"], 4)),
            "macro_f1 =" + str(round(eval_res["macro_f1"], 4)))


        write_overlay_tif(prediction_array, ground_truth_array, transform, crs, overlay_tif, nodata=nodata_value)

        compare_external(df_prediction, external_prediction_path, local_pred_col="pred")
        return



    if use_native_svm:

        model = joblib.load(svm_pipeline_pkl)

        feature_order = list(model.named_steps["scaler"].feature_names_in_)
        classes = list(model.label_encoder.classes_)


        inference_data = pd.read_parquet(inference_data_file)
       

        data_a = inference_data[feature_order]
        data_a = data_a.apply(pd.to_numeric, errors="coerce")

        data_array = data_a.to_numpy(dtype = np.float32)  
        mask = np.isfinite(data_array).all(axis=1)    

        data_array = data_array[mask]                    

        batch_size = parameter()["batch_size"]
        pred_ids = predict_svm_native_in_batches(model, data_array, batch_size = batch_size)


        full_preds = np.full(len(inference_data), nodata_value, dtype = np.int16)
        full_preds[mask] = pred_ids

        parcel_level_report(inference_data, full_preds, classes)

        pred_labels = []
        for i in full_preds:
            if i != nodata_value:
                pred_labels.append(classes[i])
            else:
                pred_labels.append("")

        df_pred = pd.DataFrame({
            "fid": inference_data["fid"].to_numpy(int),
            "pred": full_preds,
            "pred_label": pred_labels,
        })


        df_pred.to_parquet(prediction_parquet_path, index=False)

        height, width, transform, crs = raster_grid(aoi, resolution_m)

        fid_class = {}

        for fid, clas__ in zip(df_pred["fid"], df_pred["pred"]):
            fid_int = int(fid)
            class_int = int(clas__)
            if class_int != nodata_value:
                fid_class[fid_int] = class_int


        prediction_array = write_prediction(aoi, fid_class, height, width, transform, crs, out_tif, nodata=nodata_value)

        class_dict = class_dict_(classes)

        ground_truth_array = rasterize_ground_truth(shp_path, gt_class, class_dict, (height, width), transform, crs, nodata=nodata_value)

        pred_vec = prediction_array.reshape(-1)
        gt_vec = ground_truth_array.reshape(-1)
        valid_mask_vec = (pred_vec != nodata_value)
        eval_res = eval_prediction(pred_vec, valid_mask_vec, gt_vec, classes, nodata=nodata_value)


        print(
            "acc = " + str(round(eval_res["accuracy"], 4)),
            "bal_acc = " + str(round(eval_res["balanced_accuracy"], 4)),
            "macro_f1 = " + str(round(eval_res["macro_f1"], 4)),
        )


        write_overlay_tif(prediction_array, ground_truth_array, transform, crs, overlay_tif, nodata = nodata_value)

        compare_external(df_pred, external_prediction_path, local_pred_col = "pred")

        return




    #tempcnn prediction  
    inference_data = pd.read_parquet(inference_data_file)
    art = load_model_artifacts(model_config_path, classes_py_tempcnn, scaler)
    band_order = art["band_order"]
    mean = art["mean"]
    scale = art["scale"]
    t_expected = art["T"]
    classes = art["classes"]
    band_lenght = len(band_order)
    pacel_length = len(inference_data)
   

    data_list = []

    for band in band_order:
        for time in range(1, t_expected + 1):
            col = f"{band}_T{time}"
            col_z = inference_data[col].to_numpy(np.float32)
            data_list.append(col_z)


    data_tensor = np.stack(data_list, axis=1)  
    data_tensor_reshape = data_tensor.reshape(pacel_length, band_lenght, t_expected)
    data_tensor = data_tensor_reshape.transpose(0, 2, 1)

    data_tensor = (data_tensor - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)

    compare_torch_vs_onnx(onnx_path = onnx_model_path, torch_data = model_pt_path, T = t_expected, D = band_lenght, julia_layout = use_onnx_julia_format, batch = 1024)

    if use_onnx:
        prediction = predict_onnx(onnx_model_path, data_tensor, batch_size, julia_layout = use_onnx_julia_format)
    else:
        prediction = predict_torchscript(model_pt_path, data_tensor, batch_size)

    pred_labels = []
    for i in prediction:
        pred_labels.append(classes[i])

    df_pred = pd.DataFrame({
        "fid": inference_data["fid"].to_numpy(dtype=int),
        "pred": prediction,
        "pred_label": pred_labels,
    })


    parcel_level_report(inference_data, prediction, classes)
    df_pred.to_parquet(prediction_parquet_path, index = False)


    height, width, transform, crs = raster_grid(aoi, resolution_m)
    fid_s = df_pred["fid"].astype(int)
    cls_s = df_pred["pred"].astype(int)
    pair = zip(fid_s, cls_s)
    dict_pair = dict(pair)



    prediction_array = write_prediction(aoi, dict_pair, height, width, transform, crs, out_tif, nodata=nodata_value)

    class_dict = class_dict_(classes)

    ground_truth_array = rasterize_ground_truth(shp_path, gt_class, class_dict, (height, width), transform, crs, nodata=nodata_value)


    pred_vec = prediction_array.reshape(-1)
    gt_vec = ground_truth_array.reshape(-1)
    valid_mask_vec = (pred_vec != nodata_value)
    eval_res = eval_prediction(pred_vec, valid_mask_vec, gt_vec, classes, nodata = nodata_value)

    print(
        "acc = " + str(round(eval_res["accuracy"], 4)),
        "bal_acc = " + str(round(eval_res["balanced_accuracy"], 4)),
        "macro_f1 = " + str(round(eval_res["macro_f1"], 4)),
    )

    write_overlay_tif(prediction_array, ground_truth_array, transform, crs, overlay_tif, nodata = nodata_value)


    compare_external(df_pred, external_prediction_path, local_pred_col="pred")



if __name__ == "__main__":
    main()
