import re
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def identify_bands(data, pattern):
    feature_list = []
    feature_columns = data.columns
    for cols in feature_columns:
        c = str(cols)
        if re.match(pattern, c):
            feature_list.append(cols)
    return feature_list



def svm_artifacts(model, feature_order, engine_for_parquet=None):
    classes_file = "class_levels_from_python_svm.parquet"
    scaler_params_file = "scaler_params_from_python_svm.parquet"


    scaler = model.named_steps["scaler"]
    svm = model.named_steps["svm"]

    # mean
    mean_ = scaler.mean_
    mean_arr = mean_.astype(float)
    mean_list = mean_arr.tolist()

    # scale
    scaler_ = scaler.scale_
    scale_arr = scaler_.astype(float)
    scale_list = scale_arr.tolist()

    scaler_params = {
        "feature": feature_order,
        "mean": mean_list,
        "scale": scale_list
    }

    scaler_params_data_frame = pd.DataFrame(scaler_params)
    scaler_params_data_frame.to_parquet(scaler_params_file, index=False, engine=engine_for_parquet)

    label_encoder = getattr(model, "label_encoder", None)
    

    classes = label_encoder.classes_
    labels = []
    for c in classes:
        labels.append(str(c))

    classes_file = "class_levels_from_python_svm.parquet"
    classes_df = pd.DataFrame({"label": labels})
    classes_df.to_parquet(classes_file, index=False, engine=engine_for_parquet)

    feature_order_str = []
    for x in feature_order:
        feature_order_str.append(str(x))

    length_feature_order = len(feature_order)
    feat_manifest = {
        "features": feature_order_str,
        "n_features": length_feature_order,
        "dtype_expected": "float32",
        "layout": "samples_rows_features_cols"
    }
    with open("features_svm.json", "w") as f:
        json.dump(feat_manifest, f, indent=2)

    cfg = {
        "algorithm": "LinearSVC",
        "feature_order": feature_order_str,
        "class_levels_file": classes_file,
        "scaler_params_file": scaler_params_file,
        "class_weight": "balanced"
    }

    with open("model_config_svm.json", "w") as f:
        json.dump(cfg, f, indent=2)



def to_num(i):
    return pd.to_numeric(i, errors="coerce")



def evaluation(model, data, feature_order, target_column, name):
    data_feature = data[feature_order]

    data_feature = data_feature.apply(to_num)

    data_feature = data_feature.astype("float32")

    label = data[target_column].astype(str)

    labelEncoder = model.label_encoder
    y_id = labelEncoder.transform(label)

    pred = model.predict(data_feature).astype(np.int64)

    acc = accuracy_score(y_id, pred)
    macro_f1 = f1_score(y_id, pred, average="macro", zero_division=0)
    rep = classification_report(y_id, pred, target_names=labelEncoder.classes_, digits=4, zero_division=0)
    confusion_ma  = confusion_matrix(y_id, pred)

    print(str(name) + " accuracy: " + str(round(acc, 6)))
    print(str(name) + " macro_f1:"+ str(round(macro_f1, 6)))
    print(str(name) + " per-class report:\n" + str(rep))

    return acc, macro_f1, confusion_ma, rep




def _ort_session(model_path):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])



def onnx_export(model, feature_order, onnx_path = "svm_timeseries_scaler_svm.onnx", opset = 12):
    lenght_feature_order = len(feature_order)
    initial_type = [("input", FloatTensorType([None, lenght_feature_order]))]

    svm_model = model.named_steps["svm"]
    options = {
        id(svm_model): {
            "output_class_labels": True,  
            "raw_scores": True            
        }
    }

    onnx_model = convert_sklearn(model, initial_types = initial_type, target_opset = opset, options = options)
    with open(onnx_path, "wb") as i:
        i.write(onnx_model.SerializeToString())

    return onnx_path

