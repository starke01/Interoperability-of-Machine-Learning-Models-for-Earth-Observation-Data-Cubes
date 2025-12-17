import json
from pathlib import Path

import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from joblib import dump
import time
from helper_svm import (identify_bands, svm_artifacts, evaluation, onnx_export, to_num)





def hyperparameters_linear_svc():
    return {
        "C": 2.5,
        "tol": 1e-4,
        "max_iter": 1000
    }


def parameter_for_engine():
    return {
        "engine_for_parquet": "pyarrow",
        "opset": 12,
        "target": "class_name",
        "column_fid": "fid", 
        "batch_size": 1024

    }



def ml_fit(training_set, target, pattern = None):
    preds_band = identify_bands(training_set, pattern)

    features = training_set[preds_band].apply(to_num)
    labels   = training_set[target]

    label_encoder = LabelEncoder()
    labels_r = label_encoder.fit_transform(labels)

    n_samples, n_features = features.shape

    lsvc = LinearSVC(
        C=hyperparameter_c,
        tol=tolerance,
        max_iter=lvsc_iter,
        class_weight = "balanced",
        dual= False,
        loss="squared_hinge"
    )




    model = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),("svm", lsvc)])
    print("Trainig ", n_samples, " samples ", n_features, " features")


    t0 = time.perf_counter()

    try:
        model.fit(features.values, labels_r)
    except Exception as e:
        print("error", e)
        raise e
    
    t1 = time.perf_counter()
    time_cal = t1 - t0
    print("time", time_cal)

    model.label_encoder = label_encoder

    try:
        print("Train-Score:", model.score(features, labels_r))
    except Exception:
        pass
    return model




if __name__ == "__main__":
    train_data     = "..training_data.."   
    validatio_data    = "..validation_data.."      


    base_dir = Path(train_data).parent
    artifacts_directory = base_dir / "model_artifacts_svm"
    artifacts_directory.mkdir(parents=True, exist_ok=True)

    onnx_path = artifacts_directory / "svm_timeseries_scaler_svm.onnx"
    pipiline_svm = artifacts_directory / "svm_python_pipeline.pkl"

    hyperparams = hyperparameters_linear_svc()
    hyperparameter_c = hyperparams["C"]
    tolerance = hyperparams["tol"]
    lvsc_iter = hyperparams["max_iter"]

    

    column_fid = parameter_for_engine()["column_fid"]
    target = parameter_for_engine()["target"]
    engine = parameter_for_engine()["engine_for_parquet"]
    band_pattern = r"^(B0?\d{2}|NDVI)_T\d+$"


    train_data_full = pd.read_parquet(train_data, engine=engine)
    feature_cols = identify_bands(train_data_full, band_pattern)

    train_data = train_data_full[[column_fid, target] + feature_cols].reset_index(drop=True)

    vailidation_data = pd.read_parquet(validatio_data, engine=engine)
    feature_cols_val = identify_bands(vailidation_data, band_pattern)
    validation_data = vailidation_data[[column_fid, target] + feature_cols_val].reset_index(drop=True)

  
    t0 = time.perf_counter()
    model = ml_fit(train_data, target, pattern=band_pattern)
    t1 = time.perf_counter()
    time_cal = t1 - t0
    print("Time: ", time_cal)


    feature_order = list(model.named_steps["scaler"].feature_names_in_)

    acc_training, f1_training, confusion_matrix_training, rep_training = evaluation(model, train_data, feature_order, target, "Train(areas 01+02, used)")

    acc_validation, f1_validation, confusion_matrix_validation, rep_validation = evaluation(model, validation_data, feature_order, target, "Val(area 03)")


    svm_artifacts(model, feature_order, engine_for_parquet=engine )    
    opset = parameter_for_engine()["opset"]
    onnx_export(model, feature_order, onnx_path=str(onnx_path), opset=opset)

    batch_size = parameter_for_engine()["batch_size"]

    dump(model, pipiline_svm)
    length_train_data = len(train_data)
    length_validation_data = len(validation_data)

    metrics = {
        "setup": {
            "description": "data",
            "data_splits": {
                "train": "areas 01 + 02",
                "validation": "area 03 (extern)",
                "test": "area 04",
            },
            "num_samples": {
                "train": length_train_data,
                "validation": length_validation_data,
            },
        },
        "performance": {
            "train": {
                "accuracy": acc_training,
                "macro_f1": f1_training,
            },
            "validation": {
                "accuracy": acc_validation,
                "macro_f1": f1_validation,
            },
        },
    }



    with open(artifacts_directory / "metrics_svm.json", "w") as i:
        json.dump(metrics, i, indent=2)
