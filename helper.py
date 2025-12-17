####
import numpy as np
import re
import os
import torch
import logging
import pandas as pd
import json
from pathlib import Path
import torch.nn as nn
import torch.onnx
from tempcnn_python import parameters_tempcnn_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
)

from tempcnn_str import TempCNN
import onnxruntime as ort
import time
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import joblib



def f1_acc_oa(data_t, data_p):
    if data_t.size == 0:
        return 0.0, 0.0, 0.0
    acc = accuracy_score(data_t, data_p)
    bal = balanced_accuracy_score(data_t, data_p)
    mf1 = f1_score(data_t, data_p, average = "macro", zero_division = 0)
    return acc, bal, mf1



def mean_and_std(trainings_data_wide, band_names, T, mean_list, std_list):
    for per_band in band_names:
        band_feature_columns = []
        for i in range(1, T + 1):
            col_name = per_band + "_T" + str(i)
            band_feature_columns.append(col_name)

        band_data = trainings_data_wide[band_feature_columns]
        band_data = band_data.to_numpy()
        band_data = band_data.reshape(-1)
        mean_np = np.nanmean(band_data)
        std_np = np.nanstd(band_data)
        if not np.isfinite(std_np) or std_np == 0:
            std_np = 1.0

        mean_list.append(mean_np)
        std_list.append(std_np)

    mean_array = np.array(mean_list)
    std_array = np.array(std_list)

    data_ = (mean_array, std_array)
   
    return data_


def identify_bands(data, pattern):
    band_feature = []
    band_feature_columns = data.columns

    for i in band_feature_columns:
        feat = str(i)
        if re.match(pattern, feat):
            band_feature.append(i)
    return band_feature



def time_bands(band_feature):
    bands_set = set()
    time_steps_set = set()

    for i in band_feature:
        parts = i.rsplit("_T", 1)
        if len(parts) != 2:
            continue
        bands_set.add(parts[0])
        num_ = int(parts[1])
        time_steps_set.add(num_)
    T = max(time_steps_set)
    bands = sorted(bands_set)
    return bands, T



def prepare_data(trainings_data_wide, target_col, band_names, T, label_encoder, mean , std ):

    target_values = trainings_data_wide[target_col]
    target_as_str = target_values.astype(str)
    classes_ = label_encoder.transform(target_as_str)

    classes_int = classes_.astype(np.int64)
    N = len(trainings_data_wide)
    D = len(band_names)

    tensor = np.empty((N, T, D), dtype=np.float32)

    for d, per_band in enumerate(band_names):
        band_time_f = []
        for i in range(1, T + 1):
            feature_col = per_band + "_T" + str(i)
            band_time_f.append(feature_col)

        band_matrix = trainings_data_wide[band_time_f].to_numpy() 
        band_matrix = band_matrix.astype(np.float32)
        tensor[:, :, d] = band_matrix

    mean_reshaped = mean.reshape(1, 1, -1)
    std_reshaped = std.reshape(1, 1, -1)

    tensor_centered = tensor - mean_reshaped
    tensor_st = tensor_centered / std_reshaped

    #sure float32 not 64
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)


    return tensor_st, classes_int, label_encoder, mean, std




def onnx_export(data_path, T, D, ouput_onnx = "tempcnn.onnx", opset = 12):

    model_ = torch.jit.load(data_path, map_location = "cpu").eval()
    dummy_tensor = torch.randn(1, T, D, dtype = torch.float32)

    torch.onnx.export(
        model_,
        dummy_tensor,
        ouput_onnx,
        input_names = ["input"],
        output_names = ["logprobs"],
        dynamic_axes = {"input": {0: "batch_size", 1: "time_steps"},"logprobs": {0: "batch_size"},},
        opset_version = opset,
        do_constant_folding=True,
    )
    print("ONNX: " + str(ouput_onnx))
    return ouput_onnx



def saved_model_files_training(training_result, base_path, base_name = "tempcnn"):

    base = Path(base_path)
    base.mkdir(parents = True, exist_ok = True)

    band = training_result["band_names"]
    mean = training_result["mean"]
    scale = training_result["std"]
    label = training_result["label_encoder"].classes_
    model_b = training_result["model"]

    bands = training_result["band_names"]
    band_length = len(bands)
    time_count = training_result["T"]


    scaler_frame = pd.DataFrame({"band": band,"mean": mean,"scale": scale,})
    labels_frame = pd.DataFrame({"label": label})


    engine = "pyarrow"
    scaler_frame.to_parquet(base / "scaler_params_tempcnn_bands.parquet",index = False,engine = engine,)
    labels_frame.to_parquet(base / "class_levels_from_python.parquet",index = False,engine = engine,)

    config_list = {
        "algorithm": "TempCNN",
        "time_steps": time_count,
        "band_order": bands,
        "class_levels_file": "class_levels_from_python.parquet",
        "scaler_params_file": "scaler_params_tempcnn_bands.parquet",
    }

    with open(base / "model_config.json", "w") as f:
        json.dump(config_list, f, indent = 2)

    model_b_cpu = model_b.cpu()
    model_eval = model_b_cpu.eval()



    model_t = torch.jit.trace(
        model_eval,
        torch.randn(1, time_count, band_length),
    )
    pt_path = base / (base_name + ".pt")
    pt_path_str = str(pt_path)
    model_t.save(pt_path_str)

    torch.save(model_b_cpu.state_dict(), base / "tempcnn_state.pth")
    return {"pt_path": pt_path_str}





def model_training_tempcnn(trainings_data, data_validation, target_col, band_names, T, device):

    params = parameters_tempcnn_model()
    hidden_dims = params["hidden_dims"]
    kernel_size = params["kernel_size"]
    dropout = params["dropout"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    lr_gamma = params["lr_gamma"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    patience = params["patience"]
    seed = params["seed"]
    best_validation_loss = float("inf")
    best_state = None
    no_impact = 0



    torch.manual_seed(seed)
    np.random.seed(seed)


    label_Encoder = LabelEncoder()
    target_values = trainings_data[target_col]
    target_str = target_values.astype(str)
    label_Encoder = label_Encoder.fit(target_str)


    mean_list = []
    std_list = []   


    mean_band, std_band = mean_and_std(trainings_data, band_names, T, mean_list, std_list)

    features_train_np, train_class_np, label_Encoder, mean_band, std_band = prepare_data(trainings_data, target_col, band_names, T, label_encoder = label_Encoder, mean = mean_band, std = std_band)
    
    features_validation_np, validation_class_np, _, _, _ = prepare_data(data_validation, target_col, band_names, T, label_encoder = label_Encoder, mean = mean_band, std = std_band)

    features_train = torch.from_numpy(features_train_np).to(device)
    train_class = torch.from_numpy(train_class_np).to(device)
    features_validation = torch.from_numpy(features_validation_np).to(device)
    validaiton_class = torch.from_numpy(validation_class_np).to(device)

    band_length = len(band_names)
    num_class_lenght = len(label_Encoder.classes_)
    model = TempCNN(
        input_dim = band_length,
        num_classes = num_class_lenght,
        sequencelength = T,
        kernel_size = kernel_size,
        hidden_dims = hidden_dims,
        dropout = dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay, eps = 1e-8)


    features_shape = features_train.shape[0]
    loss_fn = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        permutation_torch = torch.randperm(features_shape, device=device)
        total = 0.0
        n_batch = 0

        for start in range(0, features_shape, batch_size):
            bat = start + batch_size
            end = min(bat, features_shape)
            batch_indices = permutation_torch[start:end]

            train_batch = features_train[batch_indices]
            batch_class = train_class[batch_indices]
            optimizer.zero_grad()
            logport = model(train_batch)
            loss = loss_fn(logport, batch_class)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_value_float = loss_value
            total += loss_value_float

            n_batch += 1

        total_train_loss = total / max(1, n_batch)

        model.eval()

        with torch.no_grad():
            logits_val = model(features_validation)
            validation_loss_tensor = loss_fn(logits_val, validaiton_class)
            validation_loss_value = validation_loss_tensor.item()

            pred_classes_tensor = torch.argmax(logits_val, dim=1)
            pred_classes_cpu = pred_classes_tensor.cpu()
            pred_classes_np = pred_classes_cpu.numpy()
            validation_class_cpu = validaiton_class.cpu()      
            validation_class_numpy = validation_class_cpu.numpy()       

        oa, bal, mf1 = f1_acc_oa(validation_class_numpy, pred_classes_np)
        print(oa)
        print(bal)
        print(mf1)


        for i in optimizer.param_groups:
            i["lr"] *= lr_gamma

        improved = validation_loss_value < best_validation_loss - 1e-9

        if improved:
            best_validation_loss = validation_loss_value
            best_state = {}

            for name, param in model.state_dict().items():
                detached_param = param.detach()   
                cpu_param = detached_param.cpu() 
                cloned_param = cpu_param.clone() 
                best_state[name] = cloned_param
            no_impact = 0
        else:
            no_impact += 1

        print("Epoch " + str(epoch) + "/" + str(epochs) + " Train Loss = " + str(round(total_train_loss, 4))
            + " validation Loss = " + str(round(validation_loss_value, 4)) + " OA = " + str(round(oa, 4))
            + " BalAcc =" + str(round(bal, 4)) + " mF1 =" + str(round(mf1, 4))
            + " lr = " + str(round(optimizer.param_groups[0]["lr"], 6)) + "  No Improvement: " + str(no_impact))
        if no_impact >= patience:
            print( "early stopping after " + str(no_impact) + " best val loss: " + str(round(best_validation_loss, 4)))
            break



    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return {"model": model, "label_encoder": label_Encoder, "mean": mean_band, "std": std_band, "band_names": list(band_names), "T": int(T)}



















