
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnxruntime as ort
import logging


from helper import (
    identify_bands,
    time_bands,
    prepare_data,
    onnx_export, 
    saved_model_files_training,
    model_training_tempcnn, 
    f1_acc_oa

)



#temppcnn Hyperparameter
def parameters_tempcnn_model():
    return {
        "hidden_dims": 128,
        "kernel_size": 7,
        "dropout": 0.18203942949809093,
        "lr": 5e-4,
        "weight_decay": 1e-6,
        "lr_gamma": 0.95,
        "batch_size": 64,
        "epochs": 15,
        "patience": 5,
        "seed": 1337,
    }

#Compare ONNX vs native PyTorch TempCNN
def compare_native_onnx(training_result, onnx_path):
    model = training_result["model"]
    model = model.cpu()
    model = model.eval()
    T = training_result["T"]
    D = len(training_result["band_names"])
    batch = 1024
    torch_t = torch.randn(batch, T, D)
    with torch.no_grad():
        class_torch = model(torch_t).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    data_numpy = torch_t.numpy()
    data_numpy = data_numpy.astype(np.float32)

    outputs = sess.run(None, {"input": data_numpy})
    class_onnx = outputs[0]
    diff = np.abs(class_torch - class_onnx)
    print("ONNX vs Torch — max = " + str(diff.max()) + ", mean = " + str(diff.mean()))



#prediction function
def prediction(model, data_pre, label_pre, device, batch_size=1024):
    true_batch = []
    pred_batch = []
    loss_sum = 0.0
    n_batches = 0
    loss_fn = nn.NLLLoss()
    N = data_pre.shape[0]
    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            data_batch = data_pre[start:end].to(device)
            label_b = label_pre[start:end].to(device)
            logp = model(data_batch)
            batch_loss = loss_fn(logp, label_b)      
            batch_loss_value = batch_loss.item() 
            loss_sum += float(batch_loss_value)
            n_batches += 1
            pred_batch.append(torch.argmax(logp, dim=1).cpu().numpy())
            true_batch.append(label_b.cpu().numpy())

    y = np.concatenate(true_batch)
    p = np.concatenate(pred_batch)
     

    loss = loss_sum / max(1, n_batches)
    return y, p, loss



if __name__ == "__main__":


    train_data_file = ("..training_data..")
    validation_data_file = ("..validation_data..")


    try:
        engine = "pyarrow"
        training_data = pd.read_parquet(train_data_file, engine = engine)
        validation_data = pd.read_parquet(validation_data_file, engine = engine)
    except Exception as e:
        raise RuntimeError("error data loading " + str(e))

    device = "cpu"
    structure_per_band = r"^(B0?\d{2}|NDVI)_T\d+$"
    try:
        feature_cols = identify_bands(training_data, structure_per_band)
        band_names, T = time_bands(feature_cols)
        
    except Exception as e:
        raise RuntimeError(str(e))



    try:
        training_result = model_training_tempcnn(
            trainings_data = training_data,
            data_validation = validation_data,
            target_col = "class_name",
            band_names = band_names,
            T = T,
            device = device,
        )
    except Exception as e:
        raise RuntimeError(str(e))


    print(T)
    print(band_names)
    print("here")

    batch_size = parameters_tempcnn_model()["batch_size"]
    model = training_result["model"].to(device).eval()

    training_numpy_feature, class_numpy_training, _, _, _ = prepare_data(training_data,"class_name",band_names, T, label_encoder = training_result["label_encoder"],mean = training_result["mean"],std = training_result["std"])

    validation_numpy_feature, class_numpy_validation, _, _, _ = prepare_data(validation_data,"class_name",band_names, T, label_encoder = training_result["label_encoder"],mean = training_result["mean"],std=training_result["std"])

    train_features = torch.from_numpy(training_numpy_feature)
    train_classes = torch.from_numpy(class_numpy_training)
    validation_features_ = torch.from_numpy(validation_numpy_feature)
    validation_classes = torch.from_numpy(class_numpy_validation)


    for (features, classes, split_name) in [
        (train_features, train_classes, "Train"),
        (validation_features_,   validation_classes,   "Val"),
    ]:
        
        y_true, y_pred, loss = prediction(
            model, features, classes, device, batch_size=batch_size
        )
        oa, bal, mf1 = f1_acc_oa(y_true, y_pred)

    print(oa)
    print(bal)
    print(mf1)
    oa = float(oa)
    bal = float(bal)
    mf1 = float(mf1)
      
    output = ( "..output_onnx..")
    tempcnn_string_name = "tempcnn"
    path_a = "..base.."

    saved_model_files = saved_model_files_training(training_result, base_path = path_a, base_name = tempcnn_string_name)
    band_length = len(training_result["band_names"])
    try:
        onnx_path = onnx_export(saved_model_files["pt_path"],T = training_result["T"],D = band_length, ouput_onnx = output,opset = 12,
        )
    except Exception as e:
        raise RuntimeError(str(e))



    try:
        compare_native_onnx(training_result, onnx_path)
    except Exception as e:
        raise RuntimeError(str(e))

    
 
