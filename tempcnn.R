library(jsonlite)
library(caret)      
library(yardstick)    
library(torch)
library(reticulate)   
library(arrow)  


python_bin <- "..venv.."
reticulate::use_python(python_bin, required = TRUE)


train_parquet <- "..trainings_data.."
validation_parquet   <- "..validation_data.."
target_column <- "class_name"
id_column     <- "fid"
band_pattern  <- "^(B0?\\d{2}|NDVI)_T\\d+$"
artifacts_dir <- "..Save_artifacts_directory.."
model_name    <- "tempcnn_r"
onnx_opset    <- 12L



hidden_dims   <- 128L
kernel_size   <- 7L
dropout       <- 0.18203942949809093
learning_rate <- 5e-4
weight_decay  <- 1e-6
lr_gamma      <- 0.95
batch_size    <- 64L
epochs        <- 15L
patience      <- 5L
seed          <- 1337L




read_parquet_data_frame <- function(path) {
  as.data.frame(arrow::read_parquet(path))
}




write_parquet <- function(data, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  arrow::write_parquet(data, path)
  return(path)
}



identify_predictors <- function(colnames, pattern = band_pattern) {
  is_feature <- base::grepl(pattern, colnames)
  features <- colnames[is_feature]
  return(features)
}



detect_bands <- function(feature) {
  
  base_names <- base::sub("_T\\d+$", "", feature)
  unique_bands <- base::unique(base_names)
  bands <- base::sort(unique_bands)
  
  time_str <- base::sub(".*_T", "", feature)
  time_i <- base::as.integer(time_str)
  T <- base::max(time_i, na.rm = TRUE)
  return(list(bands = bands, T = T))
}




clean_data_ <- function(data_frame, features, target_c, id_c) {
  
  feature_raw <- data_frame[, features, drop = FALSE]
  feature_numeric <- as.data.frame(lapply(feature_raw, as.numeric))
  
  out <- cbind(
    data_frame[, c(id_c, target_c), drop = FALSE],feature_numeric)
  
  rownames(out) <- NULL
  return(out)
}





std_cal <- function(dat) {
  mean_ <- mean(dat, na.rm = TRUE)
  differnece <- dat - mean_
  diff_sq <- differnece^2
  mean_sq_dev <- mean(diff_sq, na.rm = TRUE)
  sd_population <- sqrt(mean_sq_dev)
  return(sd_population)
}



compute_band_scaler_from_wide <- function(data_wide, band_names, T) {
  
  band_names_length = length(band_names)
  mean_ <- numeric(band_names_length)
  std_ <- numeric(band_names_length)
  
  for (i in seq_along(band_names)) {
    
    feature_col <- paste0(band_names[i], "_T", seq_len(T))
    band_values <- unlist(data_wide[, feature_col, drop = FALSE], use.names = FALSE)
    
    mean_band <- mean(band_values, na.rm = TRUE)
    band_sd <- std_cal(band_values)
    
    if (!is.finite(band_sd)) {
      band_sd <- 1 }
    
    if (band_sd == 0) {
      band_sd <- 1}
    
    mean_[i] <- mean_band
    std_[i] <- band_sd
  }
  
  return(list(mu = mean_, sg = std_))
}



encode_labels <- function(y, levels_vec) {
  factor(y, levels = levels_vec)
}

labels_to_int <- function(y_factor) {
  as.integer(y_factor)
}


label_factor <- function(data_) {
  data_chr  <- as.character(data_)
  data_uniq <- unique(data_chr)
  data_sort <- sort(data_uniq)
  return(list(levels = data_sort))
}


create_tempcnn_model <- function(input_dimension, sequence_lenght, class_count) {
  
  hidden  <- hidden_dims
  p <- dropout
  dense_h <- 4L * hidden
  
  conv_block <- function(in_ch, out_ch) {
    padding = kernel_size %/% 2
    nn_sequential(
      nn_conv1d(in_channels = in_ch, out_channels = out_ch,
                kernel_size = kernel_size, padding = padding),
                nn_batch_norm1d(num_features = out_ch),
                nn_relu(),
                nn_dropout(p = p)
    )
  }
  
  nn_module(
    "TempCNN",
    initialize = function() {
      self$conv1 <- conv_block(input_dimension, hidden)
      self$conv2 <- conv_block(hidden, hidden)
      self$conv3 <- conv_block(hidden, hidden)
      self$flatten<- nn_flatten(start_dim = 2)
      self$dense <- nn_sequential( nn_linear(in_features = hidden * sequence_lenght, out_features = dense_h),
                    nn_batch_norm1d(num_features = dense_h),
                    nn_relu(),
                    nn_dropout(p = p))
      
      self$head_linear <- nn_linear(in_features = dense_h, out_features = class_count)
    },
    
    forward = function(data_ten) {
      data_ten <- data_ten$transpose(2, 3)
      data_ten <- self$conv1(data_ten)
      data_ten <- self$conv2(data_ten)
      data_ten <- self$conv3(data_ten)
      data_ten <- self$flatten(data_ten)
      data_ten <- self$dense(data_ten)
      data_ten <- self$head_linear(data_ten)
      nnf_log_softmax(data_ten, dim = -1)
    }
  )()
}




wide_to_tempcnn_tensors <- function(data_wide, target_col, band_names, T, label_levels, mean_, std_) {
  
  N <- nrow(data_wide)
  D <- length(band_names)
  
  data_arr <- array(0, dim = c(N, T, D))
  
  for (i in seq_along(band_names)) {
    band <- band_names[i]
    col_ <- paste0(band, "_T", seq_len(T))
    
    data_arr[, , i] <- as.matrix(data_wide[, col_, drop = FALSE])
  }
  
  for (d in seq_along(band_names)) {
    centered <- data_arr[, , d] - mean_[d]
    scal  <- centered / std_[d]
    data_arr[, , d] <- scal
}
  
  
  target_character <- as.character(data_wide[[target_col]])
  factor_tar <- encode_labels(target_character, label_levels)
  tar_int <- labels_to_int(factor_tar)
  
  
  return(list(X = data_arr, y = tar_int))
}



metrics_from_logprobs <- function(log_probs, y_int, label_levels) {
  
  preds_raw <- torch_argmax(log_probs, dim = 2)
  preds     <- preds_raw$to(device = "cpu")
  preds_i <- as.integer(as_array(preds))
  cpu_i   <- y_int$to(device = "cpu")
  y_all   <- as.integer(as_array(cpu_i))
  labels_s    <- seq_along(label_levels)
  y_factor    <- factor(y_all,  levels = labels_s, labels = label_levels)
  pred_factor <- factor(preds_i, levels = labels_s, labels = label_levels)
  
  overall_acc <- mean(y_all == preds_i)
  # We use yardstic at this point because we do not include it in the evaluation for our work. The evaluation only takes place in the inference
  bal_acc  <- yardstick::bal_accuracy_vec(y_factor, pred_factor)
  macro_f1 <- yardstick::f_meas_vec(y_factor, pred_factor, estimator = "macro")
  
  return(list(oa = overall_acc, bal_acc = bal_acc, macro_f1 = macro_f1))
}



train_tempcnn_from_wide <- function( train_data, validation_data, target_col , band_names, T, device = torch_device("cpu")){
  
  torch_manual_seed(seed)
  target_label <- label_factor(train_data[[target_col]])
  label_levels <- target_label$levels
  
  scal  <- compute_band_scaler_from_wide(train_data, band_names, T)
  mean_  <- scal$mu
  std_  <- scal$sg

  feature_ten <- wide_to_tempcnn_tensors(train_data, target_col, band_names, T, label_levels, mean_, std_)
  validation_ten <- wide_to_tempcnn_tensors(validation_data, target_col, band_names, T, label_levels, mean_, std_)
  
  feature_tensor_train <- torch_tensor(feature_ten$X, dtype = torch_float(), device = device)
  label_tensor <- torch_tensor(feature_ten$y, dtype = torch_long(),  device = device)
  feature_tensor_val <- torch_tensor(validation_ten$X, dtype = torch_float(), device = device)
  label_tensor_val <- torch_tensor(validation_ten$y, dtype = torch_long(),  device = device)
  
  input_dimension_length <- length(band_names)
  class_lenght <- length(label_levels)
  model <- create_tempcnn_model(input_dimension = input_dimension_length, sequence_lenght = T, class_count = class_lenght)
  
  optimizer <- optim_adamw(model$parameters, lr = learning_rate, eps = 1e-8, weight_decay = weight_decay)
  loss_fn   <- nn_nll_loss()
  
  best_val_loss <- Inf
  best_state <- NULL
  no_impect <- 0L
  

  train_x <- feature_ten$X
  dim_train <- dim(train_x)
  n_train_samples <- dim_train[1]
  
  
  for (epoch in seq_len(epochs)) {
    model$train()
    
    train_loss_sum <- 0
    train_batches  <- 0
    
    perm_ran <- sample.int(n_train_samples)
    
    for (i in seq(1L, n_train_samples, by = batch_size)) {
      he <- i + batch_size - 1L
      end <- min(he, n_train_samples)
      batch_indices <- perm_ran[i:end]
      
      data_batch <- feature_tensor_train[batch_indices, , ]
      label_batch <- label_tensor[batch_indices]
  
      
      optimizer$zero_grad()
      log_probs <- model(data_batch)
      loss <- loss_fn(log_probs, label_batch)
      
      loss$backward()
      optimizer$step()
      
      loss_value <- loss$item()         
      loss_value <- as.numeric(loss_value)  
      train_loss_sum <- train_loss_sum + loss_value
      train_batches <- train_batches + 1L
    }
    
    max_l <- max(1, train_batches)
    train_loss <- train_loss_sum / max_l
    
    
    model$eval()
    torch::with_no_grad({
      log_probs_val <- model(feature_tensor_val)
      validation_loss <- loss_fn(log_probs_val, label_tensor_val)
    })
    
    
    val_loss_v <- validation_loss$item()
    val_loss_num <- as.numeric(val_loss_v)
    
    metrcis_val <- metrics_from_logprobs(log_probs_val, label_tensor_val, label_levels)
    
    message(paste0("Epoch ", epoch, "/", epochs," train_loss = ", round(train_loss, 4), "validation_loss = ", round(val_loss_num, 4)," acc(val)=", round(metrcis_val$oa, 4),
                  "BalAcc=", round(metrcis_val$bal_acc, 4), " mF1 = ", round(metrcis_val$macro_f1, 4)))
    
    
    lr_before_decay <- optimizer$param_groups[[1]]$lr
    lr_after_decay <- lr_before_decay * lr_gamma
    optimizer$param_groups[[1]]$lr <- lr_after_decay
    
    
    if (val_loss_num < best_val_loss - 1e-9) {
      best_val_loss <- val_loss_num
      best_state  <- model$state_dict()
      no_impect <- 0L
    } else {
      no_impect <- no_impect + 1L
    }
    
    if (no_impect >= patience) {
      message(paste0("Early stopping ", epoch," epoch (best val_los s =", round(best_val_loss, 4), ")"))
      break
    }
    
  }
  
  if (!is.null(best_state)) {
    model$load_state_dict(best_state)
  }
  model$eval()
  
  return( list(
    model = model,
    label_levels = label_levels,
    mu_band  = mean_,
    sg_band = std_,
    band_names  = band_names,
    T  = T
  ))
 
}

save_tempcnn_artifacts <- function(bundle, artifacts_dir, model_name) {
  
  scaler <- file.path(artifacts_dir, "scaler_params_tempcnn_bands_r.parquet")
  classes <- file.path(artifacts_dir, "class_levels_from_r.parquet")
  model_config <- file.path(artifacts_dir, "model_config.json")
  native_model <- file.path(artifacts_dir, paste0(model_name, ".pt"))
  
  write_parquet( data.frame(band = bundle$band_names, mean = bundle$mu_band, scale = bundle$sg_band),scaler)
  
  write_parquet(data.frame(label = bundle$label_levels),classes)
  
  config_ <- list(
    time_steps = bundle$T,
    band_order = bundle$band_names,
    class_levels_file = basename(classes),
    scaler_params_file = basename(scaler)
  )
  jsonlite::write_json(config_, model_config, auto_unbox = TRUE, pretty = TRUE)
  
  batch_size <- 1L
  time_steps <- bundle$T
  n_features <- length(bundle$band_names)
  
  dummy_shape <- c(batch_size, time_steps, n_features)
  dummy <- torch_randn(dummy_shape, dtype = torch_float())
  
  device_cpu <- torch_device("cpu")
  model_cpu <- bundle$model$to(device = device_cpu)
  model_cpu$eval()
  
  traced <- torch::jit_trace(model_cpu, dummy)
  torch::jit_save(traced, native_model)
  
  return(list(
    scaler = scaler,
    classes = classes,
    model_config = model_config,
    native_model = native_model
  ))
}



export_pt_to_onnx <- function(native_model, T, D, onnx_path, opset = 12L) {
  
  torch_py <- reticulate::import("torch", delay_load = FALSE)
  
  natic_path <- normalizePath(native_model, winslash = "/")
  jit_ <- torch_py$jit
  mod <- jit_$load(natic_path, map_location = "cpu")
  mod$eval()
  
  batch_size <- as.integer(1)
  time_steps <- T
  n_features <- D
  dummy_type <- torch_py$float32
  dummy <- torch_py$randn(batch_size, time_steps, n_features, dtype = dummy_type)
  
  dict <- reticulate::dict
  data_input <- dict()
  data_output <- dict()
  data_input[[as.integer(0)]] <- "batch_size"
  data_input[[as.integer(1)]] <- "time_steps"
  data_output[[0]] <- "batch_size"
  dynamic_axes <- dict()
  dynamic_axes[["input"]] <- data_input
  dynamic_axes[["logprobs"]] <- data_output
  
  torch_py$onnx$export(mod, dummy, normalizePath(onnx_path, winslash = "/"), input_names = list("input"), output_names = list("logprobs"), dynamic_axes = dynamic_axes, opset_version = opset, do_constant_folding = TRUE)
  
  return(onnx_path)
}



set.seed(seed);
torch_manual_seed(seed)


train_data <- read_parquet_data_frame(train_parquet)
features_train <- identify_predictors(names(train_data), band_pattern)
bt <- detect_bands(features_train)
band_names <- bt$bands
T_ <- bt$T


train_data_c <- clean_data_(train_data, features_train, target_column, id_column)

validation_data <- read_parquet_data_frame(validation_parquet)
validation_data  <- clean_data_(validation_data, features_train, target_column, id_column)


device <- torch_device("cpu")

bundle_t <- train_tempcnn_from_wide(train_data = train_data_c, validation_data = validation_data, target_col = target_column, band_names = band_names, T = T_, device = device)



artifacts <- save_tempcnn_artifacts(bundle_t, artifacts_dir = artifacts_dir, model_name = model_name)

model_config <- artifacts$model_config
native_model <- artifacts$native_model
scaler <- artifacts$scaler
classes  <- artifacts$classes


model_onnx <- file.path(artifacts_dir, "tempcnn.onnx")
config_json   <- jsonlite::read_json(model_config)
bands_lenght <- length(config_json$band_order)

export_pt_to_onnx(native_model = native_model,T = config_json$time_steps, D  = bands_lenght,  onnx_path = model_onnx, opset = onnx_opset)

