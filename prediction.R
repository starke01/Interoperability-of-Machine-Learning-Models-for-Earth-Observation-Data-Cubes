
library(jsonlite)
library(arrow)
library(terra)
library(sf)
library(reticulate)
library(torch)
library(caret)  
library(ggplot2)
library(tibble)
library(dplyr)

#use_virtualenv("..venv..)", required = FALSE)


compare_torch_vs_onnx_r <- function(model_pt, model_onnx, time_steps, dims, batch = 1024L, julia_layout = FALSE) {
  message("Comparison of Torch (TorchScript) with ONNX")
  
  model <- torch::jit_load(model_pt)
  model$eval()
  
  ort <- reticulate::import("onnxruntime")
  np  <- reticulate::import("numpy", convert = FALSE)
  sess <- ort$InferenceSession(model_onnx, providers = list("CPUExecutionProvider"))
  input_name <- sess$get_inputs()[[1]]$name
  

  number_elements <- batch * time_steps * dims
  vals <- stats::rnorm(number_elements)
  data_array <- array(vals, dim = c(batch, time_steps, dims))
  
  
  tensor_data <- torch::torch_tensor(data_array, dtype = torch::torch_float())
  
  torch::with_no_grad({
    pred <- model(tensor_data)
  })
  
  
  predication_array <- as.array(pred)
  
  if (julia_layout) {
    onnx_data <- aperm(data_array, c(1, 3, 2))
  } else {
    onnx_data <- data_array
  }
  
  py_data <- reticulate::r_to_py(onnx_data)
  np_arr <- np$array(py_data, dtype = np$float32)
  np_ <- np$ascontiguousarray(np_arr)
  
  
  
  onnx_pred <- sess$run(NULL, setNames(list(np_), input_name))[[1]]
  pred_r <- reticulate::py_to_r(onnx_pred)  
  dim_t <- dim(predication_array)
  dim_o <- dim(pred_r)
  
  if (!all(dim_t == dim_o)) {
     warning("error dim" , paste(dim_t, collapse = "x"), " vs ", paste(dim_o,  collapse = "x")) }
  
  difference_raw <- abs(predication_array - pred_r)
  maximal_difference_raw <- max(difference_raw)
  mean_difference_raw <- mean(difference_raw)
  
  message("max =", maximal_difference_raw, ", mean = ", mean_difference_raw)
  
  
  pred_torch <- max.col(predication_array) - 1L  
  pred_onnx <- max.col(pred_r)  - 1L
  
  number_total <- length(pred_torch)
  number_equal <- sum(pred_torch == pred_onnx)
  
  equal_n <- if (number_total > 0L) round(100 * number_equal / number_total, 2) else 0
  message( "argmax ", number_equal, " / ", number_total, " identical predication (", equal_n ," %)" )
  
  invisible(list(
    max_diff_logprobs = maximal_difference_raw,
    mean_diff_logprobs = mean_difference_raw,
    n_equal = number_equal,
    n_total = number_total,
    y_torch_logprobs = predication_array,
    y_onnx_logprobs  = pred_r
  ))
}







aoi_load <- function(path, epsg) {
  aoi <- sf::st_read(path, quiet = TRUE)
  aoi <- sf::st_transform(aoi, epsg)
  
  if (!"fid" %in% names(aoi)) {
    aoi$fid <- seq_len(nrow(aoi))
  }
  return(aoi)
}



warmup_for_onnx <- function(sess, input_name, data, np, output_names = NULL, n_runs = 3L) {

  np_arr_data <- np$array(
    reticulate::r_to_py(data),
    dtype = np$float32
  )
  
  input_dict <- setNames(list(np_arr_data), input_name)
  for (i in seq_len(n_runs)) {
    invisible(sess$run(output_names, input_dict))
  }
}



warump_for_torch <- function(model, data, n_runs = 3L) {
  tens <- torch::torch_tensor(data, dtype = torch::torch_float())
  
  for (i in seq_len(n_runs)) {
    torch::with_no_grad({
      invisible(model(tens))
    })
  }
}


build_raster <- function(aoi, resolution, epsg) {
  bb_box <- sf::st_bbox(aoi)
  
  xmin <- as.numeric(bb_box["xmin"])
  xmax <- as.numeric(bb_box["xmax"])
  ymin <- as.numeric(bb_box["ymin"])
  ymax <- as.numeric(bb_box["ymax"])
  
  width <- ceiling((xmax - xmin) / resolution)
  height <- ceiling((ymax - ymin) / resolution)

  xmax_rast_object <- xmin + width  * resolution   
  ymin_rast_object <- ymax - height * resolution  
  terra::rast(ncols = width, nrows = height, xmin = xmin, xmax = xmax_rast_object, ymin = ymin_rast_object, ymax = ymax, crs = epsg)
}


ground_truth_raster_f <- function(ground_truth, class, class_id, template_, epsg, noData) {
  
  ground_truth_data <- sf::st_read(ground_truth)
  ground_truth_data <- sf::st_transform(ground_truth_data, epsg)
  
  class_c <- as.character(ground_truth_data[[class]])
  
  mask_data <- !is.na(class_c) & class_c %in% names(class_id)
  ground_truth_data <- ground_truth_data[mask_data, ]
  
  class_labels <- class_c[mask_data]
  id <- unname(class_id[class_labels])
  
  ground_truth_data$class_label <- class_labels
  ground_truth_data$class_id <- id
  
  raster_ground_truth <- template_
  raster_ground_truth[] <- noData
  
  
  raster_ground_truth <- terra::rasterize(terra::vect(ground_truth_data), raster_ground_truth, field = "class_id", touches = FALSE, background = noData)
    
  
  return(raster_ground_truth)
}





evaluate_prediction_streaming <- function(prediction, raster_ground_truth, classes, noData, chunk_size = 256L) {
  class_lenght <- length(classes)
  
  conf_mat <- matrix(0L, nrow = class_lenght, ncol = class_lenght)
  
  number_rows <- terra::nrow(prediction)
  
  terra::readStart(prediction)
  terra::readStart(raster_ground_truth)

  on.exit({
    terra::readStop(prediction)
    terra::readStop(raster_ground_truth)
  })
  
  row_begin <- seq.int(from = 1L, to = number_rows, by = chunk_size)
  
  for (roww in row_begin) {
    n_rows_chunk <- number_rows - roww + 1L
    n_rows_chunk <- min(chunk_size, n_rows_chunk)
    
    prediction_val_raw <- terra::readValues(prediction, row = roww, nrows = n_rows_chunk, mat = FALSE)
    ground_truth_val_raw <- terra::readValues(raster_ground_truth, row = roww, nrows = n_rows_chunk, mat = FALSE)
    
    if (length(prediction_val_raw) == 0L || length(ground_truth_val_raw) == 0L) {
      next
    }
    
    prediction_values <- as.integer(prediction_val_raw)
    ground_truth_values <- as.integer(ground_truth_val_raw)
    
    valid_mask <- !is.na(prediction_values) & !is.na(ground_truth_values) & (prediction_values != noData) & (ground_truth_values!= noData)
    
    if (!any(valid_mask)) next
    

    pred_valid <- prediction_values[valid_mask]
    gt_valid <- ground_truth_values[valid_mask]
    
    pred_classes <- 0:(class_lenght - 1L)
    
    for (true_class in 0:(class_lenght - 1L)) {
      id_true_class <- which(gt_valid == true_class)

      if (length(id_true_class) == 0L) next
      
      pred_for_true_class <- pred_valid[id_true_class]
      row_idx <- true_class + 1L
      
      for (pred_class in pred_classes) {
        count_pr <- sum(pred_for_true_class == pred_class)
        prediction_class_col <- pred_class + 1L
        conf_mat[row_idx, prediction_class_col] <- conf_mat[row_idx, prediction_class_col] + count_pr
      }
    }
    
    
  }
  
  
  diag_values <- diag(conf_mat)
  count_total <- sum(diag_values)
  count_pixel <- sum(conf_mat)
  
  acc <- count_total / count_pixel
  
  true_positiv <- diag(conf_mat)
  row_totals <- rowSums(conf_mat)
  col_totals <- colSums(conf_mat)
  
  
  metrics_table_ <- data.frame(
    class = classes,
    tp = true_positiv,
    row_totals = row_totals,
    col_totals = col_totals,
    stringsAsFactors = FALSE
  )
  #ifself works element by element across all classes simultaneously
  metrics_table_$recall <- ifelse(metrics_table_$row_totals == 0L, 0,metrics_table_$tp / metrics_table_$row_totals)

  metrics_table_$precision <- ifelse(metrics_table_$col_totals == 0L, NA_real_, metrics_table_$tp / metrics_table_$col_totals)

  metrics_table_$f1 <- ifelse(is.na(metrics_table_$precision) | (metrics_table_$precision + metrics_table_$recall) == 0, 0,
    2 * metrics_table_$precision * metrics_table_$recall /
      (metrics_table_$precision + metrics_table_$recall))


  recall <- metrics_table_$recall
  precision <- metrics_table_$precision
  f1 <- metrics_table_$f1
  balacc   <- mean(recall)
  macro_f1 <- mean(f1)
  
  message("acc=", round(acc, 4), " balacc=", round(balacc, 4))
  
  
  #pixel
  for (k in seq_len(class_lenght)) {
    is_missing <- is.na(precision[k])
    prec_p <- ifelse(is_missing, NaN, precision[k])
    recall_print  <- recall[k]
    f1_print <- f1[k]
    
    message("  ", classes[k], ": precision = ", round(prec_p, 4), " recall =", round(recall_print, 4), " f1 = ", round(f1_print, 4))
  }
  
  message("macro f1 = ", round(macro_f1, 4))
  
  invisible(list(
    conf_mat = conf_mat,
    acc = acc,
    balacc = balacc,
    precision = precision,
    recall = recall,
    f1 = f1,
    metrics_tbl = metrics_table_
  ))
}


###

area04_inference <- "..inference_data.."
a <- read_parquet(area04_inference)
ground_truth_data_g  <- "..ground_truth_data_geojson.."

model_config <- "..model_config.."

native_model <- "..native_r.."
typeof(native_model)
model_onnx <- "..onnx_model.."
scaler <- "..scaler_data.."
classes <- "..classes.."
features_list <- "..features_manifest.."

svm_onnx_path <- "..svm_onnx.."
svm_artifacts_dir <- "..svm_dir.."
classes_svm_py <- "..svm_class.."

external_prediction <- "..external_prediction.."
output_tif  <- "..output_tif.."
overlay_tif_pred  <- "..overlay_tif.."

#
pt <- FALSE
use_onnx_svm      <- TRUE
onnx_yes          <- FALSE      
onnx_julia_yes    <- FALSE     
noData            <- -1L
batch_number      <- 1024L
gt_class          <- "class_name"
epsg              <- "EPSG:3857"
resolution_area   <- 20
parcel_eval <- list()

###



aoi <- aoi_load(ground_truth_data_g, epsg)
template_for_r <- build_raster(aoi, resolution_area, epsg)


if (use_onnx_svm) {
  
  class_svm_list  <- arrow::read_parquet(classes_svm_py)
  svm_classes <- as.character(class_svm_list$label)
  svm_class_id <- seq_along(svm_classes) - 1L
  class_to_id <- setNames(svm_class_id, svm_classes)
  
  features_json <- jsonlite::fromJSON(features_list)
  feature_order <- features_json$features
  
  area_inference_data <- arrow::read_parquet(area04_inference)
  
  inference_features <- area_inference_data[, feature_order, drop = FALSE]
  
  for (i in feature_order) {
    inference_features[[i]] <- as.numeric(inference_features[[i]])
  }
  
  
  inference_matrix <- as.matrix(inference_features)
  
  
  
  ort  <- reticulate::import("onnxruntime")
  np   <- reticulate::import("numpy", convert = FALSE)
  sess <- ort$InferenceSession(svm_onnx_path, providers = list("CPUExecutionProvider"))
  input_name <- sess$get_inputs()[[1]]$name
  
  n_infer <- nrow(inference_matrix)
  warm_batch <- min(n_infer, 5L)
  
  warm_data <- inference_matrix[1:warm_batch, , drop = FALSE]
  
  warmup_for_onnx(sess = sess, input_name = input_name, data = warm_data, np = np, output_names = NULL, n_runs = 3L)
  
  
  time_t0 <- Sys.time()
  
  matrix_nrow <- nrow(inference_matrix)
  all_preds <- integer(matrix_nrow)
  id_c <- 1L
  
  while (id_c <= matrix_nrow) {
    number <- min(id_c + batch_number - 1L, matrix_nrow)
    row_number_max <- id_c:number
    matrix_c <- inference_matrix[row_number_max, , drop = FALSE]
    np_array <- np$array(reticulate::r_to_py(matrix_c), dtype = np$float32)
    outs   <- sess$run(NULL, setNames(list(np_array), input_name))
    labels <- as.integer(reticulate::py_to_r(outs[[1]]))
    all_preds[id_c:number] <- labels
    id_c <- number + 1L
  }
  
  time_t1 <- Sys.time()
  time_t <- time_t1 - time_t0
  time_secs <- as.numeric(time_t, units = "secs")
  time_secs_rounded <- round(time_secs, 3)
  
  
  message("inference loop: ", time_secs_rounded, " sec")
  
  
  
  
  class_levels <- svm_classes
  class_names <- as.character(area_inference_data$class_name)
  
  true_class_id <- unname(class_to_id[class_names])
  pred_class_id <- all_preds
  
  y_true_factor <- factor(class_levels[true_class_id + 1L], levels = class_levels)
  y_pred_factor <- factor(class_levels[pred_class_id + 1L], levels = class_levels)
  
  parcel_eval$svm <- list(y_true = y_true_factor, y_pred = y_pred_factor,model = "SVM-ONNX")
  
  
  
  y_pred_parcels <- all_preds
  
  prediction_data_frame <- data.frame(fid = area_inference_data$fid, pred = y_pred_parcels)
  
  
  mactch_fid <- match(aoi$fid, prediction_data_frame$fid)
  pred_vals <- prediction_data_frame$pred[mactch_fid]
  aoi$pred <- pred_vals
  
  aoi$pred[is.na(aoi$pred)] <- noData
  
  r_prediction_raster <- terra::rasterize(terra::vect(aoi), template_for_r, field = "pred", background = noData,touches  = FALSE)
  terra::NAflag(r_prediction_raster) <- noData
  
  terra::writeRaster(r_prediction_raster, output_tif, overwrite = TRUE, filetype  = "GTiff", datatype  = "INT2S", gdal = c("COMPRESS=LZW","TILED=YES","BLOCKXSIZE=256","BLOCKYSIZE=256"), NAflag = noData)
  
  r_prediction_raster <- terra::rast(output_tif)
  terra::NAflag(r_prediction_raster) <- noData
  
  r_ground_truth_raster <- ground_truth_raster_f(ground_truth_data_g, gt_class, class_to_id, template_for_r, epsg, noData)
  gt_tempfile <- tempfile(fileext = ".tif")
  
  terra::writeRaster(r_ground_truth_raster, gt_tempfile, overwrite = TRUE)
  
  r_gt_raster_final <- terra::rast(gt_tempfile)
  terra::NAflag(r_gt_raster_final) <- noData
  
  r_gt <- r_gt_raster_final
  
  evaluate_prediction_streaming(r_prediction_raster, r_gt, svm_classes, noData)
  
  r_prediction_rasteriction <- prediction_data_frame
  arrow::write_parquet(r_prediction_rasteriction, "..prediction..")
}



if (onnx_yes || pt) {
  message(typeof(native_model))
  features_json <- jsonlite::fromJSON(model_config)
  band_order  <- features_json$band_order
  T  <- as.integer(features_json$time_steps) 

  
  class_tempcnn <- arrow::read_parquet(classes)
  typeof(class_tempcnn)
  classes <- as.character(class_tempcnn$label)
  typeof(classes)
  class_lenght <- length(classes)
  class_to_id <- setNames(seq.int(0, class_lenght - 1L), classes)
  
  
  message("scaler")
  scaler_path <- scaler
  scaler <- arrow::read_parquet(scaler_path)
  scaler <- scaler[match(band_order, scaler$band), ]
  mean_value <- as.numeric(scaler$mean)
  standard_deviation <- as.numeric(scaler$scale)
  
  area_inference_data <- arrow::read_parquet(area04_inference)
  
  parcel_row <- nrow(area_inference_data)
  dim_bands <- length(band_order)
  D <- dim_bands
  
  
  data_array <- array(NA_real_, dim = c(parcel_row, T, D))
  for (d in seq_len(D)) {
    b <- band_order[d]
    for (t in seq_len(T)) {
      colname <- sprintf("%s_T%d", b, t)
      data_array[, t, d] <- as.numeric(area_inference_data[[colname]])
    }
  }
  for (d in seq_len(D)) {
    feature_slice <- data_array[ , , d]
    
    mean_value_d <- mean_value[d]
    standard_deviation_d <- standard_deviation[d]
    feature_scaled <- (feature_slice - mean_value_d) / standard_deviation_d
    data_array[ , , d] <- feature_scaled
  }
  message("here")
  
  

  compare_torch_vs_onnx_r(model_pt = native_model, model_onnx = model_onnx, time_steps = T, dims = D, batch = 1024, julia_layout = onnx_julia_yes)
  
  message("pred...")
  
  if (onnx_yes) {
    
    ort <- reticulate::import("onnxruntime")
    np  <- reticulate::import("numpy", convert = FALSE)
    sess <- ort$InferenceSession(model_onnx, providers = list("CPUExecutionProvider"))
    input_name <- sess$get_inputs()[[1]]$name
    output_name <- sess$get_outputs()[[1]]$name
    
    if (onnx_julia_yes) {
      data_array_NTD <- data_array
      data_array_NDT <- aperm(data_array_NTD, c(1, 3, 2))
      
      data_array_py <- reticulate::r_to_py(data_array_NDT)
      array_np_data <- np$array(data_array_py, dtype = np$float32)
      
    }
  } else {
    
    message(typeof(native_model))
    
    model <- torch::jit_load(native_model)
    model$eval()
  }
  
  
  if (onnx_julia_yes) {
    warm_batch  <- min(parcel_row, batch_number)
    i <- 1:warm_batch
    warm_array_ntd <- data_array[i, , , drop = FALSE]     
    warm_array_ndt <- aperm(warm_array_ntd, c(1, 3, 2)) 
    
    warmup_for_onnx(sess = sess, input_name = input_name, data = warm_array_ndt, np = np, output_names = list(output_name), n_runs = 3L)
    
    time_t0 <- Sys.time()
    
    all_preds <- integer(parcel_row)
    for (i in seq_len(parcel_row)) {
      py_zero <- i - 1L
      py_item_i  <- reticulate::py_get_item(array_np_data, py_zero)  
      py_item_i  <- np$expand_dims(py_item_i, 0L)                       
      output_sess <- sess$run(list(output_name),setNames(list(py_item_i), input_name))[[1]]
      out_r <- reticulate::py_to_r(output_sess)     
      scores_r_i <- out_r[1, ]                
      pred_max_r <- which.max(scores_r_i)     
      pred_max_zero <- pred_max_r - 1L
      all_preds[i] <- pred_max_zero
    }
    
    time_t1<- Sys.time()
    
    time_t <- time_t1 - time_t0
    time_secs <- as.numeric(time_t, units = "secs")
    time_secs_rounded <- round(time_secs, 3)
    
    message("inference loop: ", time_secs_rounded, " sec")
    
    y_pred_parcels <- all_preds
    
    
  } else if (onnx_yes) {
    
    warm_batch <- min(parcel_row, batch_number)
    i <- 1:warm_batch
    warm_array_ntd <- data_array[i, , , drop = FALSE]  
    
    warmup_for_onnx(sess = sess, input_name = input_name, data = warm_array_ntd, np = np, output_names = list(output_name), n_runs = 3L)
    
    all_preds <- list()
    chunk_id  <- 1L
    time_t0  <- Sys.time()
    
    for (start in seq.int(1L, parcel_row, by = batch_number)) {
      b <- start + batch_number - 1L
      end  <- min(b, parcel_row)
      row_b <- start:end
      data_arr <- data_array[row_b, , , drop = FALSE]
      data_array_numpy <- np$array(reticulate::r_to_py(data_arr),dtype = np$float32)
    
      output_sess <- sess$run(list(output_name),setNames(list(data_array_numpy), input_name))[[1]]
      out_r <- reticulate::py_to_r(output_sess)   
      pred_max_r <- apply(out_r, 1L, which.max)  
      pred_max_zero <- pred_max_r - 1L
      all_preds[row_b] <- pred_max_zero
      chunk_id <- chunk_id + 1L  
    }
    
    time_t1 <- Sys.time()
    time_t <- time_t1 - time_t0
    time_secs <- as.numeric(time_t, units = "secs")
    time_secs_rounded <- round(time_secs, 3)
    
    message("inference loop: ", time_secs_rounded, " sec")
    y_pred_parcels <- unlist(all_preds, use.names = FALSE)
    
    
    
    
  } else {
    warm_batch <- min(parcel_row, batch_number)
    i <- 1:warm_batch
    warm_array_ntd <- data_array[i, , , drop = FALSE]
    
    warump_for_torch(model = model, data = warm_array_ntd, n_runs = 3L)
    
    all_preds <- list()
    chunk_id  <- 1L
    time_t0  <- Sys.time()
    
    for (start in seq.int(1L, parcel_row, by = batch_number)) {
      batch_end_raw <- start + batch_number - 1L
      batch_end <- min(batch_end_raw, parcel_row)
      batch_row <- start:batch_end
      
      batch_arr <- data_array[batch_row, , , drop = FALSE]  
      
      tens <- torch::torch_tensor(batch_arr, dtype = torch::torch_float())
      
      torch::with_no_grad({
        log_calculation <- model(tens)
      })
      
      argmax_tensor <- torch::torch_argmax(log_calculation, dim = 2L)
      argmax_array <- as.array(argmax_tensor)
      pred_r_based <- as.integer(argmax_array)
      pred <- pred_r_based - 1L
      
      all_preds[[chunk_id]] <- pred
      chunk_id <- chunk_id + 1L
    }
    
    time_t1 <- Sys.time()
    time_t <- time_t1 - time_t0
    time_secs <- as.numeric(time_t, units = "secs")
    time_secs_rounded <- round(time_secs, 3)
    
    message("inference loop: ", time_secs_rounded, " sec")
    
    
    y_pred_parcels <- unlist(all_preds, use.names = FALSE)
  }
  
  
  typeof(y_pred_parcels)
  y_pred_parcels
  
  
  
  class_levels <- classes  
  class_names  <- as.character(area_inference_data$class_name)
  
  
  true_class_id <- unname(class_to_id[class_names])
  pred_class_id <- y_pred_parcels 
  
  y_true_factor <- factor(class_levels[true_class_id + 1L], levels = class_levels)
  y_pred_factor <- factor(class_levels[pred_class_id + 1L], levels = class_levels)
  
  parcel_eval$tempcnn <- list(y_true = y_true_factor,y_pred = y_pred_factor,model = "TempCNN")
  
  
  prediction_data_frame <- data.frame(fid = area_inference_data$fid, pred = y_pred_parcels)
  
  
  match_fig <- match(aoi$fid, prediction_data_frame$fid)
  pred_mapped <- prediction_data_frame$pred[match_fig]
  aoi$pred <- pred_mapped
  
  aoi$pred[is.na(aoi$pred)] <- noData
  
  
  
  r_prediction_raster <- terra::rasterize(terra::vect(aoi), template_for_r,field = "pred", background = noData, touches  = FALSE)
  terra::NAflag(r_prediction_raster) <- noData
  
  terra::writeRaster(r_prediction_raster, output_tif, overwrite = TRUE, filetype  = "GTiff", datatype = "INT2S",  gdal = c("COMPRESS=LZW","TILED=YES","BLOCKXSIZE=256","BLOCKYSIZE=256"),NAflag = noData)
  r_prediction_raster <- terra::rast(output_tif)
  terra::NAflag(r_prediction_raster) <- noData
  
  
  raster_ground_truth_ <- ground_truth_raster_f(ground_truth_data_g, gt_class, class_to_id, template_for_r, epsg, noData)
    
    r_gt <- raster_ground_truth_
    terra::NAflag(r_gt) <- noData
    
    evaluate_prediction_streaming(r_prediction_raster, r_gt, classes, noData)
  }







parcel_evaluation <- function(parcel_eval) {
  for (i in names(parcel_eval)) {
    eval_entry <- parcel_eval[[i]]
    
    confusion_matrix_a <- caret::confusionMatrix(eval_entry$y_pred, eval_entry$y_true)
    
    print(confusion_matrix_a)
    
    confusion_matrix_table <- confusion_matrix_a$table
    confusio_matrix_class <- confusion_matrix_a$byClass
    
    total <- sum(confusion_matrix_table)
    diag_v <- diag(confusion_matrix_table)
    correct <- sum(diag_v)
    wrong <- total - correct
    
    correct_p <- 100 * correct / total
    wrong_p <- 100 * wrong   / total
    
    message("Total samples : ", total)
    message("Correct: ", correct, " (", correct_p, "%)")
    message("Wrong : ", wrong,  " (", wrong_p, "%)")
    
    per_class_support <- rowSums(confusion_matrix_table)      
    per_class_true_postiv <- diag(confusion_matrix_table)         
    per_class_not_correct <- per_class_support - per_class_true_postiv  
    per_class_recall <- per_class_true_postiv / per_class_support   
    
    per_class_frame <- data.frame(class  = rownames(confusion_matrix_table), total = as.integer(per_class_support),
                    correct = as.integer(per_class_true_postiv),
                    wrong = as.integer(per_class_not_correct),
                     acc = per_class_recall)
    
    message("class frame")
    print(per_class_frame)
    

    precision <- confusio_matrix_class[, "Pos Pred Value"] 
    recall <- confusio_matrix_class[, "Sensitivity"] 
    
    
    he <- 2 * precision * recall
    den <- precision + recall
    f1  <- he / den
    
    f1[is.nan(f1)] <- 0
    rownames_con <- rownames(confusio_matrix_class)
    for (i in seq_along(f1)) {
      message("  ", rownames_con[i],"  f1 = ", round(f1[i], 4)," (precision=", round(precision[i], 4), ", recall=",   round(recall[i], 4), ")")
    }
    
    macro_f1 <- mean(f1)
    message("macro-F1: ", round(macro_f1, 4))
    
    macro_balacc <- mean(recall)
    message( "macro Balanced Accuracy (parcel, mean recall): ",round(macro_balacc, 4))
    
  }
}



parcel_evaluation(parcel_eval)


r_prediction_rasteriction <- data.frame(fid = area_inference_data$fid, pred = y_pred_parcels)

arrow::write_parquet(
  r_prediction_rasteriction,"..prediction.."
)





compare_predictions <- function(path) {
  
  prediction_outside <- arrow::read_parquet(path)
  prediction_outside <- prediction_outside[, c("fid", "pred")]
  names(prediction_outside)[2] <- "pred_outside"
  
  pred_merge <- merge(r_prediction_rasteriction, prediction_outside, by = "fid", all = FALSE)
  
  pred_merge$diff <- pred_merge$pred != pred_merge$pred_outside
  
  count_difference  <- sum(pred_merge$diff, na.rm = TRUE)
  
  total_difference <- nrow(pred_merge)
  
  difference_yes <- if (total_difference > 0) 100 * count_difference / total_difference else 0
  
  message(" ", count_difference, " from ", total_difference, " Predictions differ (", round(difference_yes, 2), "%)")
  
  if (count_difference > 0) {
    message(" first diff:")
    difference_head_ten <- utils::head(pred_merge[pred_merge$diff, ], 10)
    lines_o <- capture.output(print(difference_head_ten))
    message(paste(lines_o, collapse = "\n"))
  }
  
  return(pred_merge)
}


compare_predictions(external_prediction)



write_overlay_r <- function(r_prediction_raster, raster_ground_truth, template_for_r, out_path, nodata_val = noData) {
  
  prediction_val <- terra::values(r_prediction_raster, mat = FALSE)
  ground_truth_val <- terra::values(raster_ground_truth, mat = FALSE)
  
  if (length(prediction_val) != length(ground_truth_val)) {
    stop("Prediction and GT grids have different numbers of cells")
  }
  
  prediction_val_length <- length(prediction_val)
  
  red <- rep(255L, prediction_val_length)
  green <- rep(255L, prediction_val_length)
  blue <- rep(255L, prediction_val_length)
  
  valid_pred <- !is.na(prediction_val) & (prediction_val != nodata_val)
  valid_gt <- !is.na(ground_truth_val) & (ground_truth_val != nodata_val)
  valid_both <- valid_pred & valid_gt
  
  correct_mask <- valid_both & (prediction_val == ground_truth_val)
  wrong_mask <- valid_both & (prediction_val != ground_truth_val)
  
  sum_valid_gt <- sum(valid_gt)
  sum_valid_pred <- sum(valid_pred)
  sum_valid_both <- sum(valid_both)
  sum_correct_mask <- sum(correct_mask)
  sum_wrong_mask <- sum(wrong_mask)
  
  message("GT-Pixel:", sum_valid_gt)
  message("Pred-Pixel:", sum_valid_pred)
  message("Compare-Pixel :", sum_valid_both)
  message("Correct Pixel:", sum_correct_mask)
  message("Wrong Pixel :", sum_wrong_mask)
  
  
  red[correct_mask]  <- 0L
  green[correct_mask] <- 0L
  blue[correct_mask] <- 255L
  
  red[wrong_mask] <- 255L
  green[wrong_mask] <- 0L
  blue[wrong_mask] <- 0L
  
 
  r_red <- template_for_r
  r_green <- template_for_r
  r_blue <- template_for_r
  
  terra::values(r_red) <- red
  terra::values(r_green) <- green
  terra::values(r_blue) <- blue
  
  overlay <- c(r_red, r_green, r_blue)
  names(overlay) <- c("red", "green", "blue")
  
  terra::writeRaster(overlay,out_path, overwrite = TRUE,filetype= "GTiff", datatype  = "INT1U", gdal = c("COMPRESS=LZW", "TILED=YES"))
  
  invisible(overlay)
}


ov <- write_overlay_r(
  r_prediction_raster = r_prediction_raster,
  raster_ground_truth = r_gt,
  template_for_r = template_for_r,
  out_path = overlay_tif_pred,
  nodata_val = noData
)

terra::plotRGB(ov, r = 1, g = 2, b = 3, scale = 255)


