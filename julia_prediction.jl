using JSON3
using ONNXRunTime
using ArchGDAL
using Printf
using Flux
using BSON
using DataFrames
using Parquet2
using CSV
using Tables
using Statistics
using StatisticalMeasures
import StatisticalMeasures.ConfusionMatrices as CM
using NNlib: logsoftmax


const inference_data   = "..inference_data.."
const ground_truth_data     = "ground_truth_area_four_prediction"
const target_class = "class_name"




# TempCNN-Artefakte
const model_config      = "..model_config.."
const model_onnx     = "..onnx_model.."
const model_bson     = "..svm_native.."
const class_levels   = "..onnx/native_classes.."
const scaler_params  = "..scaler_data.."

# SVM-ONNX-Artefakte
const svm_onnx       = "..svm_onnx.."
const svm_artifacts  = "..svm_artifacts.."
const classes_py_svm = "..svm_classes..."
const features_path_svm = "..features_svm.."


# Outputs
const out_tif_pred          = "..prediction.."
const out_tif_overlay       = "..overlay.."

const julia_prediction_parquet  = "..path_prediction_parquet"
const external_prediction_path  = "..external_prediction.."




const use_onnx = true          
const use_julia_onnx_layout = false        
const use_svm = false         
const target_epsg = 3857
const resolution_m = 20.0
const nodata_val = -1
const batch_size = 1024





function load_model_artifacts(config, classes_path, scaler_path)
    config_data = JSON3.read(read(config, String))

    classes_data = DataFrame(Parquet2.Dataset(classes_path))

    if :label in names(classes_data)
        classes = classes_data.label
    else
        classes = classes_data[:, 1]
    end

    classes = String.(classes)

    scaler_data = DataFrame(Parquet2.Dataset(scaler_path))

    band_order = String.(config_data["band_order"])

    mean_mu = Float32[]
    std = Float32[]

    for b in band_order
        mask = scaler_data.band .== b
        row_b  = scaler_data[mask, :]

        mean_val = row_b.mean[1]
        mean_val = Float32(mean_val)
        push!(mean_mu, mean_val)

        scale_val = row_b.scale[1]
        scale_val = Float32(scale_val)
        push!(std, scale_val)

    end
    time_raw = get(config_data, "time_steps", 12)  
    time_int = Int(time_raw)                      


    return (band_order = band_order,  mean_mu = mean_mu, std = std, classes = classes, time_int = time_int)
end


function prepare_vector_with_fid(vector_path; epsg = target_epsg)

    if !isfile(vector_path)
        error("vector file not found $vector_path")
    end

    ogr2ogr_v = Sys.which("ogr2ogr")
    tempo_geojson = tempname() * ".geojson"


    file_name = basename(vector_path)
    layer_name, _ = splitext(file_name)

    sql_test_q = [
        "SELECT ROW_NUMBER() OVER () AS fid, * FROM $(layer_name)",
        "SELECT CAST(_rowid_ AS INTEGER) AS fid, * FROM $(layer_name)",
        "SELECT FID AS fid, * FROM $(layer_name)",
        "SELECT OGR_FID AS fid, * FROM $(layer_name)"
    ]

    last_error = nothing

    for q in sql_test_q
        if isfile(tempo_geojson)
            rm(tempo_geojson; force=true)
        end
        #comando for ogr2ogr, applies to the others as well
        cmd = Cmd([
            ogr2ogr_v,
            "-f", "GeoJSON",
            tempo_geojson,
            vector_path,
            "-t_srs", "EPSG:$(epsg)",
            "-dialect", "SQLite",
            "-sql", q
        ])

        try
            run(cmd)
            if isfile(tempo_geojson) && filesize(tempo_geojson) > 0
                return tempo_geojson
            end
        catch e
            last_error = e
        end
    end

    error("error in ogr2ogr...: $(last_error)")
end




#grid 
function build_raster(vector_path, res, epsg = target_epsg)

    data_with_fid = prepare_vector_with_fid(vector_path; epsg=epsg)

    ArchGDAL.read(data_with_fid) do ds
        layer = ArchGDAL.getlayer(ds, 0)

        env = ArchGDAL.envelope(layer, true)

        xmin = env.MinX
        xmax = env.MaxX
        ymin = env.MinY
        ymax = env.MaxY

        diff_x = xmax - xmin
        width_in_cells = diff_x / res
        width = ceil(Int, width_in_cells)

        diff_y = ymax - ymin
        height_in_cells = diff_y / res
        height = ceil(Int, height_in_cells)

        ground_truth = (xmin, res, 0.0, ymax, 0.0, -res)

        println("raster grid: width=$(width) height=$(height) res=$(res)")
        return (height, width, ground_truth, data_with_fid)
    end
end





function read_band_gdal(data; band_index  = 1, type_arr::Type=Int32)
    ArchGDAL.read(data) do dataset
        band = ArchGDAL.getband(dataset, band_index)
        width  = ArchGDAL.width(dataset)
        height = ArchGDAL.height(dataset)
        arr = Array{type_arr}(undef, width, height)   
        ArchGDAL.read!(band, arr)
        return arr
    end
end


function rasterize_fid(vector_path, height, width, ground_truth; epsg = target_epsg, noData = nodata_val)
    tempo_tif = tempname() * ".tif"

    xmin, pixel_width, _, ymax, _, pixel_height_neg = ground_truth

    pixel_size_x = pixel_width
    pixel_size_y = -pixel_height_neg
    xmax = xmin + width * pixel_size_x
    ymin = ymax - height * pixel_size_y

    gdal_rast = Sys.which("gdal_rasterize")

    #string - input 
    cmd = Cmd([
        gdal_rast,
        "-a", "fid",
        "-ot", "Int32",
        "-a_nodata", string(noData),
        "-te", string(xmin), string(ymin), string(xmax), string(ymax),
        "-tr", string(pixel_size_x), string(pixel_size_y),
        "-tap",
        "-a_srs", "EPSG:$(epsg)",
        vector_path,
        tempo_tif
    ])

    run(cmd)


    arr_data = read_band_gdal(tempo_tif; band_index = 1, type_arr = Int32)
    arr_data[arr_data .== 0] .= noData  

    return arr_data


end



function build_class_id_in_sql(layername, classes; class_field, noData)
    when_then_clause = String[]

    for (fid, class_name) in enumerate(classes)
        label_id = fid - 1
        class_name = replace(class_name, "'" => "''") 
        push!(when_then_clause,
              "WHEN $(class_field) = '$(class_name)' THEN $(label_id)")
    end

    case_body = join(when_then_clause, "\n")

    sql =  """
    SELECT
        *,
        CASE
            $case_body
            ELSE $(noData)
        END AS cls_id
    FROM $(layername)
    """
    return sql
end



function rasterize_gt(vector_path, classes, height, width, gt; class_field = target_class, epsg = target_epsg, noData = nodata_val)

    ogr2ogr_v = Sys.which("ogr2ogr")
    gdal_rast = Sys.which("gdal_rasterize")


    file_name = basename(vector_path)
    layername, _ = splitext(file_name)

    tempo_geojson = tempname() * ".geojson"
    tempo_tif = tempname() * ".tif"
    sql = build_class_id_in_sql(layername, classes; class_field=class_field, noData=noData)

    cmd = Cmd([ogr2ogr_v, "-f", "GeoJSON", tempo_geojson, vector_path, "-t_srs", "EPSG:$(epsg)",  
                "-dialect", "SQLite", "-sql", sql])
    run(cmd)



    xmin, pixel_width, _, ymax, _, pixel_height_neg = gt
    pixel_size_x = pixel_width
    pixel_size_y = -pixel_height_neg
    xmax = xmin + width * pixel_size_x
    ymin = ymax - height * pixel_size_y


    cmd2 = Cmd([gdal_rast, "-a", "cls_id", "-ot", "Int16", "-a_nodata", string(noData), "-te", string(xmin), string(ymin), string(xmax), string(ymax),
                "-tr", string(pixel_size_x), string(pixel_size_y), "-tap", "-a_srs", "EPSG:$(epsg)", tempo_geojson, tempo_tif])

    run(cmd2)

    arr_data = read_band_gdal(tempo_tif; band_index=1, type_arr=Int16)
    return arr_data

end




#tif

#helper


function write_tiff(out_path, pred_raster, ground_truth; epsg = target_epsg, noData = nodata_val)

    width, height = size(pred_raster)

    mkpath(dirname(out_path))

    options_r = ["TILED=YES", "COMPRESS=LZW", "BLOCKXSIZE=256", "BLOCKYSIZE=256"]

    ground_truth_vec = Float64[ground_truth[1], ground_truth[2], ground_truth[3], ground_truth[4], ground_truth[5], ground_truth[6]]

    ArchGDAL.create(out_path; driver=ArchGDAL.getdriver("GTiff"),
                    width = width, height = height, nbands = 1, dtype = Int16, options = options_r) do dataset
        ArchGDAL.setgeotransform!(dataset, ground_truth_vec)
        epsg_ = ArchGDAL.importEPSG(epsg)
        ArchGDAL.setproj!(dataset, ArchGDAL.toWKT(epsg_))
        band = ArchGDAL.getband(dataset, 1)
        ArchGDAL.write!(band, pred_raster)
        ArchGDAL.setnodatavalue!(band, Float64(noData))
    end


end




function write_overlay_tiff(output_path, prediction, ground_truth_arr, gt; epsg = target_epsg, nodata = nodata_val)

    size(prediction) == size(ground_truth_arr) || error("pred und gtarr müssen die gleiche Größe haben.")

    width, height = size(prediction)
    mkpath(dirname(output_path))

    options_r = ["TILED=YES", "COMPRESS=LZW", "BLOCKXSIZE=256", "BLOCKYSIZE=256"]
    ground_truth_vec = Float64[gt[1], gt[2], gt[3], gt[4], gt[5], gt[6]]


    red = Array{UInt8}(undef, width, height)
    green = Array{UInt8}(undef, width, height)
    blue = Array{UInt8}(undef, width, height)

    fill!(red, 255)
    fill!(green, 255)
    fill!(blue, 255)

    valid_prediction = prediction .!= nodata
    valid_ground_truth   = ground_truth_arr.!= nodata
    valid_b = valid_prediction .& valid_ground_truth

    correct_mask = valid_b .& (prediction .== ground_truth_arr)
    wrong_mask   = valid_b .& (prediction .!= ground_truth_arr)

    red[correct_mask] .= 0
    green[correct_mask] .= 0
    blue[correct_mask] .= 255

    red[wrong_mask] .= 255
    green[wrong_mask] .= 0
    blue[wrong_mask] .= 0

    ArchGDAL.create(output_path; driver = ArchGDAL.getdriver("GTiff"), width = width, height = height, nbands = 3, dtype = UInt8, options = options_r) do dataset
        ArchGDAL.setgeotransform!(dataset, ground_truth_vec)
        epsg_ = ArchGDAL.importEPSG(epsg)
        ArchGDAL.setproj!(dataset, ArchGDAL.toWKT(epsg_))
        ArchGDAL.write!(ArchGDAL.getband(dataset, 1), red)
        ArchGDAL.write!(ArchGDAL.getband(dataset, 2), green)
        ArchGDAL.write!(ArchGDAL.getband(dataset, 3), blue)
    end



    sum_valid_ground_truth = sum(valid_ground_truth)
    sum_valid_prediction = sum(valid_prediction)
    sum_valid_both = sum(valid_b)
    sum_correct  = sum(correct_mask)
    sum_wrong = sum(wrong_mask)

    println("ground truth-Pixel: ",sum_valid_ground_truth)
    println("prediction-Pixel : ", sum_valid_prediction)
    println("compare-Pixel: ", sum_valid_both)
    println("Correct Pixel : ", sum_correct)
    println("Wrong Pixel : ", sum_wrong)

end



function _ort_session(onnx_path::AbstractString; execution_provider = :cpu)
    sess = ONNXRunTime.load_inference(onnx_path; execution_provider = execution_provider)
    return sess
end





function predict_parcels_onnx(inference_data, onnx_path; julia_layout = false, batch_size = 1024)

    sess = _ort_session(onnx_path)            
    input_name = ONNXRunTime.input_names(sess)[1]
    output_name = ONNXRunTime.output_names(sess)[1]

    input_symbol  = Symbol(input_name)
    out_symbol = Symbol(output_name)

    n_size = size(inference_data, 1)  
    

    if julia_layout
        inference_data_view = @view(inference_data[1, :, :])
        inference_data_permute = permutedims(inference_data_view, (2, 1))


        inference_data_permute = reshape(inference_data_permute, 1, size(inference_data_permute, 1), size(inference_data_permute, 2))      
        for _ in 1:3
             sess(NamedTuple{(input_symbol,)}((inference_data_permute,)))
        end

    else
        stop_i = min(n_size, batch_size)
        first_batch = @view inference_data[1:stop_i, :, :]
        for _ in 1:3
             sess(NamedTuple{(input_symbol,)}((first_batch,)))
        end
    end

    t0 = time()

    if julia_layout
        prediction = Vector{Int32}(undef, n_size)
        for i in 1:n_size
            inference_data_view = @view inference_data[i, :, :]
            inference_data_permute = permutedims(inference_data_view, (2, 1))
            inference_data_permute = reshape(inference_data_permute, 1, size(inference_data_permute, 1), size(inference_data_permute, 2)) 
            out = sess(NamedTuple{(input_symbol,)}((inference_data_permute,)))
            
            logits = out[out_symbol]
            best_id = argmax(logits[1, :])
            prediction[i] = Int32(best_id - 1)
        end
        t1 = time()
        time_t = t1 - t0
        println("onnx inference-loop = ", time_t, " s")
        return prediction
    end

    all_prediction = Vector{Vector{Int32}}()

    for start in 1:batch_size:n_size
        stop_b = min(start + batch_size - 1, n_size)
        inference_batch = @view inference_data[start:stop_b, :, :]            
        out = sess(NamedTuple{(input_symbol,)}((inference_batch,)))
        logits = out[out_symbol]
        best_id = map(argmax, eachrow(logits))
        batch_prediction = Int32.(best_id .- 1)
        push!(all_prediction, batch_prediction)
    end
    t1 = time()
    time_t = t1 - t0
    println("onnx inference-loop = ", time_t, " s")
    return vcat(all_prediction...)
end







function predict_flux(inference_data, bson_path, k; batch_size = 1024)

    dict_data = BSON.load(bson_path)
    @assert haskey(dict_data, :model) 
    model_ = dict_data[:model]

    Flux.testmode!(model_, true)

    n_size = size(inference_data, 1)
    all_predictions = Vector{Vector{Int32}}()

    stop_id = min(n_size, batch_size)
    first_batch = @view inference_data[1:stop_id, :, :]

    for _ in 1:3
        Base.invokelatest(model_, Array(first_batch))  
    end
    
    t0 = time()
    for start in 1:batch_size:n_size
        stop_b = min(start + batch_size - 1, n_size)
        inference_batch = @view inference_data[start:stop_b, :, :]
        logits = Base.invokelatest(model_, Array(inference_batch))
        labels = Flux.onecold(logits, 0:(k-1))
        batch_prediction = Int32.(labels)
        push!(all_predictions, batch_prediction)
    end
    t1 = time()
    time_t = t1 - t0
    println("flux inference-loop = ", time_t, " s")

    return vcat(all_predictions...)
end


model_for_export(m) = Chain(m.layers[2:end]...)



function build_export_chain(model_m, time_s)
    base = model_for_export(model_m)          
    layers = collect(base.layers)
    layers[13] = x -> reshape(x, (HIDDEN_DIMS*time_s, 1))
    return Chain(layers...)
end


function compare_flux_vs_onnx(bson_path, onnx_path, T, D; n_samples = 1024)

    dict_data = BSON.load(bson_path)
    @assert haskey(dict_data, :model) "BSON enthält kein :model"
    model_ = dict_data[:model]
    Flux.testmode!(model_, true)

    mex = build_export_chain(model_, T)

    sess = _ort_session(onnx_path)            
    
    input_name  = ONNXRunTime.input_names(sess)[1]
    output_name = ONNXRunTime.output_names(sess)[1]
    input_symbol   = Symbol(input_name)
    output_symbol  = Symbol(output_name)



    data_flux_rand = rand(Float32, T, D, 1)                
    output_flux = Base.invokelatest(mex, data_flux_rand)
    prediction_flux = Array(output_flux)


    data_onnx = permutedims(data_flux_rand, (3, 2, 1))      
    prediction_onnx    = sess(NamedTuple{(input_symbol,)}((data_onnx,)))
    prediction_onnx_arr = Array(prediction_onnx[output_symbol])                 


    size_prediction_flux = size(prediction_flux)
    size_prediction_onnx = size(prediction_onnx_arr)
    if size(prediction_onnx_arr) == (1, size(prediction_flux, 1))
        prediction_onnx_arr = permutedims(prediction_onnx_arr, (2, 1))    
    elseif size(prediction_onnx_arr) != size(prediction_flux)
        @warn "Output shape different for the first sample" flux = size_prediction_flux onnx = size_prediction_onnx
    end

    k_size = size(prediction_flux, 1)
    flux_logits_by_sample = Array{Float32}(undef, k_size, n_samples)
    onnx_logits_by_sample = Array{Float32}(undef, k_size, n_samples)

    flux_logits_by_sample[:, 1] .= vec(prediction_flux)
    onnx_logits_by_sample[:, 1] .= vec(prediction_onnx_arr)



    for i in 2:n_samples
        data_rand = rand(Float32, T, D, 1)

        prediction_flux = Array(Base.invokelatest(mex, data_rand)) 

        data_onnx_ = permutedims(data_rand, (3, 2, 1))         
        out    = sess(NamedTuple{(input_symbol,)}((data_onnx_,)))
        prediction_onnx_ = Array(out[output_symbol])
        size_flux = size(prediction_flux)
        size_onnx = size(prediction_onnx_)

        if size(prediction_onnx_) == (1, k_size)
            prediction_onnx_ = permutedims(prediction_onnx_, (2,1))     
        elseif size(prediction_onnx_) != size(prediction_flux)
            @warn "Output-Shape unterschiedlich bei Sample $i" flux = size_flux onnx = size_onnx
        end
        onnx_logits_by_sample[:, i] .= vec(prediction_onnx_)
        flux_logits_by_sample[:, i] .= vec(prediction_flux)
    end





    #compare

    abs_logit_diff = abs.(flux_logits_by_sample .- onnx_logits_by_sample)
    max_abs_logit_diff  = maximum(abs_logit_diff)
    mean_abs_logit_diff = mean(abs_logit_diff)


    println("ONNX vs Flux — max = ", max_abs_logit_diff, ", mean = ", mean_abs_logit_diff)

    pred_flux_logits = map(argmax, eachcol(flux_logits_by_sample))   
    pred_onnx_logits = map(argmax, eachcol(onnx_logits_by_sample))
    n_total = length(pred_flux_logits)


    equal_mask = pred_flux_logits .== pred_onnx_logits
    n_equal_logits = count(equal_mask)
    p_logits = 0.0
    if n_total > 0
        p_logits = 100 * n_equal_logits / n_total
    end

    println("argmax logits ", n_equal_logits, " / ", n_total, " identical predictions (", p_logits, "%)")


    #logsoftmax


    logsoftmax_flux = logsoftmax(flux_logits_by_sample; dims=1)  
    logsoftmax_onnx = logsoftmax(onnx_logits_by_sample; dims=1)

    difference_logsoftmax = abs.(logsoftmax_flux .- logsoftmax_onnx)
    max_abs_logsoftmax_diff  = maximum(difference_logsoftmax)
    mean_abs_logsoftmax_diff = mean(difference_logsoftmax)
    println("ONNX vs Flux — max = ", max_abs_logsoftmax_diff, ", mean = ", mean_abs_logsoftmax_diff)

    #argmax logsoftmax
    prediction_logsoftmax_flux = map(argmax, eachcol(logsoftmax_flux))
    prediction_logsoftmax_onnx = map(argmax, eachcol(logsoftmax_onnx))
    equal_logsm = prediction_logsoftmax_flux .== prediction_logsoftmax_onnx
    n_equal_logsm = count(equal_logsm)

    p_logsoftmax = 0.0
    if n_total > 0
        p_logsoftmax = 100 * n_equal_logsm / n_total
    end
    println(" argmax-logsoftmax ", n_equal_logsm, " / ", n_total, " identical predictions (", p_logsoftmax, "%)")


    return (
        max_diff_raw = max_abs_logit_diff,
        mean_diff_raw = mean_abs_logit_diff,
        max_diff_logsoftmax = max_abs_logsoftmax_diff,
        mean_diff_logsoftmax = mean_abs_logsoftmax_diff,
        n_equal_logits = n_equal_logits,
        n_equal_logsm = n_equal_logsm,
        n_total = n_total,
        y_flux_logits = flux_logits_by_sample,
        y_onnx_logits = onnx_logits_by_sample,
        y_flux_logprobs = logsoftmax_flux,
        y_onnx_logprobs = logsoftmax_onnx,
    )
end





function per_class_prf1_framework(pred, gt_true , classes )
    confusion_matrix__ = CM.confmat(pred, gt_true; levels=classes)
    confusion_matrix_  = CM.matrix(confusion_matrix__)

    length_class = length(classes)
    prec = zeros(Float64, length_class)
    rec  = zeros(Float64, length_class)
    f1   = zeros(Float64, length_class)

    for i in 1:length_class
        f = NaN
        r = NaN
        p = NaN


        true_positiv = confusion_matrix_[i, i]
        false_positive = sum(confusion_matrix_[:, i]) - true_positiv
        fale_negative = sum(confusion_matrix_[i, :]) - true_positiv

        precision_d = true_positiv + false_positive
        if precision_d != 0
            p = true_positiv / precision_d
        end

        recall_d = true_positiv + fale_negative
        if recall_d != 0
            r = true_positiv / recall_d
        end

        precision_recall_sum = p + r
        if !isnan(p) && !isnan(r) && precision_recall_sum != 0
            f = 2 * p * r / precision_recall_sum
        end

        prec[i] = p
        rec[i]  = r
        f1[i]   = f
    end

    return confusion_matrix__, confusion_matrix_, prec, rec, f1
end



function print_correct_wrong_stats(numerical_confusion_matrix, classes, level_name)
    total   = sum(numerical_confusion_matrix)
    diag_cm = diag(numerical_confusion_matrix)
    correct = sum(diag_cm)
    wrong   = total - correct

    println("Overall correctness ($(level_name))")
    
    println("Total samples : $(total)")

    correct_p_round = round(100 * correct / total; digits=2)
    println("Correct: $(correct) ($(correct_p_round)%)")

    wrong_p_round = round(100 * wrong / total; digits=2)
    println("Wrong : $(wrong) ($(wrong_p_round)%)")

    per_class_total = vec(sum(numerical_confusion_matrix, dims=2))
    per_class_wrong = per_class_total .- diag_cm
    per_class_acc = similar(diag_cm, Float64)
    for i in 1:length(classes)
        total_ = per_class_total[i]
        per_class_acc[i] = total_ == 0 ? 0.0 : diag_cm[i] / total_
    end

    println(rpad("class", 20), " ", rpad("total", 10), " ", rpad("correct", 10), " ", rpad("wrong", 10), " ", rpad("acc", 10))

    for i in 1:length(classes)
        println(rpad(string(classes[i]), 20), " ", lpad(string(Int(per_class_total[i])), 10), " ", lpad(string(Int(diag_cm[i])), 10), " ", lpad(string(Int(per_class_wrong[i])), 10), " ", lpad(string(round(per_class_acc[i]; digits=4)), 10))
    end
end





function parcel_report_statmeas(gt_true, pred, classes)

    cm_num, confusion_matrix_, prec, rec, f1 = per_class_prf1_framework(pred, gt_true, classes)

    print_correct_wrong_stats(confusion_matrix_, classes, "parcel-level")

    println("Parcel-level report ")
    println(cm_num)

    acc   = accuracy(pred, gt_true)
    bacc  = balanced_accuracy(pred, gt_true)
    m_f1  = multiclass_f1score(pred, gt_true)

    println(rpad("class", 20), " ", rpad("prec", 10), " ", rpad("rec", 10), " ", rpad("f1", 10), " ", rpad("support", 10))


    for (i, class) in enumerate(classes)
        sum_con = sum(confusion_matrix_[i, :])

        println(rpad(string(class), 20), " ", lpad(string(round(prec[i]; digits=4)), 10), " ", lpad(string(round(rec[i];  digits=4)), 10), " ", lpad(string(round(f1[i]; digits=4)), 10), " ", lpad(string(Int(sum_con)), 10))

    end

    println("accuracy : $(round(acc;  digits=4))")
    println("bal accuracy: $(round(bacc; digits=4))")
    println("macro F1 : $(round(m_f1; digits=4))")
end




function evaluate_prediction_statmeas(pred, ground_truth_data, classes; noData = nodata_val)

    pred_v = vec(pred)
    gt_v = vec(ground_truth_data)
    mask = (pred_v .!= noData) .& (gt_v .!= noData)

    n_valid = count(mask)
    pred_lab = Vector{String}(undef, n_valid)
    ground_truth_lab = Vector{String}(undef, n_valid)

    id = 1
    for i in eachindex(pred_v)
        if mask[i]
            pred_lab[id] = classes[pred_v[i] + 1]
            ground_truth_lab[id] = classes[gt_v[i] + 1]
            id += 1
        end
    end
    cm_num, confusion_matrix_, _, _, _ = per_class_prf1_framework(pred_lab, ground_truth_lab, classes)

    acc   = accuracy(pred_lab, ground_truth_lab)
    bacc  = balanced_accuracy(pred_lab, ground_truth_lab)
    m_f1  = multiclass_f1score(pred_lab, ground_truth_lab)

    print_correct_wrong_stats(confusion_matrix_, classes, "pixel-level")

    return confusion_matrix_, acc, bacc, m_f1
end






function compare_with_external(data_compare)


    if !isfile(external_prediction_path)
        @warn "External prediction not found: $(external_prediction_path)"
        return
    end

    dataset = Parquet2.Dataset(external_prediction_path)
    ext_df = DataFrame(dataset)


    names_ext = names(ext_df)
    name_map = Dict{String,Any}()
    for i in names_ext
        c_strip = strip(i)
        key = lowercase(c_strip)
        name_map[key] = i
    end

    fid_col = get(name_map, "fid", nothing)
    
    pred_c = get(name_map, "pred", nothing)
    
    rename_map = Pair[]
    push!(rename_map, fid_col => :fid_ext)
    push!(rename_map, pred_c  => :pred_ext)


    ext_work = select(ext_df, rename_map...)

    ext_work.fid_ext  = Int64.(ext_work.fid_ext)
    ext_work.pred_ext = Int64.(ext_work.pred_ext)




    julia_preds = select(data_compare, :fid, :pred)
    # missing detection 
    if !(eltype(julia_preds.fid) <: Integer)
        julia_preds = dropmissing(julia_preds, :fid)
    end
    if !(eltype(julia_preds.pred) <: Integer)
        julia_preds = dropmissing(julia_preds, :pred)
    end


    julia_preds.fid  = Int64.(julia_preds.fid)
    julia_preds.pred = Int64.(julia_preds.pred)

    join_fid = [:fid => :fid_ext]

    # 2) Inner Join ausführen
    joined_preds = innerjoin(julia_preds, ext_work; on = join_fid)

    if nrow(joined_preds) == 0
        @warn "no fids to compare"
        return
    end

    difference_mask = joined_preds.pred .!= joined_preds.pred_ext
    number_difference = count(difference_mask)
    number_total  = nrow(joined_preds)

    pct = number_total > 0 ? 100 * number_difference / number_total : 0.0
    println("compare ", number_difference, " from ", number_total," diff(", round(pct; digits=2), "%)")



    if number_difference > 0
        println("first difference")
        show(first(joined_preds[difference_mask, :], min(10, number_difference)); allcols=true)
        println()
    end
end





function main()

 
    if use_svm
        println("svm onnx julia...")

        dataset_class = Parquet2.Dataset(classes_py_svm)

        svm_class = DataFrame(dataset_class)


        svm_classes = String.(svm_class.label)

        features_svm = JSON3.read(read(features_path_svm, String))
        feature_order = String.(features_svm["features"])

        inference_data_dataset = Parquet2.Dataset(inference_data)

        inference_data_df = DataFrame(inference_data_dataset)

        rawnames = names(inference_data_df)


        cleaned = Symbol[]

        for nm in rawnames
            s = String(nm)
            s1 = replace(s, "\r" => "")
            s2 = replace(s1, "\n" => "")
            s = strip(s2)
            push!(cleaned, Symbol(s))
        end

        rename_pairs_save = Pair.(rawnames, cleaned)
        rename!(inference_data_df, rename_pairs_save)

        fid_col = :fid
        n_row_inference_data = nrow(inference_data_df)
        length_feature = length(feature_order)



        inference_data_arr = Array{Float32}(undef, n_row_inference_data, length_feature)

        for (j, feat) in enumerate(feature_order)
            colsym = Symbol(feat)  
            inference_data_arr[:, j] = Float32.(inference_data_df[!, colsym])
        end


        println("Raster-Grid bauen …")
        height, width, gt, v_fix = build_raster(ground_truth_data, resolution_m, target_epsg)

        sess = _ort_session(svm_onnx)
        input_name   = ONNXRunTime.input_names(sess)[1]
        output_name = ONNXRunTime.output_names(sess)
        label_name = output_name[1]
        input_symbol  = Symbol(input_name)
        label_symbol = Symbol(label_name)
       

        all_prediction = Vector{Int32}(undef, n_row_inference_data)
        batch_start = 1

        t0 = time()
        while batch_start <= n_row_inference_data
            stop_b = min(batch_start + batch_size - 1, n_row_inference_data)
            inference_batch = @view inference_data_arr[batch_start:stop_b, :]

            out = sess(NamedTuple{(input_symbol,)}((inference_batch,)))
            labels_output = out[label_symbol]
            labels_arr = Array(labels_output)
            labels_vec = vec(labels_arr)

            for (k_, val) in enumerate(labels_vec)
                all_prediction[batch_start + k_ - 1] = val
            end
            println("[pred] $(stop_b) / $(n_row_inference_data)")
            batch_start = stop_b + 1
        end
        t1 = time()


        time_t = t1 - t0
        println("svm-onnx inference-loop total = $(round(time_t; digits=4)) s")

        length_prediction = length(all_prediction)
        pred_label = Vector{String}(undef,length_prediction)
        for (i, p) in pairs(all_prediction)
            pred_label[i] = svm_classes[p + 1]
        end

        svm_pred_df = DataFrame(
            fid = inference_data_df.fid,
            pred = all_prediction,
            pred_label = pred_label,
        )

        out_df = svm_pred_df[:, [:fid, :pred]]

        Parquet2.writefile(julia_prediction_parquet, out_df)
        
        true_label = String.(inference_data_df[!, :class_name])

        prediction_label = Vector{String}(undef, length(all_prediction))
        for i in eachindex(all_prediction)
            prediction_label[i] = svm_classes[all_prediction[i] + 1]
        end

        parcel_report_statmeas(true_label, prediction_label, svm_classes)



        raster_fid = rasterize_fid(v_fix, height, width, gt; epsg = target_epsg, noData = nodata_val)
        flat_fid = reshape(raster_fid, :, 1)
        fid_to_class = Dict{Int32,Int16}()

        for i in 1:n_row_inference_data
            fid_to_class[inference_data_df.fid[i]] = all_prediction[i]
        end
        flat_fid_length = length(flat_fid)
        flat_class = Vector{Int16}(undef, flat_fid_length)

        @inbounds for i in eachindex(flat_fid)
            f = flat_fid[i]
            if f == -1
                flat_class[i] = nodata_val
            else
                flat_class[i] = get(fid_to_class, f, nodata_val)
            end
        end

        pred_wide_2 = reshape(flat_class, width, height)


        write_tiff(out_tif_pred, pred_wide_2, gt; epsg=target_epsg, noData = nodata_val)

        println("prediction-tif  $(out_tif_pred)")


        raster_gt = rasterize_gt(ground_truth_data, svm_classes, height, width, gt; class_field = target_class, epsg = target_epsg, noData = nodata_val)
        cm, acc, balacc, macro_f1 = evaluate_prediction_statmeas(pred_wide_2, raster_gt, svm_classes; noData = nodata_val)

        println("pixel-level: acc=$(round(acc, digits=4)) balacc=$(round(balacc, digits=4)) macro_f1=$(round(macro_f1, digits=4))")


        write_overlay_tiff(out_tif_overlay, pred_wide_2, raster_gt, gt; epsg = target_epsg, nodata = nodata_val)

        println("overlay $(out_tif_overlay)")
        compare_with_external(svm_pred_df)

        return
    end




#tempcnn 


    arttifacts_tempcnn = load_model_artifacts(model_config, class_levels, scaler_params)
    band_order = arttifacts_tempcnn.band_order
    classes = arttifacts_tempcnn.classes
    t_expected  = arttifacts_tempcnn.time_int
    length_class = length(classes)
    length_band = length(band_order)

    dataset_inference = Parquet2.Dataset(inference_data)
    inference_data_ = DataFrame(dataset_inference)

    rawnames = names(inference_data_)
    cleaned = Symbol[]
    for nm in rawnames
        s = String(nm)
        s = strip(replace(replace(s, "\r" => ""), "\n" => ""))
        push!(cleaned, Symbol(s))
    end
    rename!(inference_data_, Pair.(rawnames, cleaned))


    n_row_inference_data = nrow(inference_data_)
    println("n parcels (wide): $n_row_inference_data")

    name_map = Dict{String,Symbol}()
    for nm in names(inference_data_)
        name_map[lowercase(String(nm))] = Symbol(nm)
    end


    function find_feature_name(b, t)
        return Symbol("$(b)_T$(t)")
    end

    inference_data_arr = Array{Float32,3}(undef, n_row_inference_data, t_expected, length_band)

    for t in 1:t_expected
        for (j, b) in enumerate(band_order)
            colsym = find_feature_name(b, t)
            inference_data_arr[:, t, j] = Float32.(inference_data_[!, colsym])
        end
    end


   for j in 1:length_band
    mu  = arttifacts_tempcnn.mean_mu[j]
    std_ = arttifacts_tempcnn.std[j]

        for i in 1:size(inference_data_arr, 1)      
            for t in 1:size(inference_data_arr, 2)  
                inference_data_arr[i, t, j] = (inference_data_arr[i, t, j] - mu) / std_
            end
        end
    end


    if use_onnx && isfile(model_bson) && isfile(model_onnx)
        if use_julia_onnx_layout
            compare_flux_vs_onnx(model_bson, model_onnx, t_expected, length_band; n_samples = 1024)
        end
    end


    y_parcels = use_onnx ?
        predict_parcels_onnx(inference_data_arr, model_onnx; julia_layout = use_julia_onnx_layout, batch_size=batch_size) :
        predict_flux(inference_data_arr, model_bson, length_class; batch_size = batch_size)

    # predictions als csv

    prediction_label_tempcnn = Vector{String}(undef, length(y_parcels))
    for i in eachindex(y_parcels)
        p = y_parcels[i]
        prediction_label_tempcnn[i] = classes[p + 1]
    end

    tempcnn_pred_df = DataFrame(fid = inference_data_.fid, pred = y_parcels, prediction_label_tempcnn = prediction_label_tempcnn)

    # parquet
    output_df = tempcnn_pred_df[:, [:fid, :pred]]
    Parquet2.writefile(julia_prediction_parquet, output_df)


    true_label = String.(inference_data_[!, :class_name])

    prediction_label = Vector{String}(undef, length(y_parcels))
    for i in eachindex(y_parcels)
        prediction_label[i] = classes[y_parcels[i] + 1]
    end

    parcel_report_statmeas(true_label, prediction_label, classes)


    height, width, gt, v_fix = build_raster(ground_truth_data, resolution_m, target_epsg)

    raster_fid = rasterize_fid(v_fix, height, width, gt; epsg=target_epsg, noData = nodata_val)

    fid_vec = Int32(inference_data_.fid)
#type conversion for int32 to int 16 in the dictionary, so that both are the same    
    fid_to_class = Dict{Int32, Int16}()
    for i in 1:n_row_inference_data
        fid_to_class[fid_vec[i]] = Int16(y_parcels[i])
    end

    flat_fid = reshape(raster_fid, :, 1)[:]
    fid_cla = Vector{Int16}(undef, length(flat_fid))

    @inbounds for i in eachindex(flat_fid)
        f = flat_fid[i]
        if f == -1
            fid_cla[i] = nodata_val
        else
            fid_cla[i] = get(fid_to_class, f, nodata_val)
        end
    end
    pred_wide_2 = reshape(fid_cla, width, height)

    write_tiff(out_tif_pred, pred_wide_2, gt; epsg=target_epsg, noData = nodata_val)

    raster_gt = rasterize_gt(ground_truth_data, classes, height, width, gt; class_field = target_class, epsg = target_epsg, noData = nodata_val)

    cm, acc, balacc, macro_f1 = evaluate_prediction_statmeas(pred_wide_2, raster_gt, classes; noData = nodata_val)

    println("acc=$(round(acc, digits=4)), balacc=$(round(balacc, digits=4)), macro_f1=$(round(macro_f1, digits=4))")

    write_overlay_tiff(out_tif_overlay, pred_wide_2, raster_gt, gt; epsg=target_epsg, nodata = nodata_val)

    compare_with_external(tempcnn_pred_df)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
