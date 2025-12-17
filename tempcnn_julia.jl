using DataFrames, Random, Statistics
using Flux, Optimisers
using JSON3, BSON
using Printf
using Parquet2
using Parquet
using ONNXNaiveNASflux
using StatisticalMeasures
import StatisticalMeasures.ConfusionMatrices as CM

const trainings_data = "..trainings_data.."
const validatio_data   = "..validation_data.."

const target_ = "class_name"
const id_col = "fid"
const band_pattern  = r"^(B0?\d{2}|NDVI)_T\d+$"



function parameters_tempcnn_model()
    return Dict{String, Any}(
        "hidden_dims" => 128,
        "kernel_size" => 7,
        "dropout" => 0.18203942949809093,
        "lr" => 5e-4,
        "weight_decay" => 1e-6,
        "lr_gamma" => 0.95,
        "batch_size" => 64,
        "epochs" => 15,
        "patience" => 5,
        "seed" => 1337,
    )
end

const p = parameters_tempcnn_model()

const hidden_dims = p["hidden_dims"]
const kernel_size = p["kernel_size"]
const dropout = p["dropout"]
const lr = p["lr"]
const weight_decay = p["weight_decay"]
const lr_gamma = p["lr_gamma"]
const batch_size = p["batch_size"]
const epochs = p["epochs"]
const patience = p["patience"]
const seed = p["seed"]

const out_prefix = "tempcnn"
const out_dir = "."
const out_scaler_parquet = joinpath(out_dir, "scaler_params_tempcnn_bands_julia.parquet")
const out_class_parquet = joinpath(out_dir, "class_levels_from_python_julia.parquet")
const out_cfg_json = joinpath(out_dir, "model_config_julia.json")
const out_params_bson = joinpath(out_dir, "tempcnn_julia_state.bson")
const out_metrics_json = joinpath(out_dir, "metrics_train_val.json")
const out_onnx = joinpath(out_dir, "tempcnn_julia.onnx")



function read_parquet(path)
    try
        return DataFrame(Parquet2.Dataset(path))
    catch e
        @warn "parquet unreadable" file=path error=e
    end
end


function write_parquet(data, path)
    mkpath(dirname(path))
    try
        Parquet2.writefile(path, data)
    catch e
        @warn "Parquet2.writefile error" file=path error=e
    end
    return path
end


function identify_predictors(col_; pattern = band_pattern)
    out = []
    for i in col_
        if occursin(pattern, i)
            push!(out,i )
        end
    end
    return out
end

function detect_bandset_and_time(feature_)

    bands = Set{String}()
    time_s = Set{Int}()

    for c in feature_
        if !occursin(band_pattern, c); continue; end
        match_ = match(r"_T(\d+)$", c)
        if match_ !== nothing
            push!(time_s, parse(Int, match_.captures[1]))
            push!(bands, replace(c, r"_T\d+$" => ""))
        end
    end

    band_vec = collect(bands)
    sort!(band_vec)
    time_m = maximum(time_s)
    return band_vec, time_m
end

# to ensure that the correct sequence is maintained
function feature_cols_for(bands, time)
    out = String[]
    for b in bands
        for t in 1:time
            push!(out, "$(b)_T$(t)")
        end
    end
    return out
end



function load_and_clean_parquet(path, feature_; target, id_col)
    data = read_parquet(path)
    data_arr = data[:, Symbol.(feature_)]
    meta_c = [Symbol(id_col), Symbol(target)]
    meta = data[:, meta_c]
    out = hcat(meta, data_arr)
    return out
end



function filter_unknown_classes(eval_data, class, target_c)
    m = Bool[]
    for i in 1:nrow(eval_data)
        label = String(eval_data[i, Symbol(target_c)])
        keep = label in class
        push!(m, keep)
    end

    dropped = count(!, m)
    return eval_data[m, :], dropped
end


#tempcnn


function tempcnn_arc(D, class_c, T; hidden = hidden_dims, k = kernel_size, p = dropout)
    samepad=(k ÷ 2,)
    flatten_data = tensor -> reshape(tensor, :, size(tensor,3))  
    Chain(
        x_ntd -> permutedims(x_ntd, (2,3,1)), 
        Conv((k,), D => hidden; pad = samepad),
        BatchNorm(hidden; eps = 1f-5, momentum = 0.1f0), tensor->relu.(tensor), Dropout(p),
        Conv((k,), hidden => hidden; pad=samepad),
        BatchNorm(hidden; eps = 1f-5, momentum = 0.1f0), tensor->relu.(tensor), Dropout(p),
        Conv((k,), hidden => hidden; pad=samepad),
        BatchNorm(hidden; eps=1f-5, momentum = 0.1f0), tensor->relu.(tensor), Dropout(p),
        flatten_data,
        Dense(hidden*T, 4*hidden),
        BatchNorm(4*hidden; eps=1f-5, momentum = 0.1f0), tensor->relu.(tensor), Dropout(p),
        Dense(4*hidden, class_c)

    )
end

function compute_band_scaler_from_wide(data, band_names, time_c)

    mean_ = Float32[]
    std_ = Float32[]

    for b in band_names

        col_ = Symbol[]
        for t in 1:time_c
            name = Symbol("$(b)_T$(t)")
            push!(col_, name)
        end

        i = Float64[]
        for c in col_
            append!(i,data[!, c])
        end
        m = mean(i) 
        s = std(i)

        if !isfinite(s) || s == 0.0
        s = 1.0
        end
        push!(mean_,m)
        push!(std_,s)
    end
    return mean_, std_
end


function build_ten(data, band_names, T)
    N = nrow(data)
    D = length(band_names)

    tens = Array{Float32}(undef, N, T, D)
    for (i, b) in pairs(band_names)
        for t in 1:T
            c_name = Symbol("$(b)_T$(t)")
            c_val = data[!, c_name]
            tens[:, t, i] = Float32.(c_val)
        end

    end
    return tens
end


#in-place 
function znormalize_bands!(data, mean_, std_)
    N,T,D = size(data)
    @inbounds for i in 1:D
        inv = 1f0 / std_[i]
        m_ = mean_[i]
        for n in 1:N
            for t in 1:T
                data[n, t, i] = (data[n, t, i] - m_) * inv
            end
        end

    end
    return data
end





function metrics_from_logits(logits, label_v, classes)
    K, N = size(logits)

    pred  = Flux.onecold(logits, 0:(K-1))     
    pred_lab  = classes[pred .+ 1]          
    true_lab  = classes[label_v .+ 1]           

    cm_c = CM.confmat(pred_lab, true_lab; levels=classes)
    cm_mat = CM.matrix(cm_c)

    acc  = accuracy(pred_lab, true_lab)
    bal = balanced_accuracy(pred_lab, true_lab)
    mf1 = multiclass_f1score(pred_lab, true_lab)

    return acc, bal, mf1, cm_mat
end



function loss_fn(model, features, labels, class_ids)
    labels_onehot = Flux.onehotbatch(labels, class_ids)
    logits = model(features)
    loss_value = Flux.logitcrossentropy(logits, labels_onehot)
    return loss_value
end



function train_tempcnn!(model, train_features_ntd, train_labels, val_features_ntd, val_labels, clas_sort;
    epochs = epochs, batch_size = batch_size, lr = lr,
    weight_decay = weight_decay, seed = seed, patience = patience, lr_gamma = lr_gamma)

    best_val   = Inf
    best_model = nothing
    no_impact = 0

    Random.seed!(seed)
    class_id = 0:maximum(train_labels)

    opti_adam = Optimisers.AdamW(lr, (0.9, 0.999), weight_decay, 1e-8; couple=false)
    opti_set  = Optimisers.setup(opti_adam, model)

    learining_rate_ = lr   

    for epoch in 1:epochs
        Flux.trainmode!(model)
        parcel_n = size(train_features_ntd, 1)
        order = randperm(parcel_n)
        train_loss_sum = 0.0
        num_batches = 0

        for batch_start in 1:batch_size:parcel_n
            batch_end = min(batch_start + batch_size - 1, parcel_n)
            batch_indices  = @view order[batch_start:batch_end]
            batch_features = @view train_features_ntd[batch_indices, :, :]
            batch_labels   = train_labels[batch_indices]
            loss_of_model = m -> loss_fn(m, batch_features, batch_labels, class_id)
            gradient, = Flux.gradient(loss_of_model, model)
            opti_set, model = Optimisers.update!(opti_set, model, gradient)
            train_loss_sum += float(loss_fn(model, batch_features, batch_labels, class_id))
            num_batches += 1
        end
        max_num = max(num_batches, 1)
        train_loss = train_loss_sum / max_num
        Flux.testmode!(model)
        logits_va = model(val_features_ntd)
        val_loss  = float(loss_fn(model, val_features_ntd, val_labels, class_id))
        acc, bal, mf1, _ = metrics_from_logits(logits_va, val_labels, clas_sort)

        if val_loss < best_val - 1e-9
            best_val = val_loss
            best_model = deepcopy(model)
            no_impact = 0
        else
            no_impact += 1
        end


        if lr_gamma != 1.0
            learining_rate_ = learining_rate_ * lr_gamma
            try
                Optimisers.adjust!(opti_set; eta = learining_rate_)
            catch
                try
                    Optimisers.adjust!(opti_set, (; eta = learining_rate_))
                catch
                end
            end
        end

        println(
            "Epoch ", lpad(epoch, 3), " ", epochs,"train = ", round(train_loss; digits=4), "val = ",   round(val_loss; digits=4), "acc (val)=", round(acc; digits=4),
            " BalAcc = ", round(bal; digits=4)," mf1 = ", round(mf1; digits=4), "learnin rate = ", learining_rate_, "no_impact = ", no_impact)

        if no_impact >= patience
            println("Early stopping (best val=", round(best_val; digits=4), ")")
            break
        end
    end

    if best_model !== nothing
        model = best_model
    end
    Flux.testmode!(model)
    return model
end



model_for_export(model_) = Chain(model_.layers[2:end]...)

function export_tempcnn_to_onnx_julia(model_, T, D; onnx_out)
    Flux.testmode!(model_)
    base = model_for_export(model_)

    layers = collect(base.layers)
    layers[13] = x -> reshape(x, (hidden_dims*T, 1))
    export_model = Chain(layers...)
    ONNXNaiveNASflux.save(onnx_out, export_model, (T, D, 1); modelname = "TempCNN_no_perm")
    return onnx_out
end


function build_loader(data, band_names, T, mean_band, std_band, class_to_int, target_)

    ten = build_ten(data, band_names, T)
    znormalize_bands!(ten, mean_band, std_band)
    target_sym = Symbol(target_)
    labels = string.(data[!, target_sym])  
    class_id = getindex.(Ref(class_to_int), labels)
    return ten, class_id
end

function main()

    Random.seed!(seed)

    trainings_data_ = read_parquet(trainings_data)
    all_column_names = names(trainings_data_)
    all_column_names_str = string.(all_column_names)
    feature_ = identify_predictors(all_column_names_str)

    band_names, T = detect_bandset_and_time(feature_)

    train_features_ = trainings_data_[:, Symbol.(feature_)]

    train_data_r = hcat(trainings_data_[:, [Symbol(id_col), Symbol(target_)]], train_features_)

    val_path = validatio_data
    val_feature_cols = feature_cols_for(band_names, T)
    validation_data = load_and_clean_parquet(val_path, val_feature_cols; target = target_, id_col = id_col)

    println("train: n = $(nrow(train_data_r)) val: n = $(nrow(validation_data))")


    target_sym = Symbol(target_)
    col = train_data_r[!, target_sym]
    col_str = string.(col)   
    classes_unique = unique(col_str)
    classes_sorted = sort(classes_unique)
    class_to_int = Dict{String,Int}()

    for (i, class) in enumerate(classes_sorted)
        class_to_int[class] = i - 1  
    end



    mean_band, std_band = compute_band_scaler_from_wide(train_data_r, band_names, T)

    features_tr, labels_tr = build_loader(train_data_r, band_names, T, mean_band, std_band, class_to_int, target_)

    features_va, labels_va = build_loader(validation_data, band_names, T, mean_band, std_band, class_to_int, target_)

    K = length(classes_sorted)
    model = tempcnn_arc(length(band_names), K, T)
    model = train_tempcnn!(model, features_tr, labels_tr, features_va, labels_va, classes_sorted)

    Flux.testmode!(model)
    logits_tr = model(features_tr)
    oa_tr, bal_tr, mf1_tr, _ = metrics_from_logits(logits_tr, labels_tr, classes_sorted)
    logits_va = model(features_va)
    oa_va, bal_va, mf1_va, _ = metrics_from_logits(logits_va, labels_va, classes_sorted)


    write_parquet(DataFrame(band = band_names, mean = mean_band, scale = std_band), out_scaler_parquet)
    write_parquet(DataFrame(label = classes_sorted), out_class_parquet)

    JSON3.write(out_cfg_json, Dict(
        "algorithm" => "TempCNN",
        "time_steps" => T,
        "band_order" => band_names,
        "class_levels_file" => basename(out_class_parquet),
        "scaler_params_file" => basename(out_scaler_parquet)
    ); indent = 2)

    mkpath(dirname(out_params_bson))

    BSON.@save out_params_bson model band_names T classes_sorted mean_band std_band


    metrics_json = Dict("val_labels" => classes_sorted)
    JSON3.write(out_metrics_json, metrics_json; indent = 2)


    export_tempcnn_to_onnx_julia(model, T, length(band_names); onnx_out = out_onnx)


    println("train: acc = $(round(oa_tr, digits=6)) balacc = $(round(bal_tr, digits=6)) mf1 = $(round(mf1_tr, digits=6))")
    println("val: acc = $(round(oa_va, digits=6)) balacc = $(round(bal_va, digits=6)) mf1 = $(round(mf1_va, digits=6))")

end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
