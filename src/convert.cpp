#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "core/util.h"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "model_io/streaming_writer.h"
#include "model_loader.h"
#include "name_conversion.h"
#include "runtime/imatrix.h"

struct TensorExportInfo {
    TensorStorage storage;
    ggml_type type;
};

struct TensorExportJob {
    TensorExportInfo info;
    std::vector<uint8_t> data;
    std::string error;
    bool success = false;
};

static ggml_type get_export_tensor_type(ModelLoader& model_loader,
                                        const TensorStorage& tensor_storage,
                                        ggml_type type,
                                        const TensorTypeRules& tensor_type_rules) {
    const std::string& name = tensor_storage.name;
    ggml_type tensor_type   = tensor_storage.type;
    ggml_type dst_type      = type;

    for (const auto& tensor_type_rule : tensor_type_rules) {
        std::regex pattern(tensor_type_rule.first);
        if (std::regex_search(name, pattern)) {
            dst_type = tensor_type_rule.second;
            break;
        }
    }

    if (model_loader.tensor_should_be_converted(tensor_storage, dst_type)) {
        tensor_type = dst_type;
    }

    return tensor_type;
}

static bool collect_tensors_for_export(ModelLoader& model_loader,
                                       ggml_type type,
                                       const TensorTypeRules& tensor_type_rules,
                                       std::vector<TensorExportInfo>& tensors) {
    tensors.clear();
    tensors.reserve(model_loader.get_tensor_storage_map().size());
    for (const auto& kv : model_loader.get_tensor_storage_map()) {
        const TensorStorage& tensor_storage = kv.second;
        if (ends_with(tensor_storage.name, ".weight_scale")) {
            // ComfyUI fp8/int8 scale sibling. The runner multiplies layer outputs by any
            // weight_scale it finds, so a stale scale next to already-correct (e.g. quantized
            // or unscaled) weights crushes activations. Keep it only when the sibling weight
            // is really int8_tensorwise, its one legitimate consumer.
            const std::string wname =
                tensor_storage.name.substr(0, tensor_storage.name.size() - std::string("_scale").size());
            auto wit = model_loader.get_tensor_storage_map().find(wname);
            if (wit == model_loader.get_tensor_storage_map().end() || !wit->second.is_int8_tensorwise) {
                continue;
            }
        }
        TensorExportInfo info;
        info.storage = tensor_storage;
        info.type    = get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules);
        tensors.push_back(std::move(info));
    }
    LOG_INFO("collected %zu tensors for export", tensors.size());
    return true;
}

static size_t export_tensor_nbytes(const TensorExportInfo& info) {
    TensorStorage output_storage = info.storage;
    output_storage.type          = info.type;
    return static_cast<size_t>(output_storage.nbytes());
}

static TensorWritePlan tensor_write_plan_from_export_info(const TensorExportInfo& info) {
    TensorWritePlan plan;
    plan.name   = info.storage.name;
    plan.type   = info.type;
    plan.n_dims = info.storage.n_dims;
    for (int i = 0; i < SD_MAX_DIMS; i++) {
        plan.ne[i] = info.storage.ne[i];
    }
    return plan;
}

static std::vector<TensorWritePlan> tensor_write_plans_from_export_infos(const std::vector<TensorExportInfo>& tensors) {
    std::vector<TensorWritePlan> plans;
    plans.reserve(tensors.size());
    for (const TensorExportInfo& info : tensors) {
        plans.push_back(tensor_write_plan_from_export_info(info));
    }
    return plans;
}

static bool preallocate_output_file(const std::string& output_path, uint64_t file_size, std::string* error) {
    if (file_size == 0) {
        return true;
    }

    std::fstream file(output_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file.is_open()) {
        if (error != nullptr) {
            *error = "failed to open output file '" + output_path + "' for preallocation";
        }
        return false;
    }

    // This portable fallback sets the final file size. A platform-specific
    // posix_fallocate/ftruncate path can replace it later.
    file.seekp(static_cast<std::streamoff>(file_size - 1), std::ios::beg);
    file.put('\0');
    file.flush();
    if (!file) {
        if (error != nullptr) {
            *error = "failed to preallocate output file '" + output_path + "'";
        }
        return false;
    }
    return true;
}

static const TensorStorage* find_fp8_scale_sibling(ModelLoader& model_loader,
                                                      const TensorStorage& tensor_storage) {
    if (!tensor_storage.is_f8_e4m3 && !tensor_storage.is_f8_e5m2) {
        return nullptr;
    }
    auto& map = model_loader.get_tensor_storage_map();
    auto it   = map.find(tensor_storage.name + "_scale");
    if (it == map.end() || it->second.type != GGML_TYPE_F32 || it->second.nelements() != 1) {
        return nullptr;
    }
    return &it->second;
}

// ComfyUI fp8_scaled files store scaled values: true = stored * scale. Quantizing the raw
// stored values bakes in a 100-1000x error (activations detonate into static). Unscale through
// F32 first, then run the normal F32->target path (same kernels as every other convert).
static bool load_unscaled_f8_for_export(ModelLoader& model_loader, TensorExportJob& job,
                                        const TensorStorage& scale_storage) {
    const TensorStorage& tensor_storage = job.info.storage;
    const size_t ne                     = (size_t)tensor_storage.nelements();
    const int64_t ne0                   = tensor_storage.ne[0];
    if (ne == 0 || ne0 <= 0 || ne % (size_t)ne0 != 0) {
        job.error = "invalid dims for fp8 unscale on tensor '" + tensor_storage.name + "'";
        return false;
    }
    std::vector<float> f32(ne);
    if (!model_loader.load_float_tensor(tensor_storage.name, f32) || f32.size() != ne) {
        job.error = "failed to load fp8 tensor '" + tensor_storage.name + "' as F32";
        return false;
    }
    std::vector<float> scale;
    if (!model_loader.load_float_tensor(scale_storage.name, scale) || scale.size() != 1) {
        job.error = "failed to load scale tensor '" + scale_storage.name + "'";
        return false;
    }
    const float s = scale[0];
    if (!(s > 0.f) || !std::isfinite(s)) {
        job.error = "insane scale for tensor '" + tensor_storage.name + "'";
        return false;
    }
    for (size_t i = 0; i < ne; i++) {
        f32[i] *= s;
    }
    TensorStorage output_storage = tensor_storage;
    output_storage.type          = job.info.type;
    const size_t out_nbytes      = (size_t)output_storage.nbytes();
    job.data.resize(out_nbytes);
    std::string processed_name = convert_tensor_name(tensor_storage.name, model_loader.get_sd_version());
    std::vector<float> imatrix = get_imatrix_collector().get_values(processed_name);
    convert_tensor((void*)f32.data(), GGML_TYPE_F32, (void*)job.data.data(), job.info.type,
                   (int)(ne / (size_t)ne0), (int)ne0, std::move(imatrix));
    return true;
}

static bool load_tensor_for_export(ModelLoader& model_loader, TensorExportJob& job) {
    const TensorStorage& tensor_storage = job.info.storage;
    if (job.info.type != tensor_storage.type) {
        if (const TensorStorage* scale = find_fp8_scale_sibling(model_loader, tensor_storage)) {
            return load_unscaled_f8_for_export(model_loader, job, *scale);
        }
    }
    size_t mem_size = 1 * 1024 * 1024;
    mem_size += ggml_tensor_overhead();
    TensorStorage output_storage = job.info.storage;
    output_storage.type          = job.info.type;
    mem_size += static_cast<size_t>(output_storage.nbytes());

    ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});
    if (ggml_ctx == nullptr) {
        job.error = "ggml_init failed for tensor '" + job.info.storage.name + "'";
        return false;
    }

    ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, job.info.type, job.info.storage.n_dims, job.info.storage.ne);
    if (tensor == nullptr) {
        ggml_free(ggml_ctx);
        job.error = "ggml_new_tensor failed for tensor '" + job.info.storage.name + "'";
        return false;
    }
    ggml_set_name(tensor, job.info.storage.name.c_str());

    const size_t tensor_nbytes = ggml_nbytes(tensor);
    if (tensor_nbytes > 0 && !model_loader.load_tensor(job.info.storage, tensor)) {
        ggml_free(ggml_ctx);
        job.error = "failed to load tensor '" + job.info.storage.name + "'";
        return false;
    }

    job.data.resize(tensor_nbytes);
    if (tensor_nbytes > 0) {
        memcpy(job.data.data(), tensor->data, tensor_nbytes);
    }
    ggml_free(ggml_ctx);
    return true;
}

static bool stream_tensor_data(ModelLoader& model_loader,
                               const std::string& output_path,
                               const std::vector<TensorExportInfo>& tensors,
                               const StreamingModelWriter& writer,
                               int n_threads,
                               std::string* error) {
    n_threads = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    n_threads = std::max(1, n_threads);
    LOG_INFO("streaming convert with %d threads", n_threads);

    int64_t start_time       = ggml_time_ms();
    uint64_t bytes_written   = 0;
    size_t tensors_written   = 0;
    size_t next_tensor_index = 0;
    bool failed              = false;
    std::string failure;

    const size_t memory_budget = 1024ull * 1024ull * 1024ull;
    size_t reserved_bytes      = 0;

    std::mutex work_mutex;
    std::mutex progress_mutex;
    std::condition_variable memory_cv;
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    auto reserve_memory = [&](size_t bytes) -> bool {
        std::unique_lock<std::mutex> lock(work_mutex);
        memory_cv.wait(lock, [&]() {
            return failed || reserved_bytes == 0 || reserved_bytes + bytes <= memory_budget;
        });
        if (failed) {
            return false;
        }
        reserved_bytes += bytes;
        return true;
    };

    auto release_memory = [&](size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            reserved_bytes -= std::min(reserved_bytes, bytes);
        }
        memory_cv.notify_all();
    };

    auto fail = [&](const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            if (!failed) {
                failed  = true;
                failure = message;
            }
        }
        memory_cv.notify_all();
    };

    for (int worker = 0; worker < n_threads; worker++) {
        workers.emplace_back([&]() {
            std::fstream output_file(output_path, std::ios::binary | std::ios::in | std::ios::out);
            if (!output_file.is_open()) {
                fail("failed to open output file '" + output_path + "' for tensor writing");
                return;
            }

            while (true) {
                size_t tensor_index = 0;
                {
                    std::lock_guard<std::mutex> lock(work_mutex);
                    if (failed || next_tensor_index >= tensors.size()) {
                        return;
                    }
                    tensor_index = next_tensor_index++;
                }

                const size_t tensor_bytes = export_tensor_nbytes(tensors[tensor_index]);
                if (!reserve_memory(tensor_bytes)) {
                    return;
                }

                TensorExportJob job;
                job.info = tensors[tensor_index];
                try {
                    job.success = load_tensor_for_export(model_loader, job);
                } catch (const std::exception& e) {
                    job.error   = e.what();
                    job.success = false;
                }

                if (!job.success) {
                    release_memory(tensor_bytes);
                    fail(job.error.empty() ? "streaming conversion failed" : job.error);
                    return;
                }

                std::string write_error;
                if (!writer.write_tensor(output_file,
                                         tensor_index,
                                         job.data.empty() ? nullptr : job.data.data(),
                                         job.data.size(),
                                         &write_error)) {
                    release_memory(tensor_bytes);
                    fail(write_error.empty() ? "streaming conversion write failed" : write_error);
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    bytes_written += job.data.size();
                    tensors_written++;
                    float elapsed_seconds = (ggml_time_ms() - start_time) / 1000.0f;
                    pretty_bytes_progress(static_cast<int>(tensors_written),
                                          static_cast<int>(tensors.size()),
                                          bytes_written,
                                          elapsed_seconds);
                }
                release_memory(tensor_bytes);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    printf("\n");
    if (failed) {
        if (error != nullptr) {
            *error = failure;
        }
        return false;
    }
    LOG_INFO("streaming conversion completed, taking %.2fs", (ggml_time_ms() - start_time) / 1000.f);
    return true;
}

static bool write_model_file_streaming(ModelLoader& model_loader,
                                       const std::string& output_path,
                                       const std::vector<TensorExportInfo>& tensors,
                                       StreamingModelWriter& writer,
                                       int n_threads,
                                       std::string* error) {
    std::vector<TensorWritePlan> plans = tensor_write_plans_from_export_infos(tensors);
    if (!writer.write_metadata(output_path, plans, error)) {
        return false;
    }
    if (!preallocate_output_file(output_path, writer.file_size(), error)) {
        return false;
    }
    model_loader.process_model_files(false, false);
    return stream_tensor_data(model_loader, output_path, tensors, writer, n_threads, error);
}

static bool init_convert_path(ModelLoader& model_loader, const char* path, const char* prefix, bool& loaded_any) {
    if (path == nullptr || strlen(path) == 0) {
        return true;
    }
    if (!model_loader.init_from_file(path, prefix)) {
        LOG_ERROR("init model loader from file failed: '%s'", path);
        return false;
    }
    loaded_any = true;
    return true;
}

static bool export_loaded_model(ModelLoader& model_loader,
                                const char* output_path,
                                sd_type_t output_type,
                                const char* tensor_type_rules,
                                int n_threads) {
    ggml_type type             = sd_type_to_ggml_type(output_type);
    bool output_is_safetensors = ends_with(output_path, ".safetensors");
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules ? tensor_type_rules : "");

    std::vector<TensorExportInfo> tensors;
    bool success = collect_tensors_for_export(model_loader, type, type_rules, tensors);
    std::string error;
    if (success) {
        std::unique_ptr<StreamingModelWriter> writer;
        if (output_is_safetensors) {
            writer = std::make_unique<SafetensorsStreamingWriter>();
        } else {
            writer = std::make_unique<GGUFStreamingWriter>();
        }
        success = write_model_file_streaming(model_loader, output_path, tensors, *writer, n_threads, &error);
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    return success;
}

bool convert_with_components(const char* model_path,
                             const char* clip_l_path,
                             const char* clip_g_path,
                             const char* t5xxl_path,
                             const char* diffusion_model_path,
                             const char* vae_path,
                             const char* output_path,
                             sd_type_t output_type,
                             const char* tensor_type_rules,
                             bool convert_name,
                             int n_threads) {
    ModelLoader model_loader;
    bool loaded_any = false;

    if (!init_convert_path(model_loader, model_path, "", loaded_any) ||
        !init_convert_path(model_loader, clip_l_path, "text_encoders.clip_l.transformer.", loaded_any) ||
        !init_convert_path(model_loader, clip_g_path, "text_encoders.clip_g.transformer.", loaded_any) ||
        !init_convert_path(model_loader, t5xxl_path, "text_encoders.t5xxl.transformer.", loaded_any) ||
        !init_convert_path(model_loader, diffusion_model_path, "model.diffusion_model.", loaded_any) ||
        !init_convert_path(model_loader, vae_path, "vae.", loaded_any)) {
        return false;
    }

    if (!loaded_any) {
        LOG_ERROR("no input model path provided for convert");
        return false;
    }

    if (convert_name) {
        model_loader.convert_tensors_name();
    }

    return export_loaded_model(model_loader, output_path, output_type, tensor_type_rules, n_threads);
}

bool convert(const char* input_path,
             const char* vae_path,
             const char* output_path,
             sd_type_t output_type,
             const char* tensor_type_rules,
             bool convert_name) {
    return convert_with_components(input_path,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   vae_path,
                                   output_path,
                                   output_type,
                                   tensor_type_rules,
                                   convert_name,
                                   0);
}
