#ifndef __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__
#define __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__

#include <algorithm>
#include <cinttypes>

#include "model/common/ggml_block.hpp"
#include "model/common/rope.hpp"
#include "core/pe_cache.h"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

namespace MOSSImage {
    struct MOSSImageConfig {
        int64_t dim      = 3072;
        int64_t ffn_dim  = 8192;
        int64_t text_dim = 4096;
        int64_t channels = 128;
        int64_t head_dim = 128;
        int64_t depth    = 42;
        int64_t freq_dim = 256;
        // proj.q / proj.k / proj.v apart (the release weights, community GGUFs, the 8-step LoRA's targets),
        // or fused as proj.qkv (Comfy-Org's checkpoints).
        bool split_qkv = false;

        static MOSSImageConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            MOSSImageConfig config;
            int64_t depth = 0;
            for (const auto& [name, tensor] : weights) {
                if (!starts_with(name, prefix + ".")) {
                    continue;
                }
                auto key = name.substr(prefix.size() + 1);
                if (key == "video_embedder.weight") {
                    config.channels = tensor.ne[0];
                    config.dim      = tensor.ne[1];
                } else if (key == "text_embedder.proj.weight") {
                    config.text_dim = tensor.ne[0];
                } else if (key == "layers.0.proj.ffn_down.weight") {
                    config.ffn_dim = tensor.ne[0];
                } else if (key == "layers.0.q_norm.weight") {
                    config.head_dim = tensor.ne[0];
                } else if (key == "time_embedding.0.weight") {
                    config.freq_dim = tensor.ne[0];
                } else if (key == "layers.0.proj.q.weight") {
                    config.split_qkv = true;
                }
                if (starts_with(key, "layers.")) {
                    depth = std::max<int64_t>(depth, std::stoi(key.substr(7)) + 1);
                }
            }
            if (depth) {
                config.depth = depth;
                LOG_VERBOSE("moss_image: dim = %" PRId64 ", ffn_dim = %" PRId64 ", depth = %" PRId64 ", head_dim = %" PRId64,
                            config.dim, config.ffn_dim, config.depth, config.head_dim);
            }
            return config;
        }
    };

    inline ggml_tensor* situ(ggml_context* c, ggml_tensor* hidden, ggml_tensor* gate) {
        auto a = ggml_scale(c, ggml_tanh(c, ggml_scale(c, gate, 0.25f)), 4.f);
        a      = ggml_mul(c, a, ggml_sigmoid(c, gate));
        auto b = ggml_scale(c, ggml_tanh(c, ggml_scale(c, hidden, 1.f / 25.f)), 25.f);
        return ggml_mul(c, a, b);
    }

    // nightmare (dit/035): Q/K norm and rope as ONE fused op (the HTP's qknorm_rope), as dit/029 did for Ideogram 4.
    // The composed Rope::apply_rope (split-half: rope_interleaved = false) was ~13 of 18.7 s a step at 512x512 on the
    // S25 (a REPEAT per q and k in each of 42 blocks at ~54 ms, a scalar copy loop, then MUL, ADD and CONT). The fused
    // kernel rotates interleaved pairs (2j, 2j+1): q, k and the norm weight get the same channel permutation into that
    // order (new[2j] = x[j], new[2j+1] = x[j + d/2]), which leaves q.k, so the attention, unchanged.
    // NM_MOSS_COMPOSED_ROPE=1 keeps the old path (the A/B control).
    class MOSSQKNorm : public RMSNorm {
    public:
        using RMSNorm::RMSNorm;

        ggml_tensor* forward_rope(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe) {
            static const bool composed = getenv("NM_MOSS_COMPOSED_ROPE") != nullptr;
            if (composed || ctx->backend == nullptr || x->ne[3] != 1 || x->type != GGML_TYPE_F32) {
                return nullptr;
            }
            ggml_tensor* w = params["weight"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
            }
            const int64_t d    = x->ne[0];
            const int64_t half = d / 2;
            // x: [d, H, L, 1] -> [d/2, 2, H, L] -> [2, d/2, H, L]: new[2j] = x[j], new[2j+1] = x[j + d/2]
            ggml_tensor* xi = ggml_view_4d(ctx->ggml_ctx, x, half, 2, x->ne[1], x->ne[2],
                                           half * ggml_element_size(x), x->nb[1], x->nb[2], 0);
            xi              = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, xi, 1, 0, 2, 3));
            xi              = ggml_reshape_4d(ctx->ggml_ctx, xi, d, x->ne[1], x->ne[2], 1);
            ggml_tensor* wi = ggml_reshape_2d(ctx->ggml_ctx, w, half, 2);
            wi              = ggml_cont(ctx->ggml_ctx, ggml_transpose(ctx->ggml_ctx, wi));
            wi              = ggml_reshape_1d(ctx->ggml_ctx, wi, d);
            ggml_tensor* out = ggml_qknorm_rope(ctx->ggml_ctx, xi, wi, pe, eps);  // [d, L, H]
            if (!ggml_backend_supports_op(ctx->backend, out)) {
                return nullptr;
            }
            return out;
        }
    };

    // q, k, v: [d, H, L, 1] straight off the projection; q/k normed and roped, then attended (as Rope::attention does).
    inline ggml_tensor* attend(GGMLRunnerContext* ctx, const std::shared_ptr<MOSSQKNorm>& norm_q, const std::shared_ptr<MOSSQKNorm>& norm_k,
                               ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* pe) {
        ggml_tensor* q_rope = norm_q->forward_rope(ctx, q, pe);
        ggml_tensor* k_rope = q_rope != nullptr ? norm_k->forward_rope(ctx, k, pe) : nullptr;
        if (q_rope != nullptr && k_rope != nullptr) {
            return ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q_rope, k_rope, v, q->ne[1], nullptr, true,
                                          ctx->flash_attn_enabled, 1.f);
        }
        return Rope::attention(ctx, norm_q->forward(ctx, q), norm_k->forward(ctx, k), v, pe, nullptr, 1.f, false);
    }

    struct TextBlock : public GGMLBlock {
        MOSSImageConfig config;

        TextBlock(const MOSSImageConfig& config)
            : config(config) {
            blocks["norm1"]    = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["norm2"]    = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["q_norm"]   = std::make_shared<MOSSQKNorm>(config.head_dim, 1e-6f);
            blocks["k_norm"]   = std::make_shared<MOSSQKNorm>(config.head_dim, 1e-6f);
            blocks["qkv"]      = std::make_shared<Linear>(config.dim, config.dim * 3, true);
            blocks["proj"]     = std::make_shared<Linear>(config.dim, config.dim, true);
            blocks["ffn_up"]   = std::make_shared<Linear>(config.dim, config.ffn_dim, true);
            blocks["ffn_gate"] = std::make_shared<Linear>(config.dim, config.ffn_dim, true);
            blocks["ffn_down"] = std::make_shared<Linear>(config.ffn_dim, config.dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe) {
            auto c      = ctx->ggml_ctx;
            auto linear = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            auto norm   = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, h); };
            auto y      = norm("norm1", x);
            auto qkv    = linear("qkv", y);
            auto split  = [&](int i) {
                return ggml_reshape_4d(c, ggml_ext_slice(c, qkv, 0, i * config.dim, (i + 1) * config.dim), config.head_dim, config.dim / config.head_dim, x->ne[1], x->ne[2]);
            };
            y      = attend(ctx, std::dynamic_pointer_cast<MOSSQKNorm>(blocks["q_norm"]),
                            std::dynamic_pointer_cast<MOSSQKNorm>(blocks["k_norm"]), split(0), split(1), split(2), pe);
            y      = linear("proj", y);
            x      = ggml_add(c, x, y);
            y      = norm("norm2", x);
            y      = situ(c, linear("ffn_up", y), linear("ffn_gate", y));
            y      = linear("ffn_down", y);
            return ggml_add(c, x, y);
        }
    };

    struct Block : public GGMLBlock {
        MOSSImageConfig config;

        Block(const MOSSImageConfig& config)
            : config(config) {
            blocks["norm1"]            = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["norm2"]            = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["q_norm"]           = std::make_shared<MOSSQKNorm>(config.head_dim, 1e-6f);
            blocks["k_norm"]           = std::make_shared<MOSSQKNorm>(config.head_dim, 1e-6f);
            if (config.split_qkv) {
                blocks["proj.q"] = std::make_shared<Linear>(config.dim, config.dim, false);
                blocks["proj.k"] = std::make_shared<Linear>(config.dim, config.dim, false);
                blocks["proj.v"] = std::make_shared<Linear>(config.dim, config.dim, false);
            } else {
                blocks["proj.qkv"] = std::make_shared<Linear>(config.dim, config.dim * 3, false);
            }
            blocks["proj.o"]           = std::make_shared<Linear>(config.dim, config.dim, false);
            blocks["proj.ffn_up_gate"] = std::make_shared<Linear>(config.dim, config.ffn_dim * 2, false);
            blocks["proj.ffn_down"]    = std::make_shared<Linear>(config.ffn_dim, config.dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe, ggml_tensor* mod) {
            auto c      = ctx->ggml_ctx;
            auto linear = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            auto norm   = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, h); };
            auto part   = [&](int i) { return ggml_ext_slice(c, mod, 0, i * config.dim, (i + 1) * config.dim); };
            auto y      = norm("norm1", x);
            y           = ggml_add(c, ggml_add(c, y, ggml_mul(c, y, part(1))), part(0));
            auto heads  = [&](ggml_tensor* h) { return ggml_reshape_4d(c, h, config.head_dim, config.dim / config.head_dim, x->ne[1], x->ne[2]); };
            ggml_tensor* qkv = config.split_qkv ? nullptr : linear("proj.qkv", y);
            auto split  = [&](int i) {
                if (config.split_qkv) {
                    static const char* names[] = {"proj.q", "proj.k", "proj.v"};
                    return heads(linear(names[i], y));
                }
                return heads(ggml_ext_slice(c, qkv, 0, i * config.dim, (i + 1) * config.dim));
            };
            y            = attend(ctx, std::dynamic_pointer_cast<MOSSQKNorm>(blocks["q_norm"]),
                                  std::dynamic_pointer_cast<MOSSQKNorm>(blocks["k_norm"]), split(0), split(1), split(2), pe);
            y            = linear("proj.o", y);
            y            = ggml_mul(c, y, part(2));
            x            = ggml_add(c, x, y);
            y            = norm("norm2", x);
            y            = ggml_add(c, ggml_add(c, y, ggml_mul(c, y, part(4))), part(3));
            auto up_gate = linear("proj.ffn_up_gate", y);
            y            = situ(c, ggml_ext_slice(c, up_gate, 0, 0, config.ffn_dim), ggml_ext_slice(c, up_gate, 0, config.ffn_dim, 2 * config.ffn_dim));
            y            = linear("proj.ffn_down", y);
            y            = ggml_mul(c, y, part(5));
            return ggml_add(c, x, y);
        }
    };

    struct Model : public GGMLBlock {
        MOSSImageConfig config;

        Model(const MOSSImageConfig& config)
            : config(config) {
            blocks["video_embedder"]     = std::make_shared<Linear>(config.channels, config.dim, true);
            blocks["text_embedder.proj"] = std::make_shared<Linear>(config.text_dim, config.dim, true);
            for (int i = 0; i < 2; ++i) {
                blocks["text_embedder.blocks." + std::to_string(i)] = std::make_shared<TextBlock>(config);
            }
            blocks["time_embedding.0"]        = std::make_shared<Linear>(config.freq_dim, config.dim, true);
            blocks["time_embedding.2"]        = std::make_shared<Linear>(config.dim, config.dim, true);
            blocks["shared_block_modulation"] = std::make_shared<Linear>(config.dim, 6 * config.dim, true);
            blocks["video_head_modulation"]   = std::make_shared<Linear>(config.dim, 2 * config.dim, true);
            for (int64_t i = 0; i < config.depth; ++i) {
                blocks["layers." + std::to_string(i)] = std::make_shared<Block>(config);
            }
            blocks["final_norm_video"]   = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["final_linear_video"] = std::make_shared<Linear>(config.dim, config.channels, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* image, ggml_tensor* time, ggml_tensor* text, ggml_tensor* pe, const std::vector<int>& skip) {
            auto c            = ctx->ggml_ctx;
            auto linear       = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            int64_t image_len = image->ne[0] * image->ne[1];
            auto x            = linear("video_embedder", DiT::patchify(c, image, 1, 1));
            text              = linear("text_embedder.proj", text);
            auto text_pe      = ggml_ext_slice(c, pe, 3, image_len, image_len + text->ne[1]);
            for (int i = 0; i < 2; ++i) {
                text = std::dynamic_pointer_cast<TextBlock>(blocks["text_embedder.blocks." + std::to_string(i)])->forward(ctx, text, text_pe);
            }
            time      = linear("time_embedding.2", ggml_silu(c, linear("time_embedding.0", time)));
            auto mod  = linear("shared_block_modulation", time);
            auto head = linear("video_head_modulation", time);
            x         = ggml_concat(c, x, text, 1);
            for (int64_t i = 0; i < config.depth; ++i) {
                if (std::find(skip.begin(), skip.end(), i) == skip.end()) {
                    x = std::dynamic_pointer_cast<Block>(blocks["layers." + std::to_string(i)])->forward(ctx, x, pe, mod);
                    sd::ggml_graph_cut::mark_graph_cut(x, "moss_image.layers." + std::to_string(i), "x");
                }
            }
            x          = ggml_ext_slice(c, x, 1, 0, image_len);
            x          = std::dynamic_pointer_cast<RMSNorm>(blocks["final_norm_video"])->forward(ctx, x);
            auto shift = ggml_ext_slice(c, head, 0, 0, config.dim);
            auto scale = ggml_ext_slice(c, head, 0, config.dim, 2 * config.dim);
            x          = ggml_add(c, ggml_add(c, x, ggml_mul(c, x, scale)), shift);
            x          = linear("final_linear_video", x);
            return DiT::unpatchify(c, x, image->ne[1], image->ne[0], 1, 1);
        }
    };

    struct MOSSImageRunner : public DiffusionModelRunner {
        sd::PECache pe_cache_;
        MOSSImageConfig config;
        Model model;
        std::vector<float> time_data;

        MOSSImageRunner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager), config(MOSSImageConfig::detect_from_weights(weights, prefix)), model(config) {
            model.init(params_ctx, weights, prefix);
        }

        std::string get_desc() override { return "MOSS-Image"; }
        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs) override {
            if (!inputs.x || !inputs.context || !inputs.timesteps || inputs.context->empty()) {
                LOG_ERROR("MOSS-Image requires latents, timesteps and text features");
                return {};
            }
            std::vector<int> skip;
            if (auto extra = std::get_if<SkipLayerDiffusionExtra>(&inputs.extra)) {
                if (extra->skip_layers) {
                    skip = *extra->skip_layers;
                }
            }
            auto graph = [&]() {
                auto gf   = new_graph_custom(65536);
                auto x    = make_input(*inputs.x);
                auto text = make_input(*inputs.context);
                GGML_ASSERT(x->ne[3] == 1 && x->ne[2] == config.channels && text->ne[0] == config.text_dim);
                int64_t h = x->ne[1], w = x->ne[0], nt = text->ne[1];
                // Image tokens first (axis 1 = the text length, axes 2/3 centred rows/columns), then the text
                // tokens counting up on axis 1. Our fork's embed_nd gives the [2, 2, head_dim/2, n] table directly.
                sd::PECacheKey pe_key("moss_image.pe");
                pe_key.add_shape(x).add_shape(text).add(config.head_dim).add(config.channels);
                pe_key.add(circular_x_enabled).add(circular_y_enabled);
                auto pe = make_input(pe_cache_.get_or_create(pe_key, [&]() {
                    std::vector<std::vector<float>> ids;
                    for (int64_t token = 0; token < h * w + nt; ++token) {
                        float pos[4] = {0.f, float(token - h * w), 0.f, 0.f};
                        if (token < h * w) {
                            pos[1] = float(nt);
                            pos[2] = float(token / w) - float(h - 1) / 2.f;
                            pos[3] = float(token % w) - float(w - 1) / 2.f;
                        }
                        ids.push_back({pos[0], pos[1], pos[2], pos[3]});
                    }
                    auto pe_data = Rope::embed_nd(ids, 1, 10000.f, std::vector<int>(4, static_cast<int>(config.head_dim / 4)));
                    return sd::Tensor<float>({2, 2, config.head_dim / 2, h * w + nt}, std::move(pe_data));
                }));
                time_data.resize(config.freq_dim);
                for (int64_t i = 0; i < config.freq_dim / 2; ++i) {
                    double angle                       = inputs.timesteps->values()[0] * std::pow(10000.0, -double(i) / double(config.freq_dim / 2));
                    time_data[i]                       = float(std::cos(angle));
                    time_data[i + config.freq_dim / 2] = float(std::sin(angle));
                }
                auto time = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, config.freq_dim, 1, 1);
                set_backend_tensor_data(time, time_data.data());
                auto ctx = get_context();
                ggml_build_forward_expand(gf, model.forward(&ctx, x, time, text, pe, skip));
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(graph, n_threads, false), inputs.x->dim());
        }
    };
}  // namespace MOSSImage

#endif  // __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__
