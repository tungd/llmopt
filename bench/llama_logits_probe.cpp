#include <llama.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct tensor_probe {
    std::string directory;
};

bool selected_tensor(std::string_view name) {
    static constexpr std::string_view layer_zero[] = {
        "attn_norm-0", "Qcur-0", "Qcur_normed-0", "Qcur_pos-0",
        "Kcur-0", "Vcur-0", "Kcur_normed-0", "Vcur_normed-0",
        "Kcur_pos-0", "kqv_out-0", "attn_pre_o-0", "attn_post_norm-0",
        "attn_out-0", "ffn_norm-0",
        "ffn_out-0", "ffn_post_norm-0", "out_scaled-0",
    };
    return name == "inp_scaled" || name == "result_norm" ||
           name.substr(0, 6) == "l_out-" ||
           std::find(std::begin(layer_zero), std::end(layer_zero), name) !=
               std::end(layer_zero);
}

bool capture_tensor(ggml_tensor * tensor, bool ask, void * user_data) {
    const std::string_view name = ggml_get_name(tensor);
    const bool selected = selected_tensor(name);
    if (ask) return selected;
    if (!selected) return true;

    const auto * probe = static_cast<const tensor_probe *>(user_data);
    const std::string path = probe->directory + "/" + std::string(name) + ".bin";
    const size_t bytes = ggml_nbytes(tensor);
    std::vector<uint8_t> contents(bytes);
    ggml_backend_tensor_get(tensor, contents.data(), 0, bytes);
    FILE * file = std::fopen(path.c_str(), "wb");
    const bool wrote =
        file != nullptr && std::fwrite(contents.data(), 1, bytes, file) == bytes;
    const bool closed = file == nullptr ? false : std::fclose(file) == 0;
    if (!wrote || !closed) {
        std::fprintf(stderr, "failed to write callback tensor: %s\n", path.c_str());
        return false;
    }
    std::printf("callback-tensor: %s type=%s elements=%lld bytes=%zu shape=",
        name.data(), ggml_type_name(tensor->type),
        static_cast<long long>(ggml_nelements(tensor)), bytes);
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (dimension != 0) std::putchar('x');
        std::printf("%lld", static_cast<long long>(tensor->ne[dimension]));
    }
    std::printf(" output=%s\n", path.c_str());
    return true;
}

std::vector<int32_t> parse_ids(const char * text) {
    std::vector<int32_t> result;
    std::string_view input(text);
    while (!input.empty()) {
        const auto comma = input.find(',');
        const auto item = input.substr(0, comma);
        int32_t value = -1;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), value);
        if (item.empty() || parsed.ec != std::errc{} || parsed.ptr != item.data() + item.size() || value < 0) {
            std::fprintf(stderr, "invalid token ID list: %s\n", text);
            std::exit(64);
        }
        result.push_back(value);
        if (comma == std::string_view::npos) break;
        input.remove_prefix(comma + 1);
    }
    return result;
}

void print_top(const char * label, const std::vector<float> & logits, int count) {
    std::vector<std::pair<int32_t, float>> ranked;
    ranked.reserve(logits.size());
    for (int32_t token = 0; token < static_cast<int32_t>(logits.size()); ++token) {
        ranked.emplace_back(token, logits[token]);
    }
    std::partial_sort(ranked.begin(), ranked.begin() + std::min<int>(count, ranked.size()), ranked.end(),
        [](const auto & left, const auto & right) {
            if (left.second != right.second) return left.second > right.second;
            return left.first < right.first;
        });
    std::printf("%s: ", label);
    for (int index = 0; index < count && index < static_cast<int>(ranked.size()); ++index) {
        if (index != 0) std::putchar(',');
        std::printf("%d:%.9g", ranked[index].first, ranked[index].second);
    }
    std::putchar('\n');
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6 && argc != 7) {
        std::fprintf(stderr,
            "usage: llama-logits-probe <model.gguf> <prompt-token-ids> <suppress-token-ids> <top-k> [embedding-output.f32 [tensor-output-directory]]\n");
        return 64;
    }
    const auto tokens = parse_ids(argv[2]);
    const auto suppressed = parse_ids(argv[3]);
    const int top_k = std::atoi(argv[4]);
    if (tokens.empty() || top_k <= 0) return 64;

    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model\n");
        llama_backend_free();
        return 2;
    }
    auto context_params = llama_context_default_params();
    context_params.n_ctx = std::max<uint32_t>(512, tokens.size() + 1);
    context_params.n_batch = tokens.size();
    context_params.n_ubatch = tokens.size();
    tensor_probe probe{argc == 7 ? argv[6] : ""};
    context_params.embeddings = argc >= 6;
    context_params.cb_eval = argc == 7 ? capture_tensor : nullptr;
    context_params.cb_eval_user_data = argc == 7 ? &probe : nullptr;
    context_params.no_perf = false;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    auto mutable_tokens = tokens;
    auto batch = llama_batch_get_one(mutable_tokens.data(), mutable_tokens.size());
    const int decode_status = llama_decode(context, batch);
    if (decode_status != 0) {
        std::fprintf(stderr, "llama_decode failed: %d\n", decode_status);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    const auto * vocab = llama_model_get_vocab(model);
    const int32_t vocabulary = llama_vocab_n_tokens(vocab);
    const float * output = llama_get_logits_ith(context, -1);
    if (output == nullptr || vocabulary <= 0) {
        std::fprintf(stderr, "last-row logits are unavailable\n");
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    std::vector<float> raw(output, output + vocabulary);
    auto policy = raw;
    for (const int32_t token : suppressed) {
        if (token < 0 || token >= vocabulary) {
            std::fprintf(stderr, "suppressed token %d is outside vocabulary\n", token);
            return 64;
        }
        policy[token] = -std::numeric_limits<float>::infinity();
    }
    std::printf("prompt-token-count: %zu\n", tokens.size());
    std::printf("suppressed-token-ids: %s\n", argv[3]);
    print_top("raw-top-k", raw, top_k);
    print_top("policy-top-k", policy, top_k);
    if (argc >= 6) {
        const int32_t dimensions = llama_model_n_embd(model);
        const float * embedding = llama_get_embeddings_ith(context, -1);
        if (embedding == nullptr || dimensions <= 0) {
            std::fprintf(stderr, "last-token embedding is unavailable\n");
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        FILE * file = std::fopen(argv[5], "wb");
        const bool wrote =
            file != nullptr &&
            std::fwrite(embedding, sizeof(float), dimensions, file) ==
                static_cast<size_t>(dimensions);
        const bool closed = file == nullptr ? false : std::fclose(file) == 0;
        if (!wrote || !closed) {
            std::fprintf(stderr, "failed to write embedding output: %s\n", argv[5]);
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        double sum = 0.0;
        double squared = 0.0;
        for (int32_t index = 0; index < dimensions; ++index) {
            sum += embedding[index];
            squared += static_cast<double>(embedding[index]) * embedding[index];
        }
        std::printf("embedding-elements: %d\n", dimensions);
        std::printf("embedding-mean: %.9g\n", sum / dimensions);
        std::printf("embedding-rms: %.9g\n", std::sqrt(squared / dimensions));
        std::printf("embedding-output: %s\n", argv[5]);
    }

    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
