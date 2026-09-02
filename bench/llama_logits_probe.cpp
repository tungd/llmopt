#include <llama.h>

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
    if (argc != 5) {
        std::fprintf(stderr,
            "usage: llama-logits-probe <model.gguf> <prompt-token-ids> <suppress-token-ids> <top-k>\n");
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

    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
