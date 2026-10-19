#include "qwen35_tokenizer.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "core/util.h"
#include "json.hpp"

Qwen35Tokenizer::Qwen35Tokenizer(const std::string& tokenizer_json_path) {
    UNK_TOKEN = "<|endoftext|>";
    EOS_TOKEN = "<|endoftext|>";
    PAD_TOKEN = "<|endoftext|>";

    std::ifstream file(tokenizer_json_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open the Qwen3.5 tokenizer.json: " + tokenizer_json_path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    const nlohmann::json json = nlohmann::json::parse(buffer.str());
    const auto& model         = json.at("model");

    auto byte_unicode_pairs = bytes_to_unicode();
    byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
    for (auto& pair : byte_unicode_pairs) {
        byte_decoder[pair.second] = pair.first;
    }

    for (const auto& [token, id] : model.at("vocab").items()) {
        std::u32string t = utf8_to_utf32(token);
        encoder[t]       = id.get<int>();
        decoder[id.get<int>()] = t;
    }

    int rank = 0;
    for (const auto& merge : model.at("merges")) {
        std::string a, b;
        if (merge.is_array()) {
            a = merge.at(0).get<std::string>();
            b = merge.at(1).get<std::string>();
        } else {
            const std::string s = merge.get<std::string>();
            const size_t space  = s.find(' ');
            a                   = s.substr(0, space);
            b                   = s.substr(space + 1);
        }
        bpe_ranks[{utf8_to_utf32(a), utf8_to_utf32(b)}] = rank++;
    }
    bpe_len = rank;

    // The special tokens ("<|im_start|>", "<think>", ...) are matched whole in the text, before the BPE.
    int max_id = 0;
    for (const auto& added : json.at("added_tokens")) {
        const std::string content = added.at("content").get<std::string>();
        const int id              = added.at("id").get<int>();
        std::u32string t          = utf8_to_utf32(content);
        encoder[t]                = id;
        decoder[id]               = t;
        special_tokens.push_back(content);
        if (content == UNK_TOKEN) {
            UNK_TOKEN_ID = EOS_TOKEN_ID = PAD_TOKEN_ID = id;
        }
        max_id = std::max(max_id, id);
    }
    encoder_len = max_id + 1;
    LOG_VERBOSE("qwen3.5 tokenizer: %zu tokens, %d merges, %zu special, pad %d",
                encoder.size(), bpe_len, special_tokens.size(), PAD_TOKEN_ID);
}
