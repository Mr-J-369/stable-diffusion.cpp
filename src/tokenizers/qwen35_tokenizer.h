#ifndef __SD_TOKENIZERS_QWEN35_TOKENIZER_H__
#define __SD_TOKENIZERS_QWEN35_TOKENIZER_H__

#include <string>

#include "bpe_tokenizer.h"

// Qwen3.5 (MOSS-Image's text encoder): Qwen2's byte-level BPE and split regex, but its own 248k vocabulary,
// which is not ordered as bytes + merges, so it is read whole from the model's tokenizer.json.
class Qwen35Tokenizer : public BPETokenizer {
public:
    explicit Qwen35Tokenizer(const std::string& tokenizer_json_path);
};

#endif  // __SD_TOKENIZERS_QWEN35_TOKENIZER_H__
