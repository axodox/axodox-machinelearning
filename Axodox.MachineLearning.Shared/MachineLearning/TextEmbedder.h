#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"
#include "TextTokenizer.h"
#include "TextEncoder.h"
#include "Prompts/PromptAttention.h"

namespace Axodox::MachineLearning
{
  struct AXODOX_MACHINELEARNING_API TextChunk
  {
    std::string Text;
    float Attention;
  };
  
  class AXODOX_MACHINELEARNING_API TextEmbedder
  {
    struct TokenizedPrompt
    {
      Tensor TokenizedText;
      std::vector<float> AttentionMask;
      int32_t AvailableTokenCount = 0;
    };

  public:
    TextEmbedder(OnnxEnvironment& environment, const std::filesystem::path& sourcePath = {});

    int32_t ValidatePrompt(std::string_view text);

    std::vector<std::shared_ptr<Tensor>> SchedulePrompt(std::string_view text, uint32_t stepCount);

    Tensor ProcessPrompt(std::string_view text);

  private:
    TextTokenizer _textTokenizer;
    TextEncoder _textEncoder;

    TokenizedPrompt MergeTokenizedChunks(const Tensor& tokenizedChunks, std::span<const Prompts::PromptAttentionFrame> textChunks);
    void ApplyAttention(Tensor& encodedText, std::span<const float> attentionMask);

    TokenizedPrompt TokenizePrompt(std::string_view text);
  };
}