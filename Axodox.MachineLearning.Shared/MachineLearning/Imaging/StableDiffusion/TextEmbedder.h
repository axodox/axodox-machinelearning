#pragma once
#include "MachineLearning/Sessions/OnnxSession.h"
#include "MachineLearning/Text/Prompts/PromptAttention.h"
#include "MachineLearning/Tensor.h"
#include "TextTokenizer.h"
#include "TextEncoder.h"

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  struct AXODOX_MACHINELEARNING_API TextChunk
  {
    std::string Text;
    float Attention;
  };

  struct AXODOX_MACHINELEARNING_API ScheduledPrompt
  {
    std::shared_ptr<EncodedText> Tensor;
    std::vector<float> Weights;
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
    TextEmbedder(std::unique_ptr<TextTokenizer>&& tokenizer, std::unique_ptr<TextEncodingProvider>&& encoder);

    int32_t ValidatePrompt(std::string_view text);

    std::vector<ScheduledPrompt> SchedulePrompt(std::string_view text, uint32_t stepCount);

    EncodedText ProcessPrompt(std::string_view text);

  private:
    std::unique_ptr<TextTokenizer> _textTokenizer;
    std::unique_ptr<TextEncodingProvider> _textEncoder;

    TokenizedPrompt MergeTokenizedChunks(const Tensor& tokenizedChunks, std::span<const Text::Prompts::PromptAttentionFrame> textChunks);
    void ApplyAttention(Tensor& encodedText, std::span<const float> attentionMask);

    TokenizedPrompt TokenizePrompt(std::string_view text);
  };
}