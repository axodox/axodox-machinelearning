#pragma once
#include "MachineLearning/Sessions/OnnxSession.h"
#include "MachineLearning/Tensor.h"

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  class AXODOX_MACHINELEARNING_API TextTokenizer
  {
    static inline const Infrastructure::logger _logger{ "TextTokenizer" };

  public:
    static const int32_t StartToken;
    static const int32_t EndToken;
    static const size_t MaxTokenCount;

    TextTokenizer(const Sessions::OnnxSessionParameters& parameters);

    Tensor TokenizeText(std::string_view text);
    Tensor TokenizeText(const std::vector<const char*>& texts);
    Tensor GetUnconditionalTokens();

  private:
    Sessions::OnnxSessionContainer _sessionContainer;
  };
}