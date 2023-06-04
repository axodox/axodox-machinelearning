#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API TextTokenizer
  {
  public:
    static const int32_t StartToken;
    static const int32_t BlankToken;
    static const size_t MaxTokenCount;

    TextTokenizer(OnnxEnvironment& environment, const std::filesystem::path& sourcePath = {});

    Tensor TokenizeText(std::string_view text);
    Tensor TokenizeText(const std::vector<const char*>& texts);
    Tensor GetUnconditionalTokens();

  private:

    OnnxEnvironment& _environment;
    Ort::SessionOptions _sessionOptions;
    Ort::Session _session;
  };
}