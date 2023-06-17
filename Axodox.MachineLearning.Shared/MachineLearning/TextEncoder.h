#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API TextEncoder
  {
  public:
    TextEncoder(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor EncodeText(const Tensor& text);

  private:
    static const size_t _maxTokenCount;

    OnnxEnvironment& _environment;
    Ort::Session _session;
  };
}