#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API VaeDecoder
  {
    static inline const Infrastructure::logger _logger{ "VaeDecoder" };

  public:
    VaeDecoder(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor DecodeVae(const Tensor& image);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
    bool _isUsingFloat16;
  };
}