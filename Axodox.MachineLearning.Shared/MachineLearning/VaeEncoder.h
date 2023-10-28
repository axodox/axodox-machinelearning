#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API VaeEncoder
  {
    static inline const Infrastructure::logger _logger{ "VaeEncoder" };

  public:
    VaeEncoder(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor EncodeVae(const Tensor& image);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
    bool _isUsingFloat16;
  };
}
