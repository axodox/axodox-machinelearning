#pragma once
#include "MachineLearning/Sessions/OnnxSession.h"
#include "MachineLearning/Tensor.h"
#include "StableDiffusionSessionParameters.h"

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  class AXODOX_MACHINELEARNING_API VaeEncoder
  {
    static inline const Infrastructure::logger _logger{ "VaeEncoder" };

  public:
    VaeEncoder(const Sessions::OnnxSessionParameters& parameters);
    VaeEncoder(const StableDiffusionSessionParameters& parameters);

    Tensor EncodeVae(const Tensor& image);

  private:
    Sessions::OnnxSessionContainer _sessionContainer;
    bool _isUsingFloat16;
  };
}
