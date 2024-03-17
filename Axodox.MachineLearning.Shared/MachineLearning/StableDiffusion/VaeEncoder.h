#pragma once
#include "../Sessions/OnnxSession.h"
#include "../Tensor.h"

namespace Axodox::MachineLearning::StableDiffusion
{
  class AXODOX_MACHINELEARNING_API VaeEncoder
  {
    static inline const Infrastructure::logger _logger{ "VaeEncoder" };

  public:
    VaeEncoder(const Sessions::OnnxSessionParameters& parameters);

    Tensor EncodeVae(const Tensor& image);

  private:
    Sessions::OnnxSessionContainer _sessionContainer;
    bool _isUsingFloat16;
  };
}
