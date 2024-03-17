#pragma once
#include "MachineLearning/Sessions/OnnxSession.h"
#include "MachineLearning/Tensor.h"
#include "Threading/AsyncOperation.h"

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  class AXODOX_MACHINELEARNING_API VaeDecoder
  {
    static inline const Infrastructure::logger _logger{ "VaeDecoder" };

  public:
    VaeDecoder(const Sessions::OnnxSessionParameters& parameters);

    Tensor DecodeVae(const Tensor& image, Threading::async_operation_source* async = nullptr);

  private:
    Sessions::OnnxSessionContainer _sessionContainer;
    bool _isUsingFloat16;
  };
}