#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"
#include "Threading/AsyncOperation.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API VaeDecoder
  {
    static inline const Infrastructure::logger _logger{ "VaeDecoder" };

  public:
    VaeDecoder(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor DecodeVae(const Tensor& image, Threading::async_operation_source* async = nullptr);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
    bool _isUsingFloat16;
  };
}