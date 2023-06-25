#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API VaeEncoder
  {
  public:
    VaeEncoder(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor EncodeVae(const Tensor& image);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
  };
}
