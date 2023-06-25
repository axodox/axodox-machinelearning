#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API VaeDecoder
  {
  public:
    VaeDecoder(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor DecodeVae(const Tensor& image);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
  };
}