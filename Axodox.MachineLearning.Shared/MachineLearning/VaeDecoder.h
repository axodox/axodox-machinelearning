#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API VaeDecoder
  {
  public:
    VaeDecoder(OnnxEnvironment& environment);

    Tensor DecodeVae(const Tensor& text);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
  };
}