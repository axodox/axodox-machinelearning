#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API DepthEstimator
  {
  public:
    DepthEstimator(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor EstimateDepth(const Tensor& image);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
  };
}