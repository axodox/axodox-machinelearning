#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API EdgeDetector
  {
  public:
    EdgeDetector(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor DetectEdges(const Tensor& image);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
  };
}