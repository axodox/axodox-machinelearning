#pragma once
#include "OnnxEnvironment.h"
#include "TensorInfo.h"

namespace Axodox::MachineLearning
{
  struct AXODOX_MACHINELEARNING_API OnnxModelMetadata
  {
    std::string Name;
    std::unordered_map<std::string, TensorInfo> Initializers, Inputs, Outputs;

    static OnnxModelMetadata Create(OnnxEnvironment& environment, Ort::Session& session);
  };
}