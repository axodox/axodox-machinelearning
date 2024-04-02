#pragma once
#include "OnnxSession.h"
#include "../TensorInfo.h"

namespace Axodox::MachineLearning::Sessions
{
  struct AXODOX_MACHINELEARNING_API OnnxModelMetadata
  {
    std::string Name;
    std::unordered_map<std::string, TensorInfo> Initializers, Inputs, Outputs;

    static OnnxModelMetadata Create(OnnxSessionContainer& sessionContainer);
  };
}