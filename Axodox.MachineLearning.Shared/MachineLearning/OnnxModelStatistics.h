#pragma once
#ifdef USE_ONNX
#include "OnnxEnvironment.h"

namespace Axodox::MachineLearning
{
  AXODOX_MACHINELEARNING_API void OnnxPrintStatistics(OnnxEnvironment& environment, Ort::Session& session);
}
#endif