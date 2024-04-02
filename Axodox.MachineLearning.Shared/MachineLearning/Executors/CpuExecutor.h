#pragma once
#include "OnnxExecutor.h"

namespace Axodox::MachineLearning::Executors
{
  class AXODOX_MACHINELEARNING_API CpuExecutor : public OnnxExecutor
  {
  public:
    virtual void Ensure() override;
    virtual void Apply(Ort::SessionOptions& sessionOptions) override;
  };
}