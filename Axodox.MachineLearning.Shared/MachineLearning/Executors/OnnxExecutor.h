#pragma once
#include "../../includes.h"

namespace Axodox::MachineLearning::Executors
{
  class AXODOX_MACHINELEARNING_API OnnxExecutor
  {
  protected:
    Infrastructure::event_owner _events;

  public:
    Infrastructure::event_publisher<OnnxExecutor*> DeviceReset;

    OnnxExecutor();
    virtual ~OnnxExecutor() = default;

    virtual void Ensure() = 0;
    virtual void Apply(Ort::SessionOptions& sessionOptions) = 0;
  };
}