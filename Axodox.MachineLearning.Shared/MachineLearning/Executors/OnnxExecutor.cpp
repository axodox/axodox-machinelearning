#include "pch.h"
#include "OnnxExecutor.h"

namespace Axodox::MachineLearning::Executors
{
  OnnxExecutor::OnnxExecutor() :
    DeviceReset(_events)
  { }
}