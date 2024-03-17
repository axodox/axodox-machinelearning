#pragma once
#include "OnnxExecutor.h"

namespace Axodox::MachineLearning::Executors
{
  class AXODOX_MACHINELEARNING_API DmlExecutor : public OnnxExecutor
  {
  public:
    virtual void Ensure() override;
    virtual void Apply(Ort::SessionOptions& sessionOptions) override;

  private:
    std::recursive_mutex _mutex;
    winrt::com_ptr<ID3D12Device> _d3d12Device;
    winrt::com_ptr<ID3D12CommandQueue> _d3d12CommandQueue;
    winrt::com_ptr<IDMLDevice> _dmlDevice;
  };
}