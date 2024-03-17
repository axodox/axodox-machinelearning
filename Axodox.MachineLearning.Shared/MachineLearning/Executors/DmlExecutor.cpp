#include "pch.h"
#include "DmlExecutor.h"

using namespace Axodox::Infrastructure;
using namespace std;
using namespace winrt;

namespace Axodox::MachineLearning::Executors
{
  void DmlExecutor::Ensure()
  {
    lock_guard lock(_mutex);

    //Check if device needs to be (re)created
    if (_d3d12Device && _d3d12Device->GetDeviceRemovedReason() == S_OK) return;
    
    //Create D3D12 device
    check_hresult(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, guid_of<ID3D12Device>(), _d3d12Device.put_void()));

    //Create command queue
    D3D12_COMMAND_QUEUE_DESC commandQueueDesc;
    zero_memory(commandQueueDesc);
    check_hresult(_d3d12Device->CreateCommandQueue(&commandQueueDesc, guid_of<ID3D12CommandQueue>(), _d3d12CommandQueue.put_void()));

    //Create DML device
    check_hresult(DMLCreateDevice(_d3d12Device.get(), DML_CREATE_DEVICE_FLAG_NONE, guid_of<IDMLDevice>(), _dmlDevice.put_void()));

    //Raise resetted event
    _events.raise(DeviceReset, this);
  }

  void DmlExecutor::Apply(Ort::SessionOptions& sessionOptions)
  {
    lock_guard lock(_mutex);
    OrtSessionOptionsAppendExecutionProviderEx_DML(sessionOptions, _dmlDevice.get(), _d3d12CommandQueue.get());
  }
}