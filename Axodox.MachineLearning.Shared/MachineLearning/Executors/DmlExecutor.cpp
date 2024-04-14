#include "pch.h"
#include "DmlExecutor.h"

using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;
using namespace winrt;

namespace Axodox::MachineLearning::Executors
{
  DmlExecutor::DmlExecutor(uint32_t adapterIndex)
  {
    ThrowOnError(GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&_dmlApi)));
    ChangeAdapter(adapterIndex);
  }

  void DmlExecutor::ChangeAdapter(uint32_t adapterIndex)
  {
    //Get DXGI factory
    com_ptr<IDXGIFactory> factory;
    check_hresult(CreateDXGIFactory2(0, IID_PPV_ARGS(factory.put())));

    //Find adapter
    com_ptr<IDXGIAdapter> adapter;
    auto result = factory->EnumAdapters(adapterIndex, adapter.put());
    if (result == DXGI_ERROR_NOT_FOUND)
    {
      _logger.log(log_severity::warning, "Adapter with index {} could not be found.", adapterIndex);
      return;
    }

    //Update prescribed adapter
    lock_guard lock(_mutex);
    if (adapter == _dxgiAdapter) return;
    _dxgiAdapter = adapter;
    
    //Check if an adapter change is required
    if (adapter && _d3d12Device)
    {
      DXGI_ADAPTER_DESC adapterDesc;
      check_hresult(_dxgiAdapter->GetDesc(&adapterDesc));

      if (!are_equal(adapterDesc.AdapterLuid, _d3d12Device->GetAdapterLuid()))
      {
        _dmlDevice = nullptr;
        _d3d12CommandQueue = nullptr;
        _d3d12Device = nullptr;
      }
    }
  }

  void DmlExecutor::Ensure()
  {
    lock_guard lock(_mutex);

    //Check if device needs to be (re)created
    if (_d3d12Device && _d3d12Device->GetDeviceRemovedReason() == S_OK) return;
    
    //Note recreation due to device removal
    if (_d3d12Device) _logger.log(log_severity::warning, "Recreating DirectML executor due to device removal.");

    //Log DML init start
    _logger.log(log_severity::information, "Initializing DirectML executor...");

    //Create D3D12 device
    check_hresult(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(_d3d12Device.put())));

    //Create command queue
    D3D12_COMMAND_QUEUE_DESC commandQueueDesc;
    zero_memory(commandQueueDesc);
    check_hresult(_d3d12Device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(_d3d12CommandQueue.put())));

    //Create DML device
    check_hresult(DMLCreateDevice(_d3d12Device.get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(_dmlDevice.put())));

    //Raise resetted event
    _logger.log(log_severity::information, "Reinitializing sessions...");
    _events.raise(DeviceReset, this);

    //Log DML init end
    _logger.log(log_severity::information, "DirectML executor initialized successfully.");
  }

  void DmlExecutor::Apply(Ort::SessionOptions& sessionOptions)
  {
    lock_guard lock(_mutex);
    _dmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, _dmlDevice.get(), _d3d12CommandQueue.get());
  }
}