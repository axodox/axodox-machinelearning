#include "pch.h"
#include "OnnxEnvironment.h"

using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;
using namespace winrt;

namespace Axodox::MachineLearning
{
  OnnxHost::OnnxHost(const char* logId) :
    _logId(logId),
    _environment(ORT_LOGGING_LEVEL_WARNING, logId, &OnOrtLogAdded, this),
    _memoryInfo(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
    _runOptions()
  {
    _environment.CreateAndRegisterAllocator(_memoryInfo, ArenaCfg(0, 1, -1, -1));
    _environment.DisableTelemetryEvents();
    //_runOptions.AddConfigEntry("memory.enable_memory_arena_shrinkage", "gpu:0");
  }

  Ort::Env& OnnxHost::Environment()
  {
    return _environment;
  }

  Ort::MemoryInfo& OnnxHost::MemoryInfo()
  {
    return _memoryInfo;
  }

  Ort::SessionOptions OnnxHost::DefaultSessionOptions()
  {
    auto options = CpuSessionOptions();
    
    //OrtSessionOptionsAppendExecutionProviderEx_DML(options, _dmlDevice.get(), _d3d12CommandQueue.get());
    //OrtSessionOptionsAppendExecutionProvider_DML(options, DeviceId);
    return options;
  }
  
  Ort::SessionOptions OnnxHost::CpuSessionOptions()
  {
    Ort::SessionOptions options;
    options.AddConfigEntry("session.use_env_allocators", "1");
    options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
    options.DisableMemPattern();
    options.DisableCpuMemArena();
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    return options;
  }

  Ort::RunOptions& OnnxHost::RunOptions()
  {
    return _runOptions;
  }

  bool OnnxHost::IsValid() const
  {
    //return _d3d12Device->GetDeviceRemovedReason() == S_OK;
    return true;
  }

  void OnnxHost::Reset()
  {
    auto logId = _logId;
    this->~OnnxHost();
    new (this) OnnxHost(logId);
  }

  bool OnnxHost::ResetIfFailed()
  {
    if (!IsValid())
    {
      Reset();
      return true;
    }
    else
    {
      return false;
    }
  }
  
  Ort::Session OnnxHost::CreateSession(ModelSource modelSource)
  {
    auto sessionOptions = DefaultSessionOptions();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    if (holds_alternative<filesystem::path>(modelSource))
    {
      auto preferredModelPath = get<filesystem::path>(modelSource);
      preferredModelPath.make_preferred();

      return Session{ _environment, preferredModelPath.c_str(), sessionOptions };
    }
    else
    {
      auto modelData = get<span<const uint8_t>>(modelSource);
      return Session{ _environment, modelData.data(), modelData.size(), sessionOptions};
    }
  }

  Ort::Session OnnxHost::CreateSession(std::span<const uint8_t> modelBuffer)
  {
    auto sessionOptions = DefaultSessionOptions();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    return Session{ _environment, modelBuffer.data(), modelBuffer.size(), sessionOptions };
  }

  Ort::Session OnnxHost::CreateOptimizedSession(const std::filesystem::path& modelPath)
  {
    auto sessionOptions = DefaultSessionOptions();
    
    auto optimizedModelPath = modelPath;
    optimizedModelPath.replace_extension("optimized.onnx");
    optimizedModelPath.make_preferred();

    const filesystem::path* sourcePath;
    if (filesystem::exists(optimizedModelPath))
    {
      sourcePath = &optimizedModelPath;
      sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    }
    else
    {
      sourcePath = &modelPath;
      sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      sessionOptions.SetOptimizedModelFilePath(optimizedModelPath.c_str());
    }

    return Session{ _environment, sourcePath->c_str(), sessionOptions};
  }

  void ORT_API_CALL OnnxHost::OnOrtLogAdded(void* param, OrtLoggingLevel severity, const char* category, const char* logId, const char* codeLocation, const char* message)
  {
    _logger.log(
      static_cast<log_severity>(severity), 
      format("{} - {} ({})", category, message, codeLocation)
    );
  }

  OnnxEnvironment::OnnxEnvironment(const std::filesystem::path& rootPath) :
    OnnxEnvironment(make_shared<OnnxHost>(), rootPath)
  { }

  OnnxEnvironment::OnnxEnvironment(const std::shared_ptr<OnnxHost>& host, const std::filesystem::path& rootPath) :
    _host(host),
    _rootPath(rootPath)
  { }

  OnnxHost* OnnxEnvironment::operator->() const
  {
    return _host.get();
  }

  OnnxHost* OnnxEnvironment::operator*() const
  {
    return _host.get();
  }

  const std::filesystem::path& OnnxEnvironment::RootPath() const
  {
    return _rootPath;
  }
}