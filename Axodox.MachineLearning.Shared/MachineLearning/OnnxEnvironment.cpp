#include "pch.h"
#include "OnnxEnvironment.h"

using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  OnnxEnvironment::OnnxEnvironment(const std::filesystem::path& rootPath) :
    _rootPath(rootPath),
    _environment(ORT_LOGGING_LEVEL_WARNING, _rootPath.string().c_str(), &OnOrtLogAdded, this),
    _memoryInfo(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
    _runOptions()
  {
    _environment.CreateAndRegisterAllocator(_memoryInfo, ArenaCfg(0, 1, -1, -1));
    _environment.DisableTelemetryEvents();
    //_runOptions.AddConfigEntry("memory.enable_memory_arena_shrinkage", "gpu:0");
  }

  const std::filesystem::path& OnnxEnvironment::RootPath() const
  {
    return _rootPath;
  }

  Ort::Env& OnnxEnvironment::Environment()
  {
    return _environment;
  }

  Ort::MemoryInfo& OnnxEnvironment::MemoryInfo()
  {
    return _memoryInfo;
  }

  Ort::SessionOptions OnnxEnvironment::DefaultSessionOptions()
  {
    auto options = CpuSessionOptions();
    OrtSessionOptionsAppendExecutionProvider_DML(options, DeviceId);
    return options;
  }
  
  Ort::SessionOptions OnnxEnvironment::CpuSessionOptions()
  {
    Ort::SessionOptions options;
    options.AddConfigEntry("session.use_env_allocators", "1");
    options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
    options.DisableMemPattern();
    options.DisableCpuMemArena();
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    return options;
  }

  Ort::RunOptions& OnnxEnvironment::RunOptions()
  {
    return _runOptions;
  }
  
  Ort::Session OnnxEnvironment::CreateSession(ModelSource modelSource)
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

  Ort::Session OnnxEnvironment::CreateOptimizedSession(const std::filesystem::path& modelPath)
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

  void ORT_API_CALL OnnxEnvironment::OnOrtLogAdded(void* param, OrtLoggingLevel severity, const char* category, const char* logId, const char* codeLocation, const char* message)
  {
    _logger.log(
      static_cast<log_severity>(severity), 
      format("{} - {} ({})", category, message, codeLocation)
    );
  }
}