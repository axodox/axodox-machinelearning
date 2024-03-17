#include "pch.h"
#include "OnnxEnvironment.h"

using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;
using namespace winrt;

namespace Axodox::MachineLearning::Sessions
{
  OnnxEnvironment::OnnxEnvironment(std::string_view logId) :
    _logId(logId.data()),
    _environment(ORT_LOGGING_LEVEL_WARNING, logId.data(), &OnOrtLogAdded, this),
    _memoryInfo(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
    _runOptions()
  {
    _environment.CreateAndRegisterAllocator(_memoryInfo, ArenaCfg(0, 1, -1, -1));
    _environment.DisableTelemetryEvents();
    //_runOptions.AddConfigEntry("memory.enable_memory_arena_shrinkage", "gpu:0");
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
    Ort::SessionOptions options;
    options.AddConfigEntry("session.use_env_allocators", "1");
    options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
    options.DisableMemPattern();
    options.DisableCpuMemArena();
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    return options;
  }

  Ort::RunOptions& OnnxEnvironment::RunOptions()
  {
    return _runOptions;
  }

  void ORT_API_CALL OnnxEnvironment::OnOrtLogAdded(void* param, OrtLoggingLevel severity, const char* category, const char* logId, const char* codeLocation, const char* message)
  {
    _logger.log(
      static_cast<log_severity>(severity),
      format("{} - {} ({})", category, message, codeLocation)
    );
  }
}