#include "pch.h"
#include "OnnxEnvironment.h"

using namespace Ort;
using namespace std;
#ifdef PLATFORM_WINDOWS
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Foundation::Diagnostics;
#endif

namespace Axodox::MachineLearning
{
#ifdef PLATFORM_WINDOWS
    FileLoggingSession OnnxEnvironment::_loggingSession = nullptr;;
#endif

  OnnxEnvironment::OnnxEnvironment(const std::filesystem::path& rootPath) :
    _rootPath(rootPath),
    _environment(nullptr),
    _memoryInfo(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
  {
#ifdef PLATFORM_WINDOWS
     if (!_loggingSession) _loggingSession = FileLoggingSession(L"OnnxEnvironment");
    _logChannel = LoggingChannel(winrt::to_hstring(_rootPath.string()));
    _loggingSession.AddLoggingChannel(_logChannel);
#endif
    _environment = Ort::Env(ORT_LOGGING_LEVEL_WARNING, _rootPath.string().c_str(), &OrtLoggingFunctionCallback, this);
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
    OrtSessionOptionsAppendExecutionProvider_DML(options, 0);
    return options;
  }
  
  Ort::SessionOptions OnnxEnvironment::CpuSessionOptions()
  {
    Ort::SessionOptions options;
    options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
    options.DisableMemPattern();
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    return options;
  }
  
  Ort::Session OnnxEnvironment::CreateSession(const std::filesystem::path& modelPath)
  {
    auto sessionOptions = DefaultSessionOptions();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    auto preferredModelPath = modelPath;
    preferredModelPath.make_preferred();

    return Session{ _environment, preferredModelPath.c_str(), sessionOptions };
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

  void ORT_API_CALL OnnxEnvironment::OrtLoggingFunctionCallback(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message)
  {
      reinterpret_cast<OnnxEnvironment*>(param)->OrtLoggingFunction(severity, category, logid, code_location, message);
  }

  void OnnxEnvironment::OrtLoggingFunction(OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message)
  {
#ifdef PLATFORM_WINDOWS
      LoggingLevel level = LoggingLevel::Verbose;
      const char* levelStr = "";
      switch (severity)
      {
      case ORT_LOGGING_LEVEL_VERBOSE: 
          levelStr = "ORT_LOGGING_LEVEL_VERBOSE";
          level = LoggingLevel::Verbose; 
          break;
      case ORT_LOGGING_LEVEL_INFO:
          levelStr = "ORT_LOGGING_LEVEL_INFO";
          level = LoggingLevel::Information; 
          break;
      case ORT_LOGGING_LEVEL_WARNING: 
          levelStr = "ORT_LOGGING_LEVEL_WARNING";
          level = LoggingLevel::Warning; 
          break;
      case ORT_LOGGING_LEVEL_ERROR: 
          levelStr = "ORT_LOGGING_LEVEL_ERROR";
          level = LoggingLevel::Error;
          break;
      case ORT_LOGGING_LEVEL_FATAL:
          levelStr = "ORT_LOGGING_LEVEL_VERBOSE";
          level = LoggingLevel::Critical; 
          break;
      default:
          throw invalid_argument("severity");
      }
      auto logMessage = format("{0}: {1} - {2} - {3} ({4})", levelStr, category, logid, message, code_location);
      _logChannel.LogMessage(winrt::to_hstring(logMessage), level);
      logMessage += "\n";
      ::OutputDebugStringA(logMessage.c_str());
#endif
  }
}