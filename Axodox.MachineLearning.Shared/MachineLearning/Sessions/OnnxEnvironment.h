#pragma once
#include "..\..\includes.h"

namespace Axodox::MachineLearning::Sessions
{
  class AXODOX_MACHINELEARNING_API OnnxEnvironment
  {
    static inline const Infrastructure::logger _logger{ "OnnxEnvironment" };

  public:
    OnnxEnvironment(std::string_view logId = "");

    Ort::Env& Environment();
    Ort::MemoryInfo& MemoryInfo();
    Ort::SessionOptions DefaultSessionOptions();
    Ort::RunOptions& RunOptions();

  private:
    const char* _logId;
    Ort::Env _environment;
    Ort::MemoryInfo _memoryInfo;
    Ort::RunOptions _runOptions;

    static void ORT_API_CALL OnOrtLogAdded(void* param, OrtLoggingLevel severity, const char* category, const char* logId, const char* codeLocation, const char* message);
  };
}