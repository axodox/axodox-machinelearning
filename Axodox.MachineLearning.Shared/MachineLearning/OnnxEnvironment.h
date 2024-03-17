#pragma once
#include "..\includes.h"

namespace Axodox::MachineLearning
{
  typedef std::variant<std::filesystem::path, std::span<const uint8_t>> ModelSource;

  class AXODOX_MACHINELEARNING_API OnnxHost
  {
    static inline const Infrastructure::logger _logger{"OnnxHost"};

  public:
    OnnxHost(const char* logId = "");

    int DeviceId = 0;

    Ort::Env& Environment();
    Ort::MemoryInfo& MemoryInfo();
    Ort::SessionOptions DefaultSessionOptions();
    Ort::SessionOptions CpuSessionOptions();
    Ort::RunOptions& RunOptions();

    bool IsValid() const;
    void Reset();
    bool ResetIfFailed();

    Ort::Session CreateSession(ModelSource modelPath);
    Ort::Session CreateSession(std::span<const uint8_t> modelBuffer);
    Ort::Session CreateOptimizedSession(const std::filesystem::path& modelPath);

  private:
    const char* _logId;
    Ort::Env _environment;
    Ort::MemoryInfo _memoryInfo;
    Ort::RunOptions _runOptions;   

    static void ORT_API_CALL OnOrtLogAdded(void* param, OrtLoggingLevel severity, const char* category, const char* logId, const char* codeLocation, const char* message);
  };

  class AXODOX_MACHINELEARNING_API OnnxEnvironment
  {
  public:
    OnnxEnvironment(const std::filesystem::path& rootPath);
    OnnxEnvironment(const std::shared_ptr<OnnxHost>& host, const std::filesystem::path& rootPath);

    const std::filesystem::path& RootPath() const;

    OnnxHost* operator->() const;
    OnnxHost* operator*() const;

  private:
    std::shared_ptr<OnnxHost> _host;
    std::filesystem::path _rootPath;
  };
}
