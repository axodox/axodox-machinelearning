#pragma once
#include "..\includes.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API OnnxEnvironment
  {
  public:
    OnnxEnvironment(const std::filesystem::path& rootPath);

    int DeviceId = 0;

    const std::filesystem::path& RootPath() const;
    Ort::Env& Environment();
    Ort::MemoryInfo& MemoryInfo();
    Ort::SessionOptions DefaultSessionOptions();
    Ort::SessionOptions CpuSessionOptions();

    Ort::Session CreateSession(const std::filesystem::path& modelPath);
    Ort::Session CreateOptimizedSession(const std::filesystem::path& modelPath);

  private:
    std::filesystem::path _rootPath;
    Ort::Env _environment;
    Ort::MemoryInfo _memoryInfo;
  };
}
