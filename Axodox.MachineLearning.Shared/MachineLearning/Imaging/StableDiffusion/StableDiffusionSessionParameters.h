#pragma once
#include "MachineLearning/Sessions/OnnxSession.h"
#include "MachineLearning/Executors/OnnxExecutor.h"

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  class AXODOX_MACHINELEARNING_API StableDiffusionSessionParameters
  {
  public:
    StableDiffusionSessionParameters(
      const std::shared_ptr<Sessions::OnnxEnvironment>& environment = nullptr,
      const std::shared_ptr<Executors::OnnxExecutor>& executor = nullptr);
    virtual ~StableDiffusionSessionParameters() = default;

    Sessions::OnnxSessionParameters TextTokenizer() const;
    Sessions::OnnxSessionParameters TextEncoder() const;
    Sessions::OnnxSessionParameters TextEncoder2() const;
    Sessions::OnnxSessionParameters VaeEncoder() const;
    Sessions::OnnxSessionParameters ControlNet() const;
    Sessions::OnnxSessionParameters UNet() const;
    Sessions::OnnxSessionParameters VaeDecoder() const;
    Sessions::OnnxSessionParameters SafetyChecker() const;

  protected:
    virtual std::unique_ptr<Sessions::OnnxModelSource> ResolveModel(std::string_view type) const = 0;

  private:
    std::shared_ptr<Sessions::OnnxEnvironment> _environment;
    std::shared_ptr<Executors::OnnxExecutor> _gpuExecutor;
    std::shared_ptr<Executors::OnnxExecutor> _cpuExecutor;
  };

  class AXODOX_MACHINELEARNING_API StableDiffusionDirectorySessionParameters : public StableDiffusionSessionParameters
  {
  public:
    StableDiffusionDirectorySessionParameters(
      const std::filesystem::path& directory,
      const std::shared_ptr<Sessions::OnnxEnvironment>& environment = nullptr,
      const std::shared_ptr<Executors::OnnxExecutor>& executor = nullptr);

  protected:
    virtual std::unique_ptr<Sessions::OnnxModelSource> ResolveModel(std::string_view type) const override;

  private:
    std::filesystem::path _directory;
  };
}