#pragma once
#include "StableDiffustionInferer.h"

namespace Axodox::MachineLearning
{
  enum class ControlNetType
  {
    Unknown,
    Canny,
    Hed,
    Depth
  };

  struct AXODOX_MACHINELEARNING_API ControlNetOptions : public StableDiffusionOptions
  {
    ControlNetType ConditionType = ControlNetType::Unknown;
    Tensor ConditionInput;
    float ConditioningScale = 1.f;

    virtual void Validate() const override;
  };

  class AXODOX_MACHINELEARNING_API ControlNetInferer
  {
  public:
    ControlNetInferer(OnnxEnvironment& environment, const std::filesystem::path& controlnetPath, std::optional<ModelSource> unetSource = {});

    Tensor RunInference(const ControlNetOptions& options, Threading::async_operation_source* async = nullptr);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _unetSession, _controlnetSession;
    ControlNetType _controlnetType = ControlNetType::Unknown;
    std::filesystem::path _controlnetPath;
    
    void EnsureControlNet(ControlNetType type);

    static const wchar_t* ToString(ControlNetType type);
  };
}