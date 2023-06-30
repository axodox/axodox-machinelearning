#pragma once
#include "StableDiffustionInferer.h"

namespace Axodox::MachineLearning
{
  struct AXODOX_MACHINELEARNING_API ControlNetOptions : public StableDiffusionOptions
  {
    std::string ConditionType = "";
    Tensor ConditionInput;
    float ConditioningScale = 1.f;

    void Validate() const;
  };

  class AXODOX_MACHINELEARNING_API ControlNetInferer : public ImageDiffusionInferer
  {
  public:
    ControlNetInferer(OnnxEnvironment& environment, const std::filesystem::path& controlnetPath, std::optional<ModelSource> unetSource = {});

    Tensor RunInference(const ControlNetOptions& options, Threading::async_operation_source* async = nullptr);

    virtual ImageDiffusionInfererKind Type() const override;

  private:
    OnnxEnvironment& _environment;
    Ort::Session _unetSession, _controlnetSession;
    std::string _controlnetType = "";
    std::filesystem::path _controlnetPath;
    
    void EnsureControlNet(std::string_view type);
  };
}