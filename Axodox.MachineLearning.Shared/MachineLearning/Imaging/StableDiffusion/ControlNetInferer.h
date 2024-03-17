#pragma once
#include "StableDiffustionInferer.h"
#include "StableDiffusionSessionParameters.h"

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  struct AXODOX_MACHINELEARNING_API ControlNetOptions : public StableDiffusionOptions
  {
    Tensor ConditionInput;
    float ConditioningScale = 1.f;

    void Validate() const;
  };

  class AXODOX_MACHINELEARNING_API ControlNetInferer : public ImageDiffusionInferer
  {
  public:
    ControlNetInferer(const Sessions::OnnxSessionParameters& controlnetParameters, const Sessions::OnnxSessionParameters& unetParameters);
    ControlNetInferer(const Sessions::OnnxSessionParameters& controlnetParameters, const StableDiffusionDirectorySessionParameters& unetParameters);

    Tensor RunInference(const ControlNetOptions& options, Threading::async_operation_source* async = nullptr);

    virtual ImageDiffusionInfererKind Type() const override;

  private:
    Sessions::OnnxSessionContainer _controlnetSessionContainer, _unetSessionContainer;
  };
}