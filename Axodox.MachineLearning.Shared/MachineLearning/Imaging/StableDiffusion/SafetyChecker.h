#pragma once
#include "MachineLearning/Sessions/OnnxSession.h"
#include "MachineLearning/Tensor.h"
#include "SafetyCheckerOptions.h"
#include "Graphics/Textures/TextureData.h"
#include "StableDiffusionSessionParameters.h"

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  class AXODOX_MACHINELEARNING_API SafetyChecker
  {
    static inline const Infrastructure::logger _logger{ "SafetyChecker" };

  public:
    SafetyChecker(const Sessions::OnnxSessionParameters& parameters, SafetyCheckerOptions&& options = {});
    SafetyChecker(const StableDiffusionSessionParameters& parameters);

    bool IsSafe(const Graphics::TextureData& texture);

  private:
    Sessions::OnnxSessionContainer _sessionContainer;
    SafetyCheckerOptions _options;

    Tensor ToClipInput(const Graphics::TextureData& texture) const;
    Tensor ToImageInput(const Graphics::TextureData& texture) const;
  };
}