#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"
#include "SafetyCheckerOptions.h"
#include "Graphics/Textures/TextureData.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API SafetyChecker
  {
  public:
    SafetyChecker(OnnxEnvironment& environment);

    bool IsSafe(const Graphics::TextureData& texture);

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
    SafetyCheckerOptions _options;

    Tensor ToClipInput(const Graphics::TextureData& texture) const;
    Tensor ToImageInput(const Graphics::TextureData& texture) const;
  };
}