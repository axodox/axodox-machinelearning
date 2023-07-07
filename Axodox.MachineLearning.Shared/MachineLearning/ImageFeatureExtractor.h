#pragma once
#include "pch.h"
#include "Graphics/Textures/TextureData.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API ImageFeatureExtractor
  {
  public:
    virtual Graphics::TextureData ExtractFeatures(const Graphics::TextureData& value) = 0;
    virtual ~ImageFeatureExtractor() = default;
  };
}