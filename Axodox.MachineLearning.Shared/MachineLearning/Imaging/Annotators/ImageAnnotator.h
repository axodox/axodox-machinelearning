#pragma once
#include "pch.h"
#include "Graphics/Textures/TextureData.h"

namespace Axodox::MachineLearning::Imaging::Annotators
{
  class AXODOX_MACHINELEARNING_API ImageAnnotator
  {
  public:
    virtual Graphics::TextureData ExtractFeatures(const Graphics::TextureData& value) = 0;
    virtual ~ImageAnnotator() = default;
  };
}