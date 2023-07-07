#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"
#include "ImageFeatureExtractor.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API DepthEstimator : public ImageFeatureExtractor
  {
  public:
    DepthEstimator(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    Tensor EstimateDepth(const Tensor& image);

    virtual Graphics::TextureData ExtractFeatures(const Graphics::TextureData& value) override;

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
  };
}