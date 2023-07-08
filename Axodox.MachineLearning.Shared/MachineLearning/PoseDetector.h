#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"
#include "ImageFeatureExtractor.h"

namespace Axodox::MachineLearning
{
  class AXODOX_MACHINELEARNING_API PoseDetector : public ImageFeatureExtractor
  {
  public:
    PoseDetector(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    void DetectPose(const Tensor& image);

    virtual Graphics::TextureData ExtractFeatures(const Graphics::TextureData& value) override;

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
  };
}