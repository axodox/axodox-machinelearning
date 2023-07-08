#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"
#include "ImageFeatureExtractor.h"

namespace Axodox::MachineLearning
{
  const size_t PoseJointCount = 18;
  typedef std::array<DirectX::XMFLOAT2, PoseJointCount> PoseJointPositions;

  class AXODOX_MACHINELEARNING_API PoseDetector : public ImageFeatureExtractor
  {
  public:
    PoseDetector(OnnxEnvironment& environment, std::optional<ModelSource> source = {});

    std::vector<std::vector<PoseJointPositions>> EstimatePose(const Tensor& image);

    Graphics::TextureData DetectPose(const Tensor& image);

    virtual Graphics::TextureData ExtractFeatures(const Graphics::TextureData& value) override;

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;
  };
}