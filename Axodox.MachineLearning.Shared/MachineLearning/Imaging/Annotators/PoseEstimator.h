#pragma once
#include "MachineLearning/Sessions/OnnxSession.h"
#include "MachineLearning/Tensor.h"
#include "ImageAnnotator.h"

namespace Axodox::MachineLearning::Imaging::Annotators
{
  const size_t PoseJointCount = 18;
  typedef std::array<DirectX::XMFLOAT2, PoseJointCount> PoseJointPositions;

  class AXODOX_MACHINELEARNING_API PoseEstimator : public ImageAnnotator
  {
    static inline const Infrastructure::logger _logger{ "PoseEstimator" };

  public:
    PoseEstimator(const Sessions::OnnxSessionParameters& parameters);

    std::vector<std::vector<PoseJointPositions>> EstimatePose(const Tensor& image);

    virtual Graphics::TextureData ExtractFeatures(const Graphics::TextureData& value) override;

  private:
    Sessions::OnnxSessionContainer _sessionContainer;
  };
}