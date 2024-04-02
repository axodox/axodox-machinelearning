#pragma once
#include "MachineLearning/Sessions/OnnxSession.h"
#include "MachineLearning/Tensor.h"
#include "ImageAnnotator.h"

namespace Axodox::MachineLearning::Imaging::Annotators
{
  class AXODOX_MACHINELEARNING_API DepthEstimator : public ImageAnnotator
  {
    static inline const Infrastructure::logger _logger{ "DepthEstimator" };

  public:
    DepthEstimator(const Sessions::OnnxSessionParameters& parameters);

    Tensor EstimateDepth(const Tensor& image);

    virtual Graphics::TextureData ExtractFeatures(const Graphics::TextureData& value) override;

    static void NormalizeDepthTensor(Tensor& value);

  private:
    Sessions::OnnxSessionContainer _sessionContainer;
  };
}