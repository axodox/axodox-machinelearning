#pragma once
#include "MachineLearning/Sessions/OnnxSession.h"
#include "MachineLearning/Tensor.h"
#include "ImageAnnotator.h"

namespace Axodox::MachineLearning::Imaging::Annotators
{
  class AXODOX_MACHINELEARNING_API EdgeDetector : public ImageAnnotator
  {
    static inline const Infrastructure::logger _logger{ "EdgeDetector" };

  public:
    EdgeDetector(const Sessions::OnnxSessionParameters& parameters);

    Tensor DetectEdges(const Tensor& image);

    virtual Graphics::TextureData ExtractFeatures(const Graphics::TextureData& value) override;

  private:
    Sessions::OnnxSessionContainer _sessionContainer;
  };
}