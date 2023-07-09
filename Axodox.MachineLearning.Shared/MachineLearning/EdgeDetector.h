#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"
#include "ImageFeatureExtractor.h"

namespace Axodox::MachineLearning
{
  enum class EdgeDetectionMode
  {
    Canny,
    Hed
  };

  class AXODOX_MACHINELEARNING_API EdgeDetector : public ImageFeatureExtractor
  {
  public:
    EdgeDetector(OnnxEnvironment& environment, EdgeDetectionMode mode, std::optional<ModelSource> source = {});

    Tensor DetectEdges(const Tensor& image);

    virtual Graphics::TextureData ExtractFeatures(const Graphics::TextureData& value) override;

  private:
    OnnxEnvironment& _environment;
    Ort::Session _session;

    static const wchar_t* ToModelName(EdgeDetectionMode mode);
  };
}