#include "pch.h"
#include "DepthEstimator.h"

using namespace Axodox::Graphics;
using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning::Imaging::Annotators
{
  DepthEstimator::DepthEstimator(const Sessions::OnnxSessionParameters& parameters) :
    _sessionContainer(parameters)
  { }

  Tensor DepthEstimator::EstimateDepth(const Tensor& image)
  {
    _logger.log(log_severity::information, "Running inference...");

    //Get session
    auto environment = _sessionContainer.Environment();
    auto session = _sessionContainer.Session();

    //Bind values
    IoBinding bindings{ *session };
    bindings.BindInput("x", image.ToHalf().ToOrtValue());
    bindings.BindOutput("y", environment->MemoryInfo());

    //Run inference
    session->Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto result = Tensor::FromOrtValue(outputValues[0]).ToSingle();
    result.Shape = { result.Shape[0], 1, result.Shape[1], result.Shape[2] };
    
    _logger.log(log_severity::information, "Inference finished.");
    return result;
  }

  Graphics::TextureData DepthEstimator::ExtractFeatures(const Graphics::TextureData& value)
  {
    auto inputTensor = Tensor::FromTextureData(value.Resize(512, 512), ColorNormalization::LinearZeroToOne);
    auto outputTensor = EstimateDepth(inputTensor);    
    NormalizeDepthTensor(outputTensor);
    return outputTensor.ToTextureData(ColorNormalization::LinearZeroToOne).front().Resize(value.Width, value.Height);
  }

  void DepthEstimator::NormalizeDepthTensor(Tensor& value)
  {
    auto values = value.AsSpan<float>();

    float min = INFINITY, max = -INFINITY;
    for (auto value : values)
    {
      if (min > value) min = value;
      if (max < value) max = value;
    }

    auto range = max - min;
    for (auto& value : values)
    {
      value = (value - min) / range;
    }
  }
}