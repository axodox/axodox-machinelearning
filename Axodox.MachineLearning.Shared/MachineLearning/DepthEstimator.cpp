#include "pch.h"
#include "DepthEstimator.h"

using namespace Axodox::Graphics;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  DepthEstimator::DepthEstimator(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment.CreateSession(source ? *source : (_environment.RootPath() / L"annotators/depth.onnx")))
  { }

  Tensor DepthEstimator::EstimateDepth(const Tensor& image)
  {
    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("x", image.ToHalf().ToOrtValue());
    bindings.BindOutput("y", _environment.MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto result = Tensor::FromOrtValue(outputValues[0]).ToSingle();
    result.Shape = { result.Shape[0], 1, result.Shape[1], result.Shape[2] };

    return result;
  }

  Graphics::TextureData DepthEstimator::ExtractFeatures(const Graphics::TextureData& value)
  {
    auto inputTensor = Tensor::FromTextureData(value.Resize(512, 512), ColorNormalization::LinearZeroToOne);
    auto outputTensor = EstimateDepth(inputTensor);
    return outputTensor.ToTextureData(ColorNormalization::LinearZeroToOne).front().Resize(value.Width, value.Height);
  }
}