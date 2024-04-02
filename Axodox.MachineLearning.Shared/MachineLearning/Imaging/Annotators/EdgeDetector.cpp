#include "pch.h"
#include "EdgeDetector.h"

using namespace Axodox::Graphics;
using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning::Imaging::Annotators
{
  EdgeDetector::EdgeDetector(const Sessions::OnnxSessionParameters& parameters) :
    _sessionContainer(parameters)
  { }

  Tensor EdgeDetector::DetectEdges(const Tensor& image)
  {
    _logger.log(log_severity::information, "Running inference...");

    //Get session
    auto environment = _sessionContainer.Environment();
    auto session = _sessionContainer.Session();

    //Bind values
    IoBinding bindings{ *session };
    bindings.BindInput("input", image.ToOrtValue());
    bindings.BindOutput("output", environment->MemoryInfo());

    //Run inference
    session->Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto result = Tensor::FromOrtValue(outputValues[0]);

    _logger.log(log_severity::information, "Inference finished.");
    return result;
  }

  Graphics::TextureData EdgeDetector::ExtractFeatures(const Graphics::TextureData& value)
  {
    //Prepare input
    Graphics::Rect sourceRect;
    auto maxDimension = max(value.Width, value.Height);
    auto inputTensor = Tensor::FromTextureData(value.UniformResize(maxDimension, maxDimension, &sourceRect), ColorNormalization::LinearZeroToOne);

    //Detect edges
    auto outputTensor = DetectEdges(inputTensor);

    //Return output
    return TextureData(outputTensor.ToTextureData(ColorNormalization::LinearZeroToOne).front().GetTexture(sourceRect));
  }
}