#include "pch.h"
#include "EdgeDetector.h"

using namespace Axodox::Graphics;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  EdgeDetector::EdgeDetector(OnnxEnvironment& environment, EdgeDetectionMode mode, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment->CreateSession(source ? *source : (_environment.RootPath() / format(L"annotators/{}.onnx", ToModelName(mode)))))
  { }

  Tensor EdgeDetector::DetectEdges(const Tensor& image)
  {
    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("input", image.ToOrtValue());
    bindings.BindOutput("output", _environment->MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto result = Tensor::FromOrtValue(outputValues[0]);

    return result;
  }

  Graphics::TextureData EdgeDetector::ExtractFeatures(const Graphics::TextureData& value)
  {
    auto inputTensor = Tensor::FromTextureData(value, ColorNormalization::LinearZeroToOne);
    auto outputTensor = DetectEdges(inputTensor);
    return TextureData(outputTensor.ToTextureData(ColorNormalization::LinearZeroToOne).front());
  }

  const wchar_t* EdgeDetector::ToModelName(EdgeDetectionMode mode)
  {
    switch (mode)
    {
    case EdgeDetectionMode::Canny:
      return L"canny";
    case EdgeDetectionMode::Hed:
      return L"hed";
    default:
      throw logic_error("Edge detection mode not implemented.");
    }
  }
}