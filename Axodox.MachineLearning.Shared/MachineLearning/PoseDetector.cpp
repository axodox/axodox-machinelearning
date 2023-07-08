#include "pch.h"
#include "PoseDetector.h"
#include "Openpose/Openpose.h"

using namespace Axodox::Graphics;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  PoseDetector::PoseDetector(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment->CreateSession(source ? *source : (_environment.RootPath() / L"annotators/openpose.onnx")))
  { }

  Graphics::TextureData PoseDetector::DetectPose(const Tensor & image)
  {
    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("input", image.ToHalf().ToOrtValue());
    bindings.BindOutput("cmap", _environment->MemoryInfo());
    bindings.BindOutput("paf", _environment->MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto cmap = Tensor::FromOrtValue(outputValues[0]).ToSingle();
    auto paf = Tensor::FromOrtValue(outputValues[1]).ToSingle();

    auto frame = move(image.ToTextureData(ColorNormalization::LinearZeroToOne).front());

    Openpose op{ cmap.Shape };
    op.detect(cmap.AsPointer<float>(), paf.AsPointer<float>(), frame);

    return frame;
  }

  Graphics::TextureData PoseDetector::ExtractFeatures(const Graphics::TextureData& value)
  {
    auto inputTensor = Tensor::FromTextureData(value.Resize(224, 224), ColorNormalization::LinearZeroToOne);

    auto outputImage = DetectPose(inputTensor).Resize(value.Width, value.Height);
    return outputImage;
  }
}