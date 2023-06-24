#include "pch.h"
#include "DepthEstimator.h"

using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  DepthEstimator::DepthEstimator(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment.CreateSession(source ? *source : (_environment.RootPath() / L"depth_estimator/model.onnx")))
  { }

  Tensor DepthEstimator::EstimateDepth(const Tensor& image)
  {
    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("x", image.ToHalf().ToOrtValue(_environment.MemoryInfo()));
    bindings.BindOutput("y", _environment.MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto result = Tensor::FromOrtValue(outputValues[0]).ToSingle();
    result.Shape = { result.Shape[0], 1, result.Shape[1], result.Shape[2] };

    return result;
  }
}