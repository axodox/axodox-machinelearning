#include "pch.h"
#include "EdgeDetector.h"

using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  EdgeDetector::EdgeDetector(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment.CreateSession(source ? *source : (_environment.RootPath() / L"edge_detector/canny.onnx")))
  { }

  Tensor EdgeDetector::DetectEdges(const Tensor& image)
  {
    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("img", image.ToOrtValue(_environment.MemoryInfo()));
    bindings.BindOutput("thin_edges", _environment.MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto result = Tensor::FromOrtValue(outputValues[0]);

    return result;
  }
}