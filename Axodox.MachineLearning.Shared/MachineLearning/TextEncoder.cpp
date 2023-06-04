#include "pch.h"
#ifdef USE_ONNX
#include "TextEncoder.h"

using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  const size_t TextEncoder::_maxTokenCount = 77;

  TextEncoder::TextEncoder(OnnxEnvironment& environment) :
    _environment(environment),
    _session(environment.CreateSession(_environment.RootPath() / L"text_encoder/model.onnx"))
  { }

  Tensor TextEncoder::EncodeText(const Tensor& text)
  {
    //Load inputs
    auto inputValue = text.ToOrtValue(_environment.MemoryInfo());

    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("input_ids", inputValue);
    bindings.BindOutput("last_hidden_state", _environment.MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();

    return Tensor::FromOrtValue(outputValues[0]).ToSingle();
  }
}
#endif