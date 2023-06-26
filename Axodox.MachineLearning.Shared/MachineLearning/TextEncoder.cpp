#include "pch.h"
#include "TextEncoder.h"

using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  const size_t TextEncoder::_maxTokenCount = 77;

  TextEncoder::TextEncoder(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment.CreateSession(source ? *source : (_environment.RootPath() / L"text_encoder/model.onnx")))
  { }

  Tensor TextEncoder::EncodeText(const Tensor& text)
  {
    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("input_ids", text.ToOrtValue());
    bindings.BindOutput("last_hidden_state", _environment.MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();

    return Tensor::FromOrtValue(outputValues[0]).ToSingle();
  }
}