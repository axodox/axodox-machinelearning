#include "pch.h"
#include "VaeDecoder.h"

using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  VaeDecoder::VaeDecoder(OnnxEnvironment& environment) :
    _environment(environment),
    _session(environment.CreateSession(_environment.RootPath() / L"vae_decoder/model.onnx")),
    _isHalfModel(true)
  { 
    Ort::AllocatorWithDefaultOptions ortAlloc;
    const size_t inputCount = _session.GetInputCount();
    for (size_t i = 0; i < inputCount; i++)
    {
        if (strcmp(_session.GetInputNameAllocated(i, ortAlloc).get(), "latent_sample") == 0)
        {
            if (_session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType() == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                _isHalfModel = false;
            }
            break;
        }
    }
  }

  Tensor VaeDecoder::DecodeVae(const Tensor& image)
  {
    //Load inputs
    auto inputValues = image.Split(image.Shape[0]);

    Tensor results;
    for (size_t i = 0; i < image.Shape[0]; i++)
    {
      //Bind values
      IoBinding bindings{ _session };
      bindings.BindInput("latent_sample", (_isHalfModel ? inputValues[i].ToHalf() : inputValues[i].ToSingle()).ToOrtValue(_environment.MemoryInfo()));
      bindings.BindOutput("sample", _environment.MemoryInfo());

      //Run inference
      _session.Run({}, bindings);

      //Get result
      auto outputValues = bindings.GetOutputValues();
      auto result = Tensor::FromOrtValue(outputValues[0]).ToSingle();

      if (!results.IsValid())
      {
        auto shape = result.Shape;
        shape[0] = image.Shape[0];
        results = { result.Type, shape };
      }

      memcpy(results.AsPointer<float>(i), result.AsPointer<float>(), result.ByteCount());
    }

    return results;
  }
}