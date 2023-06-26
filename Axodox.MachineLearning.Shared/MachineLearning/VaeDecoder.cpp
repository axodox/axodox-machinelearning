#include "pch.h"
#include "VaeDecoder.h"

using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  VaeDecoder::VaeDecoder(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment.CreateSession(source ? *source : (_environment.RootPath() / L"vae_decoder/model.onnx")))
  { }

  Tensor VaeDecoder::DecodeVae(const Tensor& image)
  {
    //Load inputs
    auto inputValues = image.Split(image.Shape[0]);

    Tensor results;
    for (size_t i = 0; i < image.Shape[0]; i++)
    {
      //Bind values
      IoBinding bindings{ _session };
      bindings.BindInput("latent_sample", inputValues[i].ToHalf().ToOrtValue());
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