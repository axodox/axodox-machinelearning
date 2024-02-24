#include "pch.h"
#include "VaeDecoder.h"
#include "OnnxModelMetadata.h"

using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  VaeDecoder::VaeDecoder(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment->CreateSession(source ? *source : (_environment.RootPath() / L"vae_decoder/model.onnx")))
  { 
    auto metadata = OnnxModelMetadata::Create(_environment, _session);
    _isUsingFloat16 = metadata.Inputs["latent_sample"].Type == TensorType::Half;
    
    _session.Evict();
    _logger.log(log_severity::information, "Loaded.");
  }

  Tensor VaeDecoder::DecodeVae(const Tensor& image, Threading::async_operation_source* async)
  {
    _logger.log(log_severity::information, "Running inference...");

    //Load inputs
    auto inputValues = image.Split(image.Shape[0]);

    Tensor results;
    for (size_t i = 0; i < image.Shape[0]; i++)
    {
      //Update status
      if (async)
      {
        async->update_state(NAN, format("Decoding VAE {}/{}...", i + 1, image.Shape[0]));
        if (async->is_cancelled()) return {};
      }

      //Bind values
      IoBinding bindings{ _session };
      bindings.BindInput("latent_sample", inputValues[i].ToHalf(_isUsingFloat16).ToOrtValue());
      bindings.BindOutput("sample", _environment->MemoryInfo());

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

    if (async)
    {
      async->update_state(1.f, "VAE decoded.");
    }

    _session.Evict();
    _logger.log(log_severity::information, "Inference finished.");
    return results;
  }
}