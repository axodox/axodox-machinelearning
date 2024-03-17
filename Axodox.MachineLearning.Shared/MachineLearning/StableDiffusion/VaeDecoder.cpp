#include "pch.h"
#include "VaeDecoder.h"
#include "../Sessions/OnnxModelMetadata.h"

using namespace Axodox::Infrastructure;
using namespace Axodox::MachineLearning::Sessions;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning::StableDiffusion
{
  VaeDecoder::VaeDecoder(const Sessions::OnnxSessionParameters& parameters) :
    _sessionContainer(parameters)
  { 
    auto metadata = OnnxModelMetadata::Create(_sessionContainer);
    _isUsingFloat16 = metadata.Inputs["latent_sample"].Type == TensorType::Half;
    _logger.log(log_severity::information, "Loaded.");
  }

  Tensor VaeDecoder::DecodeVae(const Tensor& image, Threading::async_operation_source* async)
  {
    _logger.log(log_severity::information, "Running inference...");

    //Get session
    auto environment = _sessionContainer.Environment();
    auto session = _sessionContainer.Session();

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
      IoBinding bindings{ *session };
      bindings.BindInput("latent_sample", inputValues[i].ToHalf(_isUsingFloat16).ToOrtValue());
      bindings.BindOutput("sample", environment->MemoryInfo());

      //Run inference
      session->Run({}, bindings);

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

    //Evict model on end
    session->Evict();
    _logger.log(log_severity::information, "Inference finished.");

    return results;
  }
}