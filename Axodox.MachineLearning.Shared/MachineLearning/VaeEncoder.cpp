#include "pch.h"
#include "VaeEncoder.h"
#include "OnnxModelMetadata.h"

using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  VaeEncoder::VaeEncoder(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment->CreateSession(source ? *source : (_environment.RootPath() / L"vae_encoder/model.onnx")))
  { 
    auto metadata = OnnxModelMetadata::Create(_environment, _session);
    _isUsingFloat16 = metadata.Inputs["sample"].Type == TensorType::Half;

    _session.Evict();
    _logger.log(log_severity::information, "Loaded.");
  }

  Tensor VaeEncoder::EncodeVae(const Tensor& image)
  {
    _logger.log(log_severity::information, "Running inference...");

    //Load inputs
    auto inputValues = image.Split(image.Shape[0]);

    Tensor results;
    for (size_t i = 0; i < image.Shape[0]; i++)
    {
      //Bind values
      IoBinding bindings{ _session };
      bindings.BindInput("sample", inputValues[i].ToHalf(_isUsingFloat16).ToOrtValue());
      bindings.BindOutput("latent_sample", _environment->MemoryInfo());

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
    
    _session.Evict();
    _logger.log(log_severity::information, "Inference finished.");
    return results;
  }
}