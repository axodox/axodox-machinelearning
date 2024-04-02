#include "pch.h"
#include "VaeEncoder.h"
#include "../../Sessions/OnnxModelMetadata.h"

using namespace Axodox::Infrastructure;
using namespace Axodox::MachineLearning::Sessions;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  VaeEncoder::VaeEncoder(const Sessions::OnnxSessionParameters& parameters) :
    _sessionContainer(parameters)
  { 
    auto metadata = OnnxModelMetadata::Create(_sessionContainer);
    _isUsingFloat16 = metadata.Inputs["sample"].Type == TensorType::Half;
    _logger.log(log_severity::information, "Loaded.");
  }

  VaeEncoder::VaeEncoder(const StableDiffusionSessionParameters& parameters) :
    VaeEncoder(parameters.VaeEncoder())
  { }

  Tensor VaeEncoder::EncodeVae(const Tensor& image)
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
      //Bind values
      IoBinding bindings{ *session };
      bindings.BindInput("sample", inputValues[i].ToHalf(_isUsingFloat16).ToOrtValue());
      bindings.BindOutput("latent_sample", environment->MemoryInfo());

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
    
    _logger.log(log_severity::information, "Inference finished.");
    return results;
  }
}