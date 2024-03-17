#include "pch.h"
#include "SafetyChecker.h"
#include "Storage/FileIO.h"

using namespace Axodox::Graphics;
using namespace Axodox::Infrastructure;
using namespace Axodox::Json;
using namespace Axodox::Storage;
using namespace DirectX;
using namespace DirectX::PackedVector;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  SafetyChecker::SafetyChecker(const Sessions::OnnxSessionParameters& parameters, SafetyCheckerOptions&& options) :
    _sessionContainer(parameters),
    _options(move(options))
  { }

  SafetyChecker::SafetyChecker(const StableDiffusionSessionParameters& parameters) :
    SafetyChecker(parameters.SafetyChecker())
  { }

  bool SafetyChecker::IsSafe(const Graphics::TextureData& texture)
  {
    _logger.log(log_severity::information, "Running inference...");

    //Get session
    auto environment = _sessionContainer.Environment();
    auto session = _sessionContainer.Session();

    //Load inputs
    auto clipInput = ToClipInput(texture);
    auto imageInput = ToImageInput(texture);

    //Bind values
    IoBinding bindings{ *session };
    bindings.BindInput("clip_input", clipInput.ToHalf().ToOrtValue());
    bindings.BindInput("images", imageInput.ToHalf().ToOrtValue());
    bindings.BindOutput("has_nsfw_concepts", environment->MemoryInfo());

    //Run inference
    session->Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto result = Tensor::FromOrtValue(outputValues[0]);

    //Evict model on end
    session->Evict();
    _logger.log(log_severity::information, "Inference finished.");

    return !result.AsSpan<bool>()[0];
  }

  Tensor SafetyChecker::ToClipInput(const Graphics::TextureData& originalTexture) const
  {
    auto texture{ originalTexture };

    if (*_options.DoResize || *_options.DoCenterCrop)
    {
      texture = texture.UniformResize(*_options.CropSize->Width, *_options.CropSize->Height);
    }

    auto tensor = Tensor::FromTextureData(texture, ColorNormalization::LinearZeroToOne);

    if (*_options.DoNormalize)
    {
      for (auto channel = 0; channel < 3; channel++)
      {
        auto mean = _options.ImageMean->at(channel);
        auto std = _options.ImageStd->at(channel);
        for (auto& item : tensor.AsSubSpan<float>(0, channel))
        {
          item = (item - mean) / std;
        }
      }
    }

    return tensor;
  }

  Tensor SafetyChecker::ToImageInput(const Graphics::TextureData& texture) const
  {
    Tensor result{ TensorType::Single, 1, texture.Height, texture.Width, 3 };

    auto target = result.AsPointer<float>();
    for (auto i = 0u; i < texture.Height; i++)
    {
      auto source = texture.Row<XMUBYTEN4>(i);

      XMFLOAT4 color;
      for (auto j = 0u; j < texture.Width; j++)
      {
        XMStoreFloat4(&color, XMLoadUByteN4(source++));
        *target++ = color.z;
        *target++ = color.y;
        *target++ = color.x;
      }
    }

    return result;
  }
}