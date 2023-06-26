#include "pch.h"
#include "SafetyChecker.h"
#include "Storage/FileIO.h"

using namespace Axodox::Graphics;
using namespace Axodox::Json;
using namespace Axodox::Storage;
using namespace DirectX;
using namespace DirectX::PackedVector;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  SafetyChecker::SafetyChecker(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment.CreateSession(source ? *source : (_environment.RootPath() / L"safety_checker/model.onnx")))
  {
    auto text = try_read_text(_environment.RootPath() / L"feature_extractor/preprocessor_config.json");
    if (!text) return;

    auto value = try_parse_json<SafetyCheckerOptions>(*text);
    if(!value) return;

    _options = move(*value);
  }

  bool SafetyChecker::IsSafe(const Graphics::TextureData& texture)
  {
    auto clipInput = ToClipInput(texture);
    auto imageInput = ToImageInput(texture);

    IoBinding bindings{ _session };
    bindings.BindInput("clip_input", clipInput.ToHalf().ToOrtValue());
    bindings.BindInput("images", imageInput.ToHalf().ToOrtValue());
    bindings.BindOutput("has_nsfw_concepts", _environment.MemoryInfo());

    _session.Run({}, bindings);

    auto outputValues = bindings.GetOutputValues();
    auto result = Tensor::FromOrtValue(outputValues[0]);

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