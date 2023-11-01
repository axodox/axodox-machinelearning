#include "pch.h"
#include "TextEncoder.h"
#include "OnnxModelMetadata.h"

using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  EncodedText EncodedText::Concat(const EncodedText& other) const
  {
    return EncodedText{
      .LastHiddenState = LastHiddenState.Concat(other.LastHiddenState),
      .TextEmbeds = TextEmbeds.Concat(other.TextEmbeds)
    };
  }

  TextEncoder::TextEncoder(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment->CreateSession(source ? *source : (_environment.RootPath() / L"text_encoder/model.onnx")))
  { 
    auto metadata = OnnxModelMetadata::Create(_environment, _session);
    _has64bitInputIds = metadata.Inputs["input_ids"].Type == TensorType::Int64;
    
    _logger.log(log_severity::information, "Loaded.");
  }

  Tensor TextEncoder::EncodeText(const Tensor& text)
  {
    _logger.log(log_severity::information, "Running inference...");

    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("input_ids", text.ToInt64(_has64bitInputIds).ToOrtValue());
    bindings.BindOutput("last_hidden_state", _environment->MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto result = Tensor::FromOrtValue(outputValues[0]).ToSingle();

    _session.Evict();
    _logger.log(log_severity::information, "Inference finished.");
    return result;
  }

  TextEncoder2::TextEncoder2(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment->CreateSession(source ? *source : (_environment.RootPath() / L"text_encoder_2/model.onnx")))
  {
    auto metadata = OnnxModelMetadata::Create(_environment, _session);
    _has64bitInputIds = metadata.Inputs["input_ids"].Type == TensorType::Int64;

    _session.Evict();
    _logger.log(log_severity::information, "Loaded.");
  }

  EncodedText TextEncoder2::EncodeText(const Tensor& text)
  {
    _logger.log(log_severity::information, "Running inference...");

    //Convert text encoding
    auto input = text;
    auto isEnding = false;
    for (auto& token : input.AsSpan<int32_t>())
    {
      if (isEnding) token = 0;
      if (token == 49407) isEnding = true;
    }

    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("input_ids", input.ToInt64(_has64bitInputIds).ToOrtValue());
    bindings.BindOutput("last_hidden_state", _environment->MemoryInfo());
    bindings.BindOutput("text_embeds", _environment->MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();

    EncodedText result;
    result.LastHiddenState = Tensor::FromOrtValue(outputValues[0]).ToSingle();
    result.TextEmbeds = Tensor::FromOrtValue(outputValues[1]).ToSingle();

    _session.Evict();
    _logger.log(log_severity::information, "Inference finished.");
    return result;
  }

  TextEncodingProvider::TextEncodingProvider(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _textEncoder(environment, source)
  {
    if (source && !holds_alternative<filesystem::path>(*source)) return;

    if (source)
    {
      auto& path = get<filesystem::path>(*source);
      path = path.parent_path().parent_path() / L"text_encoder_2/model.onnx";
    }
    else
    {
      source = environment.RootPath() / L"text_encoder_2/model.onnx";
    }

    error_code ec;
    if (filesystem::exists(get<filesystem::path>(*source), ec))
    {
      _textEncoder2 = make_unique<TextEncoder2>(environment, source);
    }
  }

  EncodedText TextEncodingProvider::EncodeText(const Tensor& text)
  {
    EncodedText result;
    result.LastHiddenState = _textEncoder.EncodeText(text);

    if (_textEncoder2)
    {
      auto result2 = _textEncoder2->EncodeText(text);
      
      if (result.LastHiddenState.Shape != TensorShape{ 1, 77, 768, 0 } ||
        result2.LastHiddenState.Shape != TensorShape{ 1,77, 1280,0 }) throw bad_cast();

      Tensor combinedHiddenState{ TensorType::Single, {1, 77, 2048} };
      for (auto i = 0; i < 77; i++)
      {
        auto encoding1 = result.LastHiddenState.AsSubSpan<float>(0, i);
        auto encoding2 = result2.LastHiddenState.AsSubSpan<float>(0, i);
        auto target = combinedHiddenState.AsSubSpan<float>(0, i);
        copy(encoding1.begin(), encoding1.begin() + 768, target.begin());
        copy(encoding2.begin(), encoding2.begin() + 1280, target.begin() + 768);
      }
      
      result.LastHiddenState = move(combinedHiddenState);
      result.TextEmbeds = move(result2.TextEmbeds);
    }

    return result;
  }
}