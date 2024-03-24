#include "pch.h"
#include "TextEncoder.h"
#include "TextTokenizer.h"
#include "MachineLearning/Sessions/OnnxModelMetadata.h"

using namespace Axodox::Infrastructure;
using namespace Axodox::MachineLearning::Sessions;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  EncodedText EncodedText::Concat(const EncodedText& other) const
  {
    return EncodedText{
      .LastHiddenState = LastHiddenState.Concat(other.LastHiddenState),
      .TextEmbeds = TextEmbeds.Concat(other.TextEmbeds)
    };
  }

  TextEncoder::TextEncoder(const Sessions::OnnxSessionParameters& parameters) :
    _sessionContainer(parameters)
  { 
    auto metadata = OnnxModelMetadata::Create(_sessionContainer);
    _has64bitInputIds = metadata.Inputs["input_ids"].Type == TensorType::Int64;
    _hasHiddenLayers = metadata.Outputs.contains("hidden_states.11");    
    _logger.log(log_severity::information, "Loaded.");
  }

  TextEncoder::TextEncoder(const StableDiffusionSessionParameters& parameters) :
    TextEncoder(parameters.TextEncoder())
  { }

  Tensor TextEncoder::EncodeText(const Tensor& text)
  {
    _logger.log(log_severity::information, "Running inference...");

    //Get session
    auto environment = _sessionContainer.Environment();
    auto session = _sessionContainer.Session();

    //Bind values
    IoBinding bindings{ *session };
    bindings.BindInput("input_ids", text.ToInt64(_has64bitInputIds).ToOrtValue());
    bindings.BindOutput(_hasHiddenLayers ? "hidden_states.11" : "last_hidden_state", environment->MemoryInfo());

    //Run inference
    session->Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto result = Tensor::FromOrtValue(outputValues[0]).ToSingle();

    _logger.log(log_severity::information, "Inference finished.");
    return result;
  }

  TextEncoder2::TextEncoder2(const Sessions::OnnxSessionParameters& parameters) :
    _sessionContainer(parameters)
  {
    auto metadata = OnnxModelMetadata::Create(_sessionContainer);
    _has64bitInputIds = metadata.Inputs["input_ids"].Type == TensorType::Int64;
    _logger.log(log_severity::information, "Loaded.");
  }

  TextEncoder2::TextEncoder2(const StableDiffusionSessionParameters& parameters) :
    TextEncoder2(parameters.TextEncoder2())
  { }

  EncodedText TextEncoder2::EncodeText(const Tensor& text)
  {
    _logger.log(log_severity::information, "Running inference...");

    //Convert text encoding
    auto input = text;
    auto isEnding = false;
    for (auto& token : input.AsSpan<int32_t>())
    {
      if (isEnding) token = 0;
      if (token == TextTokenizer::EndToken) isEnding = true;
    }

    //Get session
    auto environment = _sessionContainer.Environment();
    auto session = _sessionContainer.Session();

    //Bind values
    IoBinding bindings{ *session };
    bindings.BindInput("input_ids", input.ToInt64(_has64bitInputIds).ToOrtValue());
    bindings.BindOutput("hidden_states.11", environment->MemoryInfo());
    bindings.BindOutput("text_embeds", environment->MemoryInfo());

    //Run inference
    session->Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();

    EncodedText result;
    result.LastHiddenState = Tensor::FromOrtValue(outputValues[0]).ToSingle();
    result.TextEmbeds = Tensor::FromOrtValue(outputValues[1]).ToSingle();
    _logger.log(log_severity::information, "Inference finished.");
    return result;
  }

  TextEncodingProvider::TextEncodingProvider(
    const Sessions::OnnxSessionParameters& encoder1Parameters,
    const Sessions::OnnxSessionParameters& encoder2Parameters) :
    _textEncoder(encoder1Parameters)
  {
    if (encoder2Parameters.IsValid())
    {
      _textEncoder2 = make_unique<TextEncoder2>(encoder2Parameters);
    }
  }

  TextEncodingProvider::TextEncodingProvider(const StableDiffusionSessionParameters& parameters) :
    TextEncodingProvider(parameters.TextEncoder(), parameters.TextEncoder2())
  { }

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