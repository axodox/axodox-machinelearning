#include "pch.h"
#include "TextTokenizer.h"

using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  const size_t TextTokenizer::MaxTokenCount = 77;
  const int32_t TextTokenizer::StartToken = 49406;
  const int32_t TextTokenizer::EndToken = 49407;

  TextTokenizer::TextTokenizer(const Sessions::OnnxSessionParameters& parameters) :
    _sessionContainer(parameters)
  { }

  TextTokenizer::TextTokenizer(const StableDiffusionSessionParameters& parameters) :
    TextTokenizer(parameters.TextTokenizer())
  { }

  Tensor TextTokenizer::TokenizeText(std::string_view text)
  {
    return TokenizeText(std::vector<const char*>{ text.data() });
  }

  Tensor TextTokenizer::TokenizeText(const std::vector<const char*>& texts)
  {
    _logger.log(log_severity::information, "Running inference...");

    //Get session
    auto environment = _sessionContainer.Environment();
    auto session = _sessionContainer.Session();

    //Load inputs
    Allocator allocator{ *session, environment->MemoryInfo() };
    
    vector<int64_t> inputShape{ int64_t(texts.size()) };
    auto inputValue = Value::CreateTensor(allocator, inputShape.data(), inputShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    
    inputValue.FillStringTensor(texts.data(), texts.size());

    //Bind values
    IoBinding bindings{ *session };
    bindings.BindInput("string_input", inputValue);
    bindings.BindOutput("input_ids", environment->MemoryInfo());
    bindings.BindOutput("attention_mask", environment->MemoryInfo());

    //Run inference
    session->Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto outputSpan = Tensor::FromOrtValue(outputValues[0]);
    auto attentionMask = Tensor::FromOrtValue(outputValues[1]);

    //Pad results to a fixed size
    Tensor result{ TensorType::Int32, outputSpan.Shape[0], outputSpan.Shape[1] };

    for (size_t i = 0; i < outputSpan.Shape[0]; i++)
    {
      auto sToken = result.AsPointer<int32_t>(i);
      auto pToken = sToken;

      auto pSource = outputSpan.AsPointer<int64_t>(i);
      auto pMask = attentionMask.AsPointer<int64_t>(i);
      for (size_t j = 0; j < outputSpan.Shape[1]; j++)
      {
        *pToken++ = *pMask++ ? int32_t(*pSource++) : EndToken;
      }
    }

    _logger.log(log_severity::information, "Inference finished.");
    return result;
  }
  
  Tensor TextTokenizer::GetUnconditionalTokens()
  {
    Tensor result{ TensorType::Int32, 1, MaxTokenCount };
    ranges::fill(result.AsSpan<int32_t>(), EndToken);

    *result.AsPointer<int32_t>(0) = StartToken;
    return result;
  }
}