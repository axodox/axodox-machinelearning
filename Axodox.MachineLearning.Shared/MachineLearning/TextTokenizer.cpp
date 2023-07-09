#include "pch.h"
#include "TextTokenizer.h"
#include "OnnxExtensions.h"

using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  const size_t TextTokenizer::MaxTokenCount = 77;
  const int32_t TextTokenizer::StartToken = 49406;
  const int32_t TextTokenizer::BlankToken = 49407;

  TextTokenizer::TextTokenizer(OnnxEnvironment& environment, const std::filesystem::path& sourcePath) :
    _environment(environment),
    _sessionOptions(),
    _session(nullptr)
  {
    auto rootPath = sourcePath.empty() ? _environment.RootPath() / L"text_tokenizer" : sourcePath;

    _sessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    _sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    _sessionOptions.RegisterCustomOpsLibrary((rootPath / L"ortextensions.dll").c_str());
    _session = { _environment->Environment(), (rootPath / L"custom_op_cliptok.onnx").c_str(), _sessionOptions};
  }

  Tensor TextTokenizer::TokenizeText(std::string_view text)
  {
    return TokenizeText(std::vector<const char*>{ text.data() });
  }

  Tensor TextTokenizer::TokenizeText(const std::vector<const char*>& texts)
  {
    //Load inputs
    Allocator allocator{ _session, _environment->MemoryInfo() };
    
    vector<int64_t> inputShape{ int64_t(texts.size()) };
    auto inputValue = Value::CreateTensor(allocator, inputShape.data(), inputShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    
    inputValue.FillStringTensor(texts.data(), texts.size());

    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("string_input", inputValue);
    bindings.BindOutput("input_ids", _environment->MemoryInfo());
    bindings.BindOutput("attention_mask", _environment->MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

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
        *pToken++ = *pMask++ ? int32_t(*pSource++) : BlankToken;
      }
    }

    return result;
  }
  
  Tensor TextTokenizer::GetUnconditionalTokens()
  {
    Tensor result{ TensorType::Int32, 1, MaxTokenCount };
    ranges::fill(result.AsSpan<int32_t>(), BlankToken);

    *result.AsPointer<int32_t>(0) = 49406;
    return result;
  }
}