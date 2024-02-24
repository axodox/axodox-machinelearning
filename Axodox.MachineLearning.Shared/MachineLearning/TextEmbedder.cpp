#include "pch.h"
#include "TextEmbedder.h"
#include "Prompts/PromptSplitter.h"
#include "Prompts/PromptScheduler.h"
#include "Prompts/PromptParser.h"

using namespace Axodox::Infrastructure;
using namespace Axodox::MachineLearning::Prompts;
using namespace std;

namespace Axodox::MachineLearning
{
  TextEmbedder::TextEmbedder(OnnxEnvironment& environment, const std::filesystem::path& tokenizerPath, std::optional<ModelSource> encoderSource) :
    _textTokenizer(environment, tokenizerPath),
    _textEncoder(environment, encoderSource)
  { }

  int32_t TextEmbedder::ValidatePrompt(std::string_view text)
  {
    int32_t availableTokenCount = int32_t(TextTokenizer::MaxTokenCount);
    try
    {
      for (auto& fragment : SplitPrompt(text))
      {
        auto frames = ::SchedulePrompt(fragment.Prompt);

        for (auto& frame : frames)
        {
          auto tokenizedPrompt = TokenizePrompt(frame.Text);
          if (availableTokenCount > tokenizedPrompt.AvailableTokenCount) availableTokenCount = tokenizedPrompt.AvailableTokenCount;
        }
      }      
    }
    catch (...)
    {
      availableTokenCount = -1;
    }

    return availableTokenCount;
  }

  std::vector<ScheduledPrompt> TextEmbedder::SchedulePrompt(std::string_view text, uint32_t stepCount)
  {
    //Prepare fragments
    auto fragments = SplitPrompt(text);
    vector<vector<string>> scheduledFragments;
    scheduledFragments.reserve(fragments.size());

    auto totalWeight = 0.f;
    for (auto& fragment : fragments)
    {
      totalWeight += fragment.Weight;
      scheduledFragments.push_back(::SchedulePrompt(fragment.Prompt, stepCount));
    }

    //Create embeddings
    vector<ScheduledPrompt> embeddings;
    embeddings.reserve(stepCount);

    unordered_map<string, ScheduledPrompt> embeddingsByPrompt;
    for (auto i = 0u; i < stepCount; i++)
    {
      string key;
      for (auto j = 0u; j < fragments.size(); j++)
      {
        key += ";" + scheduledFragments[j][i];
      }

      auto& embedding = embeddingsByPrompt[key];
      if (!embedding.Tensor)
      {
        for (auto j = 0u; j < fragments.size(); j++)
        {
          auto partialEmbedding = make_shared<EncodedText>(ProcessPrompt(scheduledFragments[j][i]));

          embedding.Weights.push_back(fragments[j].Weight / totalWeight);

          if (j == 0)
          {
            embedding.Tensor = move(partialEmbedding);
          }
          else
          {
            *embedding.Tensor = embedding.Tensor->Concat(*partialEmbedding);
          }
        }
      }

      embeddings.push_back(embedding);
    }

    return embeddings;
  }

  EncodedText TextEmbedder::ProcessPrompt(std::string_view text)
  {
    auto tokenizedPrompt = TokenizePrompt(text);
    auto encodedText = _textEncoder.EncodeText(tokenizedPrompt.TokenizedText);
    ApplyAttention(encodedText.LastHiddenState, tokenizedPrompt.AttentionMask);

    return encodedText;
  }

  TextEmbedder::TokenizedPrompt TextEmbedder::MergeTokenizedChunks(const Tensor& tokenizedChunks, std::span<const PromptAttentionFrame> textChunks)
  {
    Tensor tokenizedTensor{ TensorType::Int32, 1, TextTokenizer::MaxTokenCount };
    auto tokenTarget = tokenizedTensor.AsSpan<int32_t>();
    auto pTokenTarget = tokenTarget.data();
    *pTokenTarget++ = TextTokenizer::StartToken;

    vector<float> attentionMask;
    attentionMask.resize(TextTokenizer::MaxTokenCount);
    auto pAttention = attentionMask.data();
    *pAttention++ = 1;

    int32_t availableSpace = int32_t(TextTokenizer::MaxTokenCount) - 1;
    for (size_t i = 0; i < tokenizedChunks.Shape[0]; i++)
    {
      auto tokenizedChunk = tokenizedChunks.AsSubSpan<int32_t>(i);
      auto lastToken = tokenizedChunk.end() - 1;
      while (lastToken > tokenizedChunk.begin() && *lastToken == TextTokenizer::BlankToken) lastToken--;

      auto tokensToCopy = int32_t(distance(tokenizedChunk.begin(), lastToken));
      auto copiableLength = min(tokensToCopy, availableSpace);
            
      copy(tokenizedChunk.begin() + 1, tokenizedChunk.begin() + 1 + copiableLength, pTokenTarget);
      pTokenTarget += copiableLength;
      
      fill(pAttention, pAttention + copiableLength, textChunks[i].Attention);
      pAttention += copiableLength;

      availableSpace -= copiableLength;
    }

    fill(pTokenTarget, tokenTarget.data() + tokenTarget.size(), TextTokenizer::BlankToken);
    fill(pAttention, attentionMask.data() + attentionMask.size(), 1.f);

    return { tokenizedTensor, attentionMask, availableSpace };
  }

  void TextEmbedder::ApplyAttention(Tensor& encodedText, std::span<const float> attentionMask)
  {
    auto encodedTokens = encodedText.AsSpan<float>();
    auto oldAverage = accumulate(encodedTokens.begin(), encodedTokens.end(), 0.f) / encodedTokens.size();

    for (auto i = 0; i < attentionMask.size(); i++)
    {
      auto encodedToken = encodedText.AsSubSpan<float>(0, i);
      
      auto scale = attentionMask[i];
      for (auto& encodedSubtoken : encodedToken)
      {
        encodedSubtoken *= scale;
      }
    }

    auto newAverage = accumulate(encodedTokens.begin(), encodedTokens.end(), 0.f) / encodedTokens.size();
    auto compensation = oldAverage / newAverage;
    for (auto& encodedSubtoken : encodedTokens)
    {
      encodedSubtoken *= compensation;
    }
  }

  TextEmbedder::TokenizedPrompt TextEmbedder::TokenizePrompt(std::string_view text)
  {
    auto chunks = ParseAttentionFrames(text.data());

    vector<const char*> texts;
    texts.reserve(chunks.size());
    for (auto& chunk : chunks)
    {
      CheckPromptCharacters(chunk.Text);
      texts.push_back(chunk.Text.c_str());
    }

    auto tokenizedTexts = _textTokenizer.TokenizeText(texts);
    return MergeTokenizedChunks(tokenizedTexts, chunks);
  }
}