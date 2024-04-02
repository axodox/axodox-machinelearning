#include "pch.h"
#include "PromptSplitter.h"
#include "PromptScheduler.h"

using namespace Axodox::Infrastructure;
using namespace std;

namespace
{
  float ParseWeight(std::string_view& text)
  {
    auto start = text.find_last_of(':') + 1;

    auto pChar = text.data() + start;
    for (auto i = start; i < text.size(); i++)
    {
      if (isdigit(*pChar) || isspace(*pChar) || *pChar == '.')
      {
        pChar++;
      }
      else
      {
        return 1.f;
      }
    }

    float result;
    auto parseResult = from_chars(text.data() + start, text.data() + text.size(), result);
    if (parseResult.ec == errc())
    {
      text = { text.data(), start - 1 };
      return result;
    }
    else
    {
      return 1.f;
    }
  }
}

namespace Axodox::MachineLearning::Text::Prompts
{
  std::vector<PartialPrompt> SplitPrompt(std::string_view prompt)
  {
    auto fragments = split(prompt, ';');
    
    vector<PartialPrompt> results;
    results.reserve(fragments.size());
    for (auto fragment : fragments)
    {
      if (fragment.empty()) continue;

      auto weight = ParseWeight(fragment);
      
      PartialPrompt result;
      result.Weight = weight;
      result.Prompt = fragment;
      results.push_back(move(result));
    }

    return results;
  }
}