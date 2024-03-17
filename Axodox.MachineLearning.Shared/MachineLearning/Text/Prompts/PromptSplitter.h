#pragma once
#include "../../../includes.h"

namespace Axodox::MachineLearning::Text::Prompts
{
  struct AXODOX_MACHINELEARNING_API PartialPrompt
  {
    float Weight;
    std::string Prompt;
  };

  AXODOX_MACHINELEARNING_API std::vector<PartialPrompt> SplitPrompt(std::string_view prompt);
}