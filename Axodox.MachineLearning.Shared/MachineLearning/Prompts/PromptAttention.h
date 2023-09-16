#pragma once
#include "../../includes.h"

namespace Axodox::MachineLearning::Prompts
{
  struct AXODOX_MACHINELEARNING_API PromptAttentionFrame
  {
    std::string Text;
    float Attention;

    bool operator==(const PromptAttentionFrame&) const = default;
  };

  AXODOX_MACHINELEARNING_API std::vector<PromptAttentionFrame> ParseAttentionFrames(std::string_view prompt, float attention = 1.f);
}
