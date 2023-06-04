#pragma once
#include "pch.h"

namespace Axodox::MachineLearning::Prompts
{
  struct AXODOX_MACHINELEARNING_API PromptTimeFrame
  {
    std::string Text;
    float Start;
    float End;

    bool operator==(const PromptTimeFrame&) const = default;
  };

  AXODOX_MACHINELEARNING_API std::vector<PromptTimeFrame> ParseTimeFrames(std::string_view prompt, float start = 0.f, float end = 1.f);

  AXODOX_MACHINELEARNING_API std::vector<PromptTimeFrame> SchedulePrompt(std::string_view prompt);
  AXODOX_MACHINELEARNING_API std::vector<std::string> SchedulePrompt(std::string_view prompt, uint32_t steps);
}