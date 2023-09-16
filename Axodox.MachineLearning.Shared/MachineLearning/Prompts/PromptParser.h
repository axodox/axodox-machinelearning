#pragma once
#include "../../includes.h"

namespace Axodox::MachineLearning::Prompts
{
  AXODOX_MACHINELEARNING_API std::pair<const char*, const char*> ToRange(std::string_view text);

  AXODOX_MACHINELEARNING_API std::string_view TrimWhitespace(std::string_view text);

  AXODOX_MACHINELEARNING_API std::optional<float> TryParseNumber(std::string_view text);

  AXODOX_MACHINELEARNING_API std::vector<std::string_view> SplitToSegments(const char*& text, char opener, char delimiter, char closer);

  AXODOX_MACHINELEARNING_API void CheckPromptCharacters(std::string_view text);
}