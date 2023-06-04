#include "pch.h"
#include "PromptParser.h"

using namespace std;

namespace Axodox::MachineLearning::Prompts
{
  std::pair<const char*, const char*> ToRange(std::string_view text)
  {
    return {
      text.data(),
      text.data() + text.size()
    };
  }

  std::string_view TrimWhitespace(std::string_view text)
  {
    if (text.empty()) return{};

    auto [start, end] = ToRange(text);

    auto loc = locale();
    while (isspace(*start, loc) && start < end) start++;
    while (isspace(*(end - 1), loc) && start < end) end--;

    return { start, end };
  }

  std::optional<float> TryParseNumber(std::string_view text)
  {
    if (text.empty()) return nullopt;

    auto [start, end] = ToRange(TrimWhitespace(text));
    
    float result;
    auto parseResult = from_chars(start, end, result);
    if (parseResult.ec == errc() && parseResult.ptr == end)
    {
      return result;
    }
    else
    {
      return nullopt;
    }
  }

  std::vector<std::string_view> SplitToSegments(const char*& text, char opener, char delimiter, char closer)
  {
    if (*text != opener) throw runtime_error("Frame must start with bracket.");

    vector<string_view> segments;

    auto depth = 0;
    const char* segment = nullptr;

    do
    {
      if (*text == opener)
      {
        depth++;
        if (depth == 1)
        {
          segment = text + 1;
        }
      }
      else if (*text == delimiter)
      {
        if (depth == 1)
        {
          segments.push_back({ segment, text });
          segment = text + 1;
        }
      }
      else if(*text == closer)
      {
        if (depth == 1)
        {
          segments.push_back({ segment, text });
        }
        depth--;
      }

      text++;
    } while (*text != '\0' && depth > 0);

    if (depth > 0) throw runtime_error("Unclosed bracket encountered.");

    return segments;
  }

  void CheckPromptCharacters(std::string_view text)
  {
    static set<char> specialCharacters{',', '.', ':', '?', '!', '/', '(', ')', '<', '>', '[', ']', '\'', '-', '_' };

    auto loc = locale();
    for (auto c : text)
    {
      if (!isalnum(c, loc) &&
        !isspace(c, loc) &&
        !specialCharacters.contains(c))
        throw runtime_error("Invalid character encountered.");
    }
  }
}