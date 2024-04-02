#include "pch.h"
#include "PromptAttention.h"
#include "PromptParser.h"

using namespace Axodox::MachineLearning::Text::Prompts;
using namespace std;

namespace
{
  void ReadFrame(const char*& text, float& attention, std::string_view& prompt)
  {
    //Split up to segments
    auto segments = SplitToSegments(text, '(', ':', ')');

    //Interpret segments
    string_view promptSegment, attentionSegment;
    switch (segments.size())
    {
    case 0: // ()
      //Do nothing
      break;
    case 1: // (prompt)
      promptSegment = segments[0];
      break;
    case 2: // (prompt:attention)
      promptSegment = segments[0];
      attentionSegment = segments[1];
      break;
    default:
      throw runtime_error("Too many segments in a prompt attention frame.");
    }

    auto attentionValue = TryParseNumber(attentionSegment);
    if(!attentionSegment.empty() && !attentionValue) throw runtime_error("Could not parse attention frame value.");
    attention = attentionValue ? *attentionValue : 1.f;

    prompt = promptSegment;
  }

  std::vector<PromptAttentionFrame> CleanAttentionFrames(const std::vector<PromptAttentionFrame>& frames)
  {
    vector<PromptAttentionFrame> results;
    results.reserve(frames.size());

    for (auto i = 0; i < frames.size(); i++)
    {
      auto frame = frames[i];
      frame.Text = TrimWhitespace(frame.Text);

      if (!frame.Text.empty() && frame.Attention > 0.f)
      {
        if (!results.empty() && results.back().Attention == frame.Attention)
        {
          results.back().Text += " " + frame.Text;
        }
        else
        {
          results.push_back(frame);
        }
      }
    }

    return results;
  }
}

namespace Axodox::MachineLearning::Text::Prompts
{
  std::vector<PromptAttentionFrame> ParseAttentionFrames(std::string_view prompt, float attention)
  {
    if (prompt.empty()) return {};

    vector<PromptAttentionFrame> results;
    auto stop = prompt.data() + prompt.size();

    const char* segment = nullptr;
    auto text = prompt.data();
    while (text <= stop)
    {
      //Add normal segment
      if (segment && (text == stop || *text == '('))
      {
        results.push_back({ string{ segment, text }, attention });
        segment = nullptr;
      }

      //Add attention segment
      if (*text == '(' && text != stop)
      {
        float childAttention;
        string_view childPrompt;
        ReadFrame(text, childAttention, childPrompt);

        auto childFrames = ParseAttentionFrames(childPrompt, childAttention);
        for (auto& childFrame : childFrames)
        {
          results.push_back({ childFrame.Text, attention * childFrame.Attention });
        }
      }
      //Flag unexpected brackets
      else if (*text == ')' && text != stop)
      {
        throw runtime_error("Unexpected closing bracket.");
      }
      //Build normal segments
      else
      {
        if (!segment) segment = text;
        text++;
      }
    }

    return CleanAttentionFrames(results);
  }
}