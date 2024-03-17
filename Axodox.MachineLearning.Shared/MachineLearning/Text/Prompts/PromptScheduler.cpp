#include "pch.h"
#include "PromptParser.h"
#include "PromptScheduler.h"

using namespace Axodox::MachineLearning::Text::Prompts;
using namespace std;

namespace 
{
  void ReadFrame(const char*& text, float& start, float& end, std::string_view& prompt)
  {
    //Split up to segments
    auto segments = SplitToSegments(text, '[', '<', ']');

    //Interpret segments
    string_view startSegment, promptSegment, endSegment;
    switch (segments.size())
    {
    case 0: // []
      //Do nothing
      break;
    case 1: // [prompt]
      promptSegment = segments[0];
      break;
    case 2: // [0.2 < prompt] or [ prompt < 0.8]
      if (TryParseNumber(segments[0]))
      {
        startSegment = segments[0];
        promptSegment = segments[1];
      }
      else
      {
        promptSegment = segments[0];
        endSegment = segments[1];
      }
      break;
    case 3: // [0.2 < prompt < 0.8]
      startSegment = segments[0];
      promptSegment = segments[1];
      endSegment = segments[2];
      break;
    default:
      throw runtime_error("Too many segments in a prompt scheduling frame.");
    }

    //Parse segments
    auto startValue = TryParseNumber(startSegment);
    if (!startSegment.empty() && !startValue) throw runtime_error("Could not parse scheduling frame start.");
    start = startValue ? *startValue : 0.f;

    auto endValue = TryParseNumber(endSegment);
    if (!endSegment.empty() && !endValue) throw runtime_error("Could not parse scheduling frame end.");
    end = endValue ? *endValue : 1.f;

    prompt = promptSegment;
  }

  std::vector<PromptTimeFrame> ConvertTextFramesToTimeFrames(std::span<const PromptTimeFrame> frames)
  {
    //Collect key frames
    set<float> keyFrames;
    keyFrames.emplace(0.f);
    keyFrames.emplace(1.f);

    for (auto& frame : frames)
    {
      keyFrames.emplace(frame.Start);
      keyFrames.emplace(frame.End);
    }

    //Build key frames
    vector<PromptTimeFrame> results;
    for (auto it = keyFrames.begin(); it != keyFrames.end(); it++)
    {
      auto keyFrame = *it;
      if (keyFrame == 1.f) break;

      PromptTimeFrame timeFrame;

      //Collect prompt
      for (auto& frame : frames)
      {
        if (keyFrame < frame.Start || keyFrame >= frame.End) continue;

        timeFrame.Text += frame.Text;
      }

      //Collect start and end
      timeFrame.Start = keyFrame;
      timeFrame.End = *next(it);

      results.push_back(timeFrame);
    }

    return results;
  }
}

namespace Axodox::MachineLearning::Text::Prompts
{
  std::vector<PromptTimeFrame> ParseTimeFrames(std::string_view prompt, float start, float end)
  {
    if (prompt.empty()) return {};

    vector<PromptTimeFrame> results;
    auto stop = prompt.data() + prompt.size();

    const char* segment = nullptr;
    auto text = prompt.data();
    while (text <= stop)
    {
      //Add normal segment
      if (segment && (text == stop || *text == '['))
      {
        results.push_back({ string{ segment, text }, start, end });
        segment = nullptr;
      }

      //Add scheduled segment
      if (*text == '[' && text != stop)
      {
        //Read frame
        float childStart, childEnd;
        string_view childPrompt;
        ReadFrame(text, childStart, childEnd, childPrompt);

        auto childFrames = ParseTimeFrames(childPrompt, childStart, childEnd);
        for (auto& childFrame : childFrames)
        {
          results.push_back({ childFrame.Text, lerp(start, end, childFrame.Start), lerp(start, end, childFrame.End) });
        }
      }
      //Flag unexpected brackets
      else if (*text == ']' && text != stop)
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

    return results;
  }

  std::vector<PromptTimeFrame> SchedulePrompt(std::string_view prompt)
  {
    auto textFrames = ParseTimeFrames(prompt);
    return ConvertTextFramesToTimeFrames(textFrames);
  }

  std::vector<std::string> SchedulePrompt(std::string_view prompt, uint32_t steps)
  {
    auto timeFrames = SchedulePrompt(prompt);

    vector<string> results;
    for (uint32_t i = 0; i < steps; i++)
    {
      //Scale time from 0 to 1
      auto t = i / float(steps - 1);

      //Find matching frame
      bool foundMatch = false;
      for (auto& frame : timeFrames)
      {
        if (t < frame.Start || t >= frame.End) continue;

        results.push_back(frame.Text);
        foundMatch = true;
      }

      //If not found use the last frame
      if (!foundMatch)
      {
        results.push_back(timeFrames.back().Text);
      }
    }

    return results;
  }
}