#include "pch.h"
#include "CppUnitTest.h"

using namespace Axodox::MachineLearning::Text::Prompts;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace Axodox::MachineLearning::Test
{
  TEST_CLASS(PromptSchedulerTest)
  {
  public:
    TEST_METHOD(TestEmptyPrompt)
    {
      auto result = ParseTimeFrames("");
      Assert::IsTrue(result == vector<PromptTimeFrame>{});
    }

    TEST_METHOD(TestBasicPrompt)
    {
      auto result = ParseTimeFrames("test");
      Assert::IsTrue(result == vector<PromptTimeFrame>{
        { "test", 0.f, 1.f }
      });
    }

    TEST_METHOD(TestEmptyBrackets)
    {
      auto result = ParseTimeFrames("test[]");
      Assert::IsTrue(result == vector<PromptTimeFrame>{
        { "test", 0.f, 1.f }
      });
    }

    TEST_METHOD(TestFromPrompt)
    {
      auto result = ParseTimeFrames("test[0.2<this]");
      Assert::IsTrue(result == vector<PromptTimeFrame>{
        { "test", 0.f, 1.f },
        { "this", 0.2f, 1.f }
      });
    }

    TEST_METHOD(TestToPrompt)
    {
      auto result = ParseTimeFrames("test [this < 0.8 ]");
      Assert::IsTrue(result == vector<PromptTimeFrame>{
        { "test ", 0.f, 1.f },
        { "this ", 0.f, 0.8f }
      });
    }

    TEST_METHOD(TestFromToPrompt)
    {
      auto result = ParseTimeFrames("test [0.2<this < 0.8 ]");
      Assert::IsTrue(result == vector<PromptTimeFrame>{
        { "test ", 0.f, 1.f },
        { "this ", 0.2f, 0.8f }
      });
    }

    TEST_METHOD(TestTwoPrompts)
    {
      auto result = ParseTimeFrames("test [0.2<this < 0.8 ] at [0.2<once]");
      Assert::IsTrue(result == vector<PromptTimeFrame>{
        { "test ", 0.f, 1.f },
        { "this ", 0.2f, 0.8f },
        { " at ", 0.f, 1.f },
        { "once", 0.2f, 1.f }
      });
    }

    TEST_METHOD(TestTightPrompts)
    {
      auto result = ParseTimeFrames("test [0.2<this at< 0.8 ][0.2<once]");
      Assert::IsTrue(result == vector<PromptTimeFrame>{
        { "test ", 0.f, 1.f },
        { "this at", 0.2f, 0.8f },
        { "once", 0.2f, 1.f }
      });
    }

    TEST_METHOD(TestNestedPrompts)
    {
      auto result = ParseTimeFrames("test [0.2<this [0.5<at]< 0.8 ][0.2<once]");
      Assert::IsTrue(result == vector<PromptTimeFrame>{
        { "test ", 0.f, 1.f },
        { "this ", 0.2f, 0.8f },
        { "at", 0.5f, 0.8f },
        { "once", 0.2f, 1.f }
      });
    }

    TEST_METHOD(TestUnclosedBracket)
    {
      try
      {
        ParseTimeFrames("test [ asd");
        Assert::Fail();
      }
      catch (...)
      {
      }
    }

    TEST_METHOD(TestUnexpectedClosedBracket)
    {
      try
      {
        ParseTimeFrames("test [ asd]]");
        Assert::Fail();
      }
      catch (...)
      {
      }
    }

    TEST_METHOD(TestBadSegments)
    {
      try
      {
        ParseTimeFrames("test [ asd<asd]");
        Assert::Fail();
      }
      catch (...)
      {
      }
    }

    TEST_METHOD(TestTooManySegments)
    {
      try
      {
        ParseTimeFrames("test [ 0.1<0.2<0.3<0.4]");
        Assert::Fail();
      }
      catch (...)
      {
      }
    }
  };
}
