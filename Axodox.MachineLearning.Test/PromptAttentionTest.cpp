#include "pch.h"
#include "CppUnitTest.h"

using namespace Axodox::MachineLearning::Text::Prompts;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace Axodox::MachineLearning::Test
{
  TEST_CLASS(PromptAttentionTest)
  {
  public:
    TEST_METHOD(TestEmptyPrompt)
    {
      auto result = ParseAttentionFrames("");
      Assert::IsTrue(result == vector<PromptAttentionFrame>{});
    }

    TEST_METHOD(TestBasicPrompt)
    {
      auto result = ParseAttentionFrames("test");
      Assert::IsTrue(result == vector<PromptAttentionFrame>{
        { "test", 1.f }
      });
    }

    TEST_METHOD(TestEmptyBrackets)
    {
      auto result = ParseAttentionFrames("test()");
      Assert::IsTrue(result == vector<PromptAttentionFrame>{
        { "test", 1.f }
      });
    }

    TEST_METHOD(TestTightPrompt)
    {
      auto result = ParseAttentionFrames("test (this:0.2)");
      Assert::IsTrue(result == vector<PromptAttentionFrame>{
        { "test", 1.f },
        { "this", 0.2f }
      });
    }

    TEST_METHOD(TestSpacedPrompt)
    {
      auto result = ParseAttentionFrames("test( this : 0.2 )");
      Assert::IsTrue(result == vector<PromptAttentionFrame>{
        { "test", 1.f },
        { "this", 0.2f }
      });
    }

    TEST_METHOD(TestTwoPrompts)
    {
      auto result = ParseAttentionFrames("test (this : 0.8 ) at (once: 0.2)");
      Assert::IsTrue(result == vector<PromptAttentionFrame>{
        { "test", 1.f },
        { "this", 0.8f },
        { "at", 1.f },
        { "once", 0.2f }
      });
    }

    TEST_METHOD(TestTightPrompts)
    {
      auto result = ParseAttentionFrames("test (this at :0.5 )(once:0.2)");
      Assert::IsTrue(result == vector<PromptAttentionFrame>{
        { "test", 1.f },
        { "this at", 0.5f },
        { "once", 0.2f }
      });
    }

    TEST_METHOD(TestNestedPrompts)
    {
      auto result = ParseAttentionFrames("test (this (at:0.5) : 0.8) once");
      Assert::IsTrue(result == vector<PromptAttentionFrame>{
        { "test", 1.f },
        { "this", 0.8f },
        { "at", 0.4f },
        { "once", 1.f }
      });
    }

    TEST_METHOD(TestUnclosedBracket)
    {
      try
      {
        ParseAttentionFrames("test ( asd");
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
        ParseAttentionFrames("test ( asd))");
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
        ParseAttentionFrames("test ( asd:asd)");
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
        ParseAttentionFrames("test ( 0.1:0.2:0.3:0.4)");
        Assert::Fail();
      }
      catch (...)
      {
      }
    }
  };
}
