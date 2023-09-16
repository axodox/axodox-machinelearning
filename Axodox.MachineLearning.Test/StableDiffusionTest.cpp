#include "pch.h"
#include "CppUnitTest.h"
#include "Storage/FileIO.h"

using namespace Axodox::Graphics;
using namespace Axodox::Storage;
using namespace Axodox::MachineLearning;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace Axodox::MachineLearning::Test
{
  TEST_CLASS(StableDiffusionTest)
  {
    TEST_METHOD(TestStableDiffusion)
    {
      auto modelFolder = (lib_folder() / "..\\..\\..\\models").lexically_normal();
      OnnxEnvironment environment(modelFolder / "stable_diffusion");

      StableDiffusionOptions options{};

      //Create text embedding
      {
        TextEmbedder textEmbedder(environment);
        auto positiveEmbedding = textEmbedder.ProcessPrompt("a clean bedroom");
        auto negativeEmbedding = textEmbedder.ProcessPrompt("blurry, render");

        options.TextEmbeddings.Tensor = negativeEmbedding.Concat(positiveEmbedding);
        options.TextEmbeddings.Weights = { -1.f, 1.f };
      }

      //Run StableDiffusion
      Tensor image;
      {
        StableDiffusionInferer uNet{ environment };

        image = uNet.RunInference(options);
      }

      //Decode VAE
      {
        VaeDecoder vaeDecoder{ environment };

        image = vaeDecoder.DecodeVae(image);
      }

      //Save result
      {
        auto imageTexture = image.ToTextureData(ColorNormalization::LinearPlusMinusOne);
        auto imageBuffer = imageTexture[0].ToBuffer();
        write_file(lib_folder() / "stablediffusion.png", imageBuffer);
      }
    }
  };
}
