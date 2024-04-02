#include "pch.h"
#include "CppUnitTest.h"
#include "Storage/FileIO.h"

using namespace Axodox::Graphics;
using namespace Axodox::Storage;
using namespace Axodox::MachineLearning::Sessions;
using namespace Axodox::MachineLearning::Imaging::StableDiffusion;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace Axodox::MachineLearning::Test
{
  TEST_CLASS(StableDiffusionTest)
  {
    TEST_METHOD(TestStableDiffusion)
    {
      StableDiffusionDirectorySessionParameters sessionParameters{ lib_folder() / "../../../models/stable_diffusion" };

      StableDiffusionOptions options{};

      //Create text embedding
      {
        TextEmbedder textEmbedder{ sessionParameters };
        auto positiveEmbedding = textEmbedder.ProcessPrompt("a clean bedroom");
        auto negativeEmbedding = textEmbedder.ProcessPrompt("blurry, render");

        options.TextEmbeddings.Tensor = negativeEmbedding.Concat(positiveEmbedding);
        options.TextEmbeddings.Weights = { -1.f, 1.f };
      }

      //Run StableDiffusion
      Tensor image;
      {
        StableDiffusionInferer uNet{ sessionParameters };

        image = uNet.RunInference(options);
      }

      //Decode VAE
      {
        VaeDecoder vaeDecoder{ sessionParameters };

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
