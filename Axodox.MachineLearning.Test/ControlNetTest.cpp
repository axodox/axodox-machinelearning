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
  TEST_CLASS(ControlNetTest)
  {
    TEST_METHOD(TestControlNet)
    {
      auto modelFolder = (lib_folder() / "..\\..\\..\\models").lexically_normal();
      OnnxEnvironment environment(modelFolder / "stable_diffusion");

      ControlNetOptions options{};

      //Create text embedding
      {
        TextEmbedder textEmbedder(environment);
        //auto positiveEmbedding = textEmbedder.ProcessPrompt("a clean bedroom");
        auto positiveEmbedding = textEmbedder.ProcessPrompt("jpop singer on stage, best quality, extremely detailed");
        //auto negativeEmbedding = textEmbedder.ProcessPrompt("blurry, render");
        auto negativeEmbedding = textEmbedder.ProcessPrompt("monochrome, lowres, bad anatomy, worst quality, low quality");

        options.TextEmbeddings = negativeEmbedding.Concat(positiveEmbedding);
      }
      
      //Load conditioning input
      {
        auto imagePath = (lib_folder() / "..\\..\\..\\inputs\\canny.png").lexically_normal();
        auto imageData = read_file(imagePath);
        auto imageTexture = TextureData::FromBuffer(imageData);
        
        options.ConditionInput = Tensor::FromTextureData(imageTexture, ColorNormalization::LinearZeroToOne);
        options.ConditionType = ControlNetType::Canny;
      }

      //Run ControlNet
      Tensor image;
      {
        auto controlnetFolder = (lib_folder() / "..\\..\\..\\models\\controlnet").lexically_normal();
        ControlNetInferer controlNet{ environment, controlnetFolder };

        image = controlNet.RunInference(options);
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
        write_file(lib_folder() / "controlnet.png", imageBuffer);
      }
    }
  };
}
