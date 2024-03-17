#include "pch.h"
#include "CppUnitTest.h"
#include "Storage/FileIO.h"

using namespace Axodox::Graphics;
using namespace Axodox::Storage;
using namespace Axodox::MachineLearning;
using namespace Axodox::MachineLearning::Imaging::StableDiffusion;
using namespace Axodox::MachineLearning::Sessions;
using namespace Microsoft::VisualStudio::CppUnitTestFramework; 
using namespace std;

namespace Axodox::MachineLearning::Test
{
  TEST_CLASS(ControlNetTest)
  {
    TEST_METHOD(TestControlNet)
    {
      StableDiffusionDirectorySessionParameters sessionParameters{ lib_folder() / "../../../models/stable_diffusion" };

      ControlNetOptions options{};

      //Create text embedding
      {
        TextEmbedder textEmbedder(sessionParameters);
        auto positiveEmbedding = textEmbedder.ProcessPrompt("a clean bedroom");
        auto negativeEmbedding = textEmbedder.ProcessPrompt("blurry, render");

        options.TextEmbeddings.Tensor = negativeEmbedding.Concat(positiveEmbedding);
        options.TextEmbeddings.Weights = { -1.f, 1.f };
      }
      
      //Load conditioning input
      {
        auto imagePath = (lib_folder() / "..\\..\\..\\inputs\\depth.png").lexically_normal();
        auto imageData = read_file(imagePath);
        auto imageTexture = TextureData::FromBuffer(imageData);
        
        options.ConditionInput = Tensor::FromTextureData(imageTexture, ColorNormalization::LinearZeroToOne);
      }

      //Run ControlNet
      Tensor image;
      {
        auto controlnetParameters = OnnxSessionParameters::Create(lib_folder() / "../../../models/controlnet", OnnxExecutorType::Dml);
        ControlNetInferer controlNet{ controlnetParameters, sessionParameters };

        image = controlNet.RunInference(options);
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
        write_file(lib_folder() / "controlnet.png", imageBuffer);
      }
    }
  };
}
