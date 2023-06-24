#include "pch.h"
#include "CppUnitTest.h"
#include "MachineLearning/DepthEstimator.h"
#include "Storage/FileIO.h"

using namespace Axodox::Graphics;
using namespace Axodox::Storage;
using namespace Axodox::MachineLearning::Prompts;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace Axodox::MachineLearning::Test
{
  TEST_CLASS(DepthEstimationTest)
  {
  public:
    TEST_METHOD(TestDepthEstimation)
    {
      //Load input data
      auto imagePath = lib_folder() / "..\\..\\..\\inputs\\bedroom.png";
      auto imageData = read_file(imagePath);
      auto imageTexture = TextureData::FromBuffer(imageData).Resize(512, 512);
      auto imageTensor = Tensor::FromTextureData(imageTexture, ColorNormalization::LinearZeroToOne);

      //Load model
      auto modelFolder = (lib_folder() / "..\\..\\..\\models").lexically_normal();

      //Run depth estimation
      OnnxEnvironment environment{ modelFolder };
      DepthEstimator depthEstimator{ environment };

      //Convert output to image
      auto result = depthEstimator.EstimateDepth(imageTensor);
      auto values = result.AsSpan<float>();

      //float min = 3000, max = 5000;
      float min = INFINITY, max = -INFINITY;
      for (auto value : values)
      {
        if (min > value) min = value;
        if (max < value) max = value;
      }
      
      auto range = max - min;
      for (auto& value : values)
      {
        value = (value - min) / range;
      }

      auto depthTexture = result.ToTextureData(ColorNormalization::LinearZeroToOne);
      auto depthData = depthTexture[0].ToBuffer();      
      auto outputPath = lib_folder() / "depth.png";
      write_file(outputPath, depthData);

      printf("asd");
    }
  };
}
