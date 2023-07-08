#include "pch.h"
#include "CppUnitTest.h"
#include "MachineLearning/DepthEstimator.h"
#include "MachineLearning/EdgeDetector.h"
#include "MachineLearning/PoseDetector.h"
#include "Storage/FileIO.h"

using namespace Axodox::Graphics;
using namespace Axodox::Storage;
using namespace Axodox::MachineLearning::Prompts;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace Axodox::MachineLearning::Test
{
  TEST_CLASS(FeatureExtractorTest)
  {
  private:
    inline static TextureData _imageTexture;

  public:
    TEST_CLASS_INITIALIZE(FeatureExtractorInitialize)
    {
      //Load input data
      auto imagePath = lib_folder() / "..\\..\\..\\inputs\\bedroom.png";
      auto imageData = read_file(imagePath);
      _imageTexture = TextureData::FromBuffer(imageData);
    }

    TEST_METHOD(TestDepthEstimation)
    {
      //Prepare input
      auto imageTexture = _imageTexture.Resize(512, 512);
      auto imageTensor = Tensor::FromTextureData(imageTexture, ColorNormalization::LinearZeroToOne);

      //Load model
      auto modelFolder = (lib_folder() / "..\\..\\..\\models").lexically_normal();

      //Run depth estimation
      OnnxEnvironment environment{ modelFolder };
      DepthEstimator depthEstimator{ environment };

      //Convert output to image
      auto result = depthEstimator.EstimateDepth(imageTensor);      
      DepthEstimator::NormalizeDepthTensor(result);

      auto depthTexture = result.ToTextureData(ColorNormalization::LinearZeroToOne);
      auto depthData = depthTexture[0].ToBuffer();      
      auto outputPath = lib_folder() / "depth.png";
      write_file(outputPath, depthData);
    }

    TEST_METHOD(TestCannyEdgeDetection)
    {
      //Prepare input
      auto imageTensor = Tensor::FromTextureData(_imageTexture, ColorNormalization::LinearZeroToOne);

      //Load model
      auto modelFolder = (lib_folder() / "..\\..\\..\\models").lexically_normal();

      //Run depth estimation
      OnnxEnvironment environment{ modelFolder };
      EdgeDetector edgeDetector{ environment, EdgeDetectionMode::Canny };

      //Convert output to image
      auto result = edgeDetector.DetectEdges(imageTensor);
      auto values = result.AsSpan<float>();

      auto edgeTexture = result.ToTextureData(ColorNormalization::LinearZeroToOne);
      auto edgeData = edgeTexture[0].ToBuffer();
      auto outputPath = lib_folder() / "canny.png";
      write_file(outputPath, edgeData);
    }

    TEST_METHOD(TestHedEdgeDetection)
    {
      //Prepare input
      auto imageTensor = Tensor::FromTextureData(_imageTexture, ColorNormalization::LinearZeroToOne);

      //Load model
      auto modelFolder = (lib_folder() / "..\\..\\..\\models").lexically_normal();

      //Run depth estimation
      OnnxEnvironment environment{ modelFolder };
      EdgeDetector edgeDetector{ environment, EdgeDetectionMode::Hed };

      //Convert output to image
      auto result = edgeDetector.DetectEdges(imageTensor);
      auto values = result.AsSpan<float>();

      auto edgeTexture = result.ToTextureData(ColorNormalization::LinearZeroToOne);
      auto edgeData = edgeTexture[0].ToBuffer();
      auto outputPath = lib_folder() / "hed.png";
      write_file(outputPath, edgeData);
    }

    TEST_METHOD(TestPoseDetection)
    {
      //Prepare input
      auto imagePath = lib_folder() / "..\\..\\..\\inputs\\football.jpg";
      auto imageData = read_file(imagePath);
      auto imageTexture = TextureData::FromBuffer(imageData);

      //Load model
      auto modelFolder = (lib_folder() / "..\\..\\..\\models").lexically_normal();

      //Run depth estimation
      OnnxEnvironment environment{ modelFolder };
      PoseDetector poseDetector{ environment };

      //Convert output to image
      poseDetector.ExtractFeatures(imageTexture);
    }
  };
}
