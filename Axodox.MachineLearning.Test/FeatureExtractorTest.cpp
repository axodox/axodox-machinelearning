#include "pch.h"
#include "CppUnitTest.h"
#include "MachineLearning/Imaging/Annotators/DepthEstimator.h"
#include "MachineLearning/Imaging/Annotators/EdgeDetector.h"
#include "MachineLearning/Imaging/Annotators/PoseEstimator.h"
#include "Storage/FileIO.h"

using namespace Axodox::Graphics;
using namespace Axodox::Storage;
using namespace Axodox::MachineLearning::Sessions;
using namespace Axodox::MachineLearning::Imaging::Annotators;
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
      auto modelPath = lib_folder() / "../../../models/annotators/depth.onnx";
      DepthEstimator depthEstimator{ OnnxSessionParameters::Create(modelPath, OnnxExecutorType::Dml) };

      //Run depth estimation
      auto result = depthEstimator.EstimateDepth(imageTensor);

      //Convert output to image
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
      auto modelPath = lib_folder() / "../../../models/annotators/canny.onnx";
      EdgeDetector edgeDetector{ OnnxSessionParameters::Create(modelPath, OnnxExecutorType::Dml) };

      //Run depth estimation
      auto result = edgeDetector.DetectEdges(imageTensor);

      //Convert output to image
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
      auto modelPath = lib_folder() / "../../../models/annotators/hed.onnx";
      EdgeDetector edgeDetector{ OnnxSessionParameters::Create(modelPath, OnnxExecutorType::Dml) };

      //Run depth estimation
      auto result = edgeDetector.DetectEdges(imageTensor);

      //Convert output to image
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
      auto modelPath = lib_folder() / "../../../models/annotators/openpose.onnx";
      PoseEstimator poseDetector{ OnnxSessionParameters::Create(modelPath, OnnxExecutorType::Dml) };

      //Run depth estimation
      auto poseTexture = poseDetector.ExtractFeatures(imageTexture);

      //Convert output to image
      auto edgeData = poseTexture.ToBuffer();
      auto outputPath = lib_folder() / "pose.png";
      write_file(outputPath, edgeData);
    }
  };
}
