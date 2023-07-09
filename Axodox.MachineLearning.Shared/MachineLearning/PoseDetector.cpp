#include "pch.h"
#include "PoseDetector.h"
#include "Openpose/Openpose.h"
#include "MachineLearning/Munkres/CostGraph.h"
#include "MachineLearning/Munkres/PairGraph.h"
#include "MachineLearning/Munkres/Munkres2.h"

using namespace Axodox::Graphics;
using namespace Axodox::MachineLearning;
using namespace Axodox::MachineLearning::Munkres;
using namespace D2D1;
using namespace DirectX;
using namespace Ort;
using namespace std;
using namespace winrt;

namespace {
  const size_t PoseMaxBodyCount = 100;
  const size_t PoseMinJointCount = 2;
  const float PoseConfidenceMapThreshold = 0.1f;
  const float BoneAffinityThreshold = 0.1f;
  const int PoseJointSearchRadius = 5;
  const int PoseJointRefineRadius = 2;
  const int PoseBoneAffinityIntegralSamples = 7;

  typedef std::array<std::vector<DirectX::XMFLOAT2>, PoseJointCount> PoseJointPositionCandidates;

  struct PoseBoneJointMapping
  {
    size_t BoneA, BoneB, JointA, JointB;
  };

  const PoseBoneJointMapping PoseBones[] = {
    { 0, 1, 15, 13 },
    { 2, 3, 13, 11 },
    { 4, 5, 16, 14 },
    { 6, 7, 14, 12 },
    { 8, 9, 11, 12 },
    { 10, 11, 5, 7 },
    { 12, 13, 6, 8 },
    { 14, 15, 7, 9 },
    { 16, 17, 8, 10 },
    { 18, 19, 1, 2 },
    { 20, 21, 0, 1 },
    { 22, 23, 0, 2 },
    { 24, 25, 1, 3 },
    { 26, 27, 2, 4 },
    { 28, 29, 3, 5 },
    { 30, 31, 4, 6 },
    { 32, 33, 17, 0 },
    { 34, 35, 17, 5 },
    { 36, 37, 17, 6 },
    { 38, 39, 17, 11 },
    { 40, 41, 17, 12 }
  };

  std::vector<PoseJointPositionCandidates> EstimateJointLocations(const Tensor& jointPositionConfidenceMap)
  {
    //The input tensor is laid out the following way: (batch, joint, height, width)
    if (jointPositionConfidenceMap.Shape[1] != PoseJointCount) throw logic_error("The number of channels in the confidence map is invalid.");

    auto width = int(jointPositionConfidenceMap.Shape[3]);
    auto height = int(jointPositionConfidenceMap.Shape[2]);

    vector<PoseJointPositionCandidates> results;
    results.reserve(jointPositionConfidenceMap.Shape[0]);

    for (size_t batch = 0; batch < jointPositionConfidenceMap.Shape[0]; batch++)
    {
      PoseJointPositionCandidates joints;

      for (size_t joint = 0; joint < jointPositionConfidenceMap.Shape[1]; joint++)
      {
        auto& jointPositions = joints[joint];

        auto value = jointPositionConfidenceMap.AsPointer<float>(batch, joint);
        for (auto y = 0; y < height; y++)
        {
          for (auto x = 0; x < width; x++, value++)
          {
            //When we find a value above the threshold
            if (*value < PoseConfidenceMapThreshold) continue;

            //We check the area around it
            auto xMin = max(0, x - PoseJointSearchRadius);
            auto yMin = max(0, y - PoseJointSearchRadius);
            auto xMax = min(width, x + PoseJointSearchRadius);
            auto yMax = min(height, y + PoseJointSearchRadius);

            auto isPeak = true;
            auto peakValue = *value;
            for (auto yWindow = yMin; isPeak && yWindow < yMax; yWindow++)
            {
              auto neighbour = jointPositionConfidenceMap.AsPointer<float>(batch, joint, yWindow, xMin);
              for (auto xWindow = xMin; isPeak && xWindow < xMax; xWindow++, neighbour++)
              {
                if (*neighbour > peakValue)
                {
                  isPeak = false;
                }
              }
            }

            //And if its not the largest we return it as a peak
            if (!isPeak) continue;

            //Then calculate the precise location
            auto weightSum = 0.f;
            XMFLOAT2 refinedLocation{ 0.f, 0.f };

            xMin = max(0, x - PoseJointRefineRadius);
            yMin = max(0, y - PoseJointRefineRadius);
            xMax = min(width, x + PoseJointRefineRadius + 1);
            yMax = min(height, y + PoseJointRefineRadius + 1);

            for (auto yWindow = yMin; isPeak && yWindow < yMax; yWindow++)
            {
              auto neighbour = jointPositionConfidenceMap.AsPointer<float>(batch, joint, yWindow);
              for (auto xWindow = xMin; isPeak && xWindow < xMax; xWindow++, neighbour++)
              {
                auto weight = *neighbour;
                refinedLocation.x += xWindow * weight;
                refinedLocation.y += yWindow * weight;
                weightSum += weight;
              }
            }

            refinedLocation.x /= weightSum;
            refinedLocation.y /= weightSum;
            jointPositions.push_back(refinedLocation);
          }
        }
      }
      results.push_back(joints);
    }

    return results;
  }

  Tensor CalculateBoneAffinityScores(const Tensor& boneAffinityMap, const std::vector<PoseJointPositionCandidates>& jointConfigurationCollection)
  {
    //The input tensor is laid out the following way: (batch, bone, height, width)
    if (boneAffinityMap.Shape[1] != 2 * size(PoseBones)) throw logic_error("The number of channels in the bone affinity map is invalid.");

    auto batchCount = boneAffinityMap.Shape[0];
    auto width = int(boneAffinityMap.Shape[3]);
    auto height = int(boneAffinityMap.Shape[2]);

    //The output tensor describes the bone affinities for each joint position
    Tensor result{ TensorType::Single, batchCount, size(PoseBones), PoseMaxBodyCount, PoseMaxBodyCount };

    for (size_t batch = 0; batch < batchCount; batch++)
    {
      auto& jointConfiguration = jointConfigurationCollection[batch];

      for (size_t bone = 0; bone < size(PoseBones); bone++)
      {
        auto graph = result.AsPointer<float>(batch, bone);
        auto boneJointMapping = PoseBones[bone];

        auto& jointPositionsA = jointConfiguration[boneJointMapping.JointA];
        auto& jointPositionsB = jointConfiguration[boneJointMapping.JointB];

        auto boneAffinityA = boneAffinityMap.AsPointer<float>(batch, boneJointMapping.BoneA);
        auto boneAffinityB = boneAffinityMap.AsPointer<float>(batch, boneJointMapping.BoneB);

        for (auto a = 0; a < jointPositionsA.size(); a++)
        {
          auto jointPositionA = XMLoadFloat2(&jointPositionsA[a]);
          for (auto b = 0; b < jointPositionsB.size(); b++)
          {
            auto jointPositionB = XMLoadFloat2(&jointPositionsB[b]);

            auto jointDistanceVector = jointPositionB - jointPositionA;
            auto normalizedJointDistanceVector = XMVector2Normalize(jointDistanceVector);

            auto integral = 0.f;
            auto increment = 1.f / (PoseBoneAffinityIntegralSamples - 1);
            auto progress = 0.f;
            for (auto t = 0; t < PoseBoneAffinityIntegralSamples; t++)
            {
              XMINT2 pixel;
              XMStoreSInt2(&pixel, jointPositionA + progress * jointDistanceVector);

              if (pixel.x < 0 || pixel.y < 0 || pixel.x >= width || pixel.y >= height) continue;

              auto boneAffinity = XMVectorSet(boneAffinityB[pixel.y * width + pixel.x], boneAffinityA[pixel.y * width + pixel.x], 0, 0);

              auto boneAffinityScore = XMVectorGetX(XMVector2Dot(boneAffinity, normalizedJointDistanceVector));
              integral += boneAffinityScore;

              progress += increment;
            }

            integral /= PoseBoneAffinityIntegralSamples;
            graph[a * PoseMaxBodyCount + b] = integral;
          }
        }
      }
    }

    return result;
  }

  Tensor CalculateBoneAssignments(const Tensor& boneAffinityScores, const std::vector<PoseJointPositionCandidates>& jointConfigurationCollection)
  {
    auto batchCount = boneAffinityScores.Shape[0];

    Tensor result{ TensorType::Int32, batchCount, size(PoseBones), 2, PoseMaxBodyCount };
    ranges::fill(result.AsSpan<int32_t>(), -1);

    for (size_t batch = 0; batch < batchCount; batch++)
    {
      for (size_t bone = 0; bone < size(PoseBones); bone++)
      {
        auto connections = result.AsSubSpan<int32_t>(batch, bone);
        auto boneJointMapping = PoseBones[bone];

        auto rowCount = jointConfigurationCollection[batch][boneJointMapping.JointA].size();
        auto columnCount = jointConfigurationCollection[batch][boneJointMapping.JointB].size();

        CostGraph costGraph{ rowCount, columnCount };
        for (size_t row = 0; row < rowCount; row++)
        {
          auto boneAffinityScore = boneAffinityScores.AsPointer<float>(batch, bone, row);
          for (size_t column = 0; column < columnCount; column++)
          {
            costGraph.At(row, column) = -*boneAffinityScore++;
          }
        }

        PairGraph starGraph{ rowCount, columnCount };
        SolveMunkres(costGraph, starGraph);

        for (size_t row = 0; row < rowCount; row++)
        {
          auto boneAffinityScore = boneAffinityScores.AsPointer<float>(batch, bone, row);
          for (size_t column = 0; column < columnCount; column++)
          {
            if (!starGraph.IsPair(row, column) || *boneAffinityScore++ < BoneAffinityThreshold) continue;

            connections[row] = int32_t(column);
            connections[PoseMaxBodyCount + column] = int32_t(row);
          }
        }
      }
    }

    return result;
  }

  std::vector<std::vector<PoseJointPositions>> AssembleBodies(const Tensor& connectionMap, const std::vector<PoseJointPositionCandidates>& jointConfigurationCollection)
  {
    auto batchCount = connectionMap.Shape[0];
    vector<vector<PoseJointPositions>> results;
    results.reserve(jointConfigurationCollection.size());

    for (size_t batch = 0; batch < batchCount; batch++)
    {
      vector<bool> visitations;
      visitations.resize(PoseJointCount * PoseMaxBodyCount);

      vector<PoseJointPositions> bodies;

      auto objectCount = 0;
      for (size_t joint = 0; joint < PoseJointCount && objectCount < PoseMaxBodyCount; joint++)
      {
        auto positionCandidateCount = jointConfigurationCollection[batch][joint].size();

        for (size_t positionCandidate = 0; positionCandidate < positionCandidateCount && objectCount < PoseMaxBodyCount; positionCandidate++)
        {
          queue<pair<size_t, size_t>> candidateQueue;
          candidateQueue.push({ joint, positionCandidate });

          PoseJointPositions bodyJointPositions;
          ranges::fill(bodyJointPositions, XMFLOAT2{ NAN, NAN });

          auto locatedJointCount = 0;

          while (!candidateQueue.empty())
          {
            auto [currentJoint, currentPositionCandidate] = candidateQueue.front();
            candidateQueue.pop();

            auto visitationIndex = currentJoint * PoseMaxBodyCount + currentPositionCandidate;
            if (visitations[visitationIndex]) continue;

            visitations[visitationIndex] = true;

            bodyJointPositions[currentJoint] = jointConfigurationCollection[batch][currentJoint][currentPositionCandidate];
            locatedJointCount++;

            for (size_t bone = 0; bone < size(PoseBones); bone++)
            {
              auto boneJointMapping = PoseBones[bone];
              auto connections = connectionMap.AsSubSpan<int32_t>(batch, bone);

              if (boneJointMapping.JointA == currentJoint)
              {
                auto connection = connections[currentPositionCandidate];
                if (connection >= 0) candidateQueue.push({ boneJointMapping.JointB, connection });
              }

              if (boneJointMapping.JointB == currentJoint)
              {
                auto connection = connections[PoseMaxBodyCount + currentPositionCandidate];
                if (connection >= 0) candidateQueue.push({ boneJointMapping.JointA, connection });
              }
            }
          }

          if (locatedJointCount > PoseMinJointCount)
          {
            bodies.push_back(bodyJointPositions);
          }
        }
      }

      results.push_back(bodies);
    }

    return results;
  }

  std::vector<std::vector<PoseJointPositions>> ExtractSkeletons(const Tensor& jointPositionConfidenceMap, const Tensor& boneAffinityMap)
  {
    auto jointConfigurationCollection = EstimateJointLocations(jointPositionConfidenceMap);
    auto boneAffinityScores = CalculateBoneAffinityScores(boneAffinityMap, jointConfigurationCollection);
    auto boneAssignments = CalculateBoneAssignments(boneAffinityScores, jointConfigurationCollection);
    auto results = AssembleBodies(boneAssignments, jointConfigurationCollection);

    auto width = float(boneAffinityMap.Shape[3] - 1);
    auto height = float(boneAffinityMap.Shape[2] - 1);
    for (auto& frame : results)
    {
      for (auto& body : frame)
      {
        for (auto& joint : body)
        {
          joint.x /= width;
          joint.y /= height;
        }
      }
    }

    return results;
  }

  TextureData VisualizeBodies(std::span<const PoseJointPositions> bodies, uint32_t width, uint32_t height)
  {
    GraphicsDevice device{};
    DrawingController drawing{ device };

    Texture2DDefinition textureDefinition{ width, height, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, Texture2DFlags::None };
    DrawingTarget2D target{ drawing, textureDefinition };
    StagingTexture2D stage{ device, textureDefinition };

    auto factory = drawing.DrawFactory();
    auto context = drawing.DrawingContext();

    com_ptr<ID2D1SolidColorBrush> brush;
    check_hresult(context->CreateSolidColorBrush(D2D1::ColorF(1.f, 0.f, 0.f), brush.put()));

    context->BeginDraw();
    context->SetTarget(target.Bitmap());

    for (auto& body : bodies)
    {
      for (auto bone : PoseBones)
      {
        auto jointA = body[bone.JointA];
        auto jointB = body[bone.JointB];

        if (isnan(jointA.x) || isnan(jointA.y) || isnan(jointB.x) || isnan(jointB.y)) continue;

        auto positionA = Point2F(jointA.x * width, jointA.y * height);
        auto positionB = Point2F(jointB.x * width, jointB.y * height);
        context->DrawLine(positionA, positionB, brush.get());
      }
    }

    check_hresult(context->EndDraw());

    target.Copy(&stage);
    return stage.Download();
  }
}

namespace Axodox::MachineLearning
{
  PoseDetector::PoseDetector(OnnxEnvironment& environment, std::optional<ModelSource> source) :
    _environment(environment),
    _session(environment->CreateSession(source ? *source : (_environment.RootPath() / L"annotators/openpose.onnx")))
  { }

  std::vector<std::vector<PoseJointPositions>> PoseDetector::EstimatePose(const Tensor & image)
  {
    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("input", image.ToHalf().ToOrtValue());
    bindings.BindOutput("cmap", _environment->MemoryInfo());
    bindings.BindOutput("paf", _environment->MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto jointPositionConfidenceMap = Tensor::FromOrtValue(outputValues[0]).ToSingle();
    auto boneAffinityMap = Tensor::FromOrtValue(outputValues[1]).ToSingle();

    //Extract skeletons
    auto skeletons = ExtractSkeletons(jointPositionConfidenceMap, boneAffinityMap);
    return skeletons;
  }

  Graphics::TextureData PoseDetector::DetectPose(const Tensor & image)
  {
    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("input", image.ToHalf().ToOrtValue());
    bindings.BindOutput("cmap", _environment->MemoryInfo());
    bindings.BindOutput("paf", _environment->MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto cmap = Tensor::FromOrtValue(outputValues[0]).ToSingle();
    auto paf = Tensor::FromOrtValue(outputValues[1]).ToSingle();

    auto frame = move(image.ToTextureData(ColorNormalization::LinearZeroToOne).front());

    Openpose op{ cmap.Shape };
    op.detect(cmap.AsPointer<float>(), paf.AsPointer<float>(), frame);

    return frame;
  }

  Graphics::TextureData PoseDetector::ExtractFeatures(const Graphics::TextureData& value)
  {
    auto inputTensor = Tensor::FromTextureData(value.Resize(224, 224), ColorNormalization::LinearZeroToOne);

    auto skeletons = EstimatePose(inputTensor);
    return VisualizeBodies(skeletons[0], value.Width, value.Height);
    //auto outputImage = DetectPose(inputTensor).Resize(value.Width, value.Height);
  }
}