#include "pch.h"
#include "OpenposeHelper.h"
#include "MachineLearning/Munkres/CostGraph.h"
#include "MachineLearning/Munkres/PairGraph.h"
#include "MachineLearning/Munkres/Munkres2.h"

using namespace Axodox::MachineLearning::Munkres;
using namespace DirectX;
using namespace std;

namespace Axodox::MachineLearning
{
  const float PoseConfidenceMapThreshold = 0.1f;
  const int PoseJointSearchRadius = 5;
  const int PosePartAffinityIntegralSamples = 7;

  std::vector<PoseJointPositionCandidates> FindPeaks(const Tensor& jointPositionConfidenceMap)
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
            auto xMax = min(0, x + PoseJointSearchRadius);
            auto yMax = min(0, y + PoseJointSearchRadius);

            auto isPeak = true;
            auto peakValue = *value;
            for (auto yWindow = yMin; isPeak && yWindow < yMax; yWindow++)
            {
              auto neighbour = jointPositionConfidenceMap.AsPointer<float>(batch, joint, yWindow);
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

        results.push_back(joints);
      }
    }

    return results;
  }

  Tensor CalculatePartAffinityScores(const Tensor& partAffinityMap, const std::vector<PoseJointPositionCandidates>& jointConfigurations)
  {
    //The input tensor is laid out the following way: (batch, bone, height, width)
    if (partAffinityMap.Shape[1] != 2 * size(PoseBones)) throw logic_error("The number of channels in the part affinity map is invalid.");

    auto batchCount = partAffinityMap.Shape[0];
    auto width = int(partAffinityMap.Shape[3]);
    auto height = int(partAffinityMap.Shape[2]);

    //The output tensor describes the bone affinities for each joint position
    Tensor result{ TensorType::Single, batchCount, size(PoseBones), PoseMaxBodyCount, PoseMaxBodyCount };

    for (size_t batch = 0; batch < batchCount; batch++)
    {
      auto& jointConfiguration = jointConfigurations[batch];

      for (size_t bone = 0; bone < size(PoseBones); bone++)
      {
        auto graph = result.AsPointer<float>(batch, bone);
        auto boneJointMapping = PoseBones[bone];

        auto& jointPositionsA = jointConfiguration[boneJointMapping.ConfidenceA];
        auto& jointPositionsB = jointConfiguration[boneJointMapping.ConfidenceB];

        auto partAffinityA = partAffinityMap.AsPointer<float>(batch, boneJointMapping.PartAffinityA);
        auto partAffinityB = partAffinityMap.AsPointer<float>(batch, boneJointMapping.PartAffinityB);

        for (auto a = 0; a < jointPositionsA.size(); a++)
        {
          auto jointPositionA = XMLoadFloat2(&jointPositionsA[a]);
          for (auto b = 0; b < jointPositionsB.size(); b++)
          {
            auto jointPositionB = XMLoadFloat2(&jointPositionsB[b]);

            auto jointDistanceVector = jointPositionA - jointPositionB;
            auto normalizedJointDistanceVector = XMVector2Normalize(jointDistanceVector);

            auto integral = 0.f;
            auto increment = 1.f / (PosePartAffinityIntegralSamples - 1);
            auto progress = 0.f;
            for (auto t = 0; t < PosePartAffinityIntegralSamples; t++)
            {
              XMINT2 pixel;
              XMStoreSInt2(&pixel, XMConvertVectorFloatToInt(jointPositionA + progress * jointDistanceVector, 0));

              if (pixel.x < 0 || pixel.y < 0 || pixel.x >= width || pixel.y >= height) continue;

              auto partAffinity = XMVectorSet(partAffinityA[pixel.y * width + pixel.x], partAffinityB[pixel.y * width + pixel.x], 0, 0);

              auto partAffinityScore = XMVectorGetX(XMVector2Dot(partAffinity, normalizedJointDistanceVector));
              integral += partAffinityScore;

              progress += increment;
            }

            integral /= PosePartAffinityIntegralSamples;
            graph[a * PoseMaxBodyCount + b] = integral;
          }
        }
      }
    }

    return result;
  }

  Tensor CalculatePartAssignments(const Tensor& partAffinityScores, const std::vector<PoseJointPositionCandidates>& jointConfiguration)
  {
    auto batchCount = partAffinityScores.Shape[0];

    Tensor result{ TensorType::Int32, batchCount, size(PoseBones), 2, PoseMaxBodyCount };
    ranges::fill(result.AsSpan<int32_t>(), -1);

    for (size_t batch = 0; batch < batchCount; batch++)
    {
      auto connections = result.AsPointer<int32_t>();

      for (size_t bone = 0; bone < size(PoseBones); bone++)
      {
        auto boneJointMapping = PoseBones[bone];

        auto rowCount = jointConfiguration[batch][boneJointMapping.PartAffinityA].size();
        auto columnCount = jointConfiguration[batch][boneJointMapping.PartAffinityB].size();

        CostGraph costGraph{ rowCount, columnCount };
        auto costs = costGraph.AsSpan();
        auto pafs = partAffinityScores.AsSubSpan<float>(batch, bone);
        copy(pafs.begin(), pafs.end(), costs.begin());

        for (auto& value : costs)
        {
          value = -value;
        }

        PairGraph starGraph{ rowCount, columnCount };
        SolveMunkres(costGraph, starGraph);

        for (size_t row = 0; row < rowCount; row++)
        {
          for (size_t column = 0; column < columnCount; column++)
          {
            connections[row] = int32_t(column);
            connections[PoseMaxBodyCount + column] = int32_t(row);
          }
        }
      }
    }

    return result;
  }

  std::vector<std::vector<PoseJointPositions>> ConnectParts(const Tensor& connectionMap, const std::vector<PoseJointPositionCandidates>& jointConfiguration)
  {
    auto batchCount = connectionMap.Shape[0];
    vector<vector<PoseJointPositions>> results;
    results.reserve(jointConfiguration.size());

    vector<bool> visitations;
    visitations.resize(PoseJointCount * PoseMaxBodyCount);

    for (size_t batch = 0; batch < batchCount; batch++)
    {
      vector<PoseJointPositions> bodies;

      auto objectCount = 0;
      for (size_t joint = 0; joint < PoseJointCount && objectCount < PoseMaxBodyCount; joint++)
      {
        auto positionCandidateCount = jointConfiguration[batch][joint].size();

        for (size_t positionCandidate = 0; positionCandidate < positionCandidateCount && objectCount < PoseMaxBodyCount; positionCandidate++)
        {
          bool isNewObject = false;

          queue<pair<size_t, size_t>> candidateQueue;
          candidateQueue.push({ joint, positionCandidate });

          PoseJointPositions bodyJointPositions;
          ranges::fill(bodyJointPositions, XMFLOAT2{ NAN, NAN });

          while (!candidateQueue.empty())
          {
            auto [currentJoint, currentPositionCandidate] = candidateQueue.front();
            candidateQueue.pop();

            auto visitationIndex = currentJoint * PoseMaxBodyCount + currentPositionCandidate;
            if (visitations[visitationIndex]) continue;

            visitations[visitationIndex] = true;
            isNewObject = true;
            bodyJointPositions[currentJoint] = jointConfiguration[batch][joint][currentPositionCandidate];

            for (size_t bone = 0; bone < size(PoseBones); bone++)
            {
              auto boneJointMapping = PoseBones[bone];
              auto connections = connectionMap.AsSubSpan<int32_t>(batch, bone);

              if (boneJointMapping.ConfidenceA == currentJoint)
              {
                auto connection = connections[currentPositionCandidate];
                if (connection >= 0) candidateQueue.push({ boneJointMapping.ConfidenceB, connection });
              }

              if (boneJointMapping.ConfidenceB == currentJoint)
              {
                auto connection = connections[PoseMaxBodyCount + currentPositionCandidate];
                if (connection >= 0) candidateQueue.push({ boneJointMapping.ConfidenceA, connection });
              }
            }
          }

          if (isNewObject)
          {
            bodies.push_back(bodyJointPositions);
          }
        }
      }

      results.push_back(bodies);
    }

    return results;
  }
}
