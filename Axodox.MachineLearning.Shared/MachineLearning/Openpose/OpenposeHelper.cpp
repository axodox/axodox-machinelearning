#include "pch.h"
#include "OpenposeHelper.h"
#include "MachineLearning/Munkres/CostGraph.h"
#include "MachineLearning/Munkres/PairGraph.h"
#include "MachineLearning/Munkres/Munkres.h"

using namespace Axodox::MachineLearning::Munkres;
using namespace DirectX;
using namespace std;

namespace Axodox::MachineLearning
{
  const float PoseConfidenceMapThreshold = 0.1f;
  const int PoseConfidenceWindowSize = 5;
  const int PosePartAffinityIntegralSamples = 7;

  PosePeakCandidateCollection FindPeaks(const Tensor& confidenceMap)
  {
    if (confidenceMap.Shape[1] != PoseChannelCount) throw logic_error("The number of channels in the confidence map is invalid.");

    auto width = int(confidenceMap.Shape[3]);
    auto height = int(confidenceMap.Shape[2]);

    PosePeakCandidateCollection results;
    results.reserve(confidenceMap.Shape[0]);

    for (size_t batch = 0; batch < confidenceMap.Shape[0]; batch++)
    {
      PosePeakCandidates candidate;

      for (size_t channel = 0; channel < confidenceMap.Shape[1]; channel++)
      {
        auto& peaks = candidate[channel];

        auto value = confidenceMap.AsPointer<float>(batch, channel);
        for (auto y = 0; y < height; y++)
        {
          for (auto x = 0; x < width; x++, value++)
          {
            //When we find a value above the threshold            
            if (*value < PoseConfidenceMapThreshold) continue;

            //We check the area around it
            auto xMin = max(0, x - PoseConfidenceWindowSize);
            auto yMin = max(0, y - PoseConfidenceWindowSize);
            auto xMax = min(0, x + PoseConfidenceWindowSize);
            auto yMax = min(0, y + PoseConfidenceWindowSize);

            auto isPeak = true;
            auto peakValue = *value;
            for (auto yWindow = yMin; isPeak && yWindow < yMax; yWindow++)
            {
              auto neighbour = confidenceMap.AsPointer<float>(batch, channel, yWindow);
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
              auto neighbour = confidenceMap.AsPointer<float>(batch, channel, yWindow);
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
            peaks.push_back(refinedLocation);
          }
        }

        results.push_back(candidate);
      }
    }

    return results;
  }

  Tensor CalculatePartAffinityScores(const Tensor& partAffinityMap, const PosePeakCandidateCollection& peakCollection)
  {
    if (partAffinityMap.Shape[1] != 2 * size(PoseTopology)) throw logic_error("The number of channels in the part affinity map is invalid.");

    auto batchCount = partAffinityMap.Shape[0];
    auto width = int(partAffinityMap.Shape[3]);
    auto height = int(partAffinityMap.Shape[2]);

    Tensor result{ TensorType::Single, batchCount, size(PoseTopology), PoseMaxBodyCount, PoseMaxBodyCount };

    for (size_t batch = 0; batch < batchCount; batch++)
    {
      auto& peaks = peakCollection[batch];

      for (size_t channel = 0; channel < size(PoseTopology); channel++)
      {
        auto graph = result.AsPointer<float>(batch, channel);
        auto topology = PoseTopology[channel];

        auto& peaksA = peaks[topology.ConfidenceA];
        auto& peaksB = peaks[topology.ConfidenceB];

        auto pafA = partAffinityMap.AsPointer<float>(batch, topology.PartAffinityA);
        auto pafB = partAffinityMap.AsPointer<float>(batch, topology.PartAffinityB);

        for (auto a = 0; a < peaksA.size(); a++)
        {
          auto peakA = XMLoadFloat2(&peaksA[a]);
          for (auto b = 0; b < peaksB.size(); b++)
          {
            auto peakB = XMLoadFloat2(&peaksB[b]);

            auto vectorAB = peakA - peakB;
            auto normAB = XMVector2Normalize(vectorAB);

            auto integral = 0.f;
            auto increment = 1.f / (PosePartAffinityIntegralSamples - 1);
            auto progress = 0.f;
            for (auto t = 0; t < PosePartAffinityIntegralSamples; t++)
            {
              XMINT2 pixel;
              XMStoreSInt2(&pixel, XMConvertVectorFloatToInt(peakA + progress * vectorAB, 0));

              if (pixel.x < 0 || pixel.y < 0 || pixel.x >= width || pixel.y >= height) continue;

              auto paf = XMVectorSet(pafA[pixel.y * width + pixel.x], pafB[pixel.y * width + pixel.x], 0, 0);

              auto dot = XMVectorGetX(XMVector2Dot(paf, normAB));
              integral += dot;

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

  Tensor CalculatePartAssignments(const Tensor& partAffinityScores, const PosePeakCandidateCollection& peaks)
  {
    auto batchCount = partAffinityScores.Shape[0];

    Tensor result{ TensorType::Int32, batchCount, size(PoseTopology), 2, PoseMaxBodyCount };

    for (size_t batch = 0; batch < batchCount; batch++)
    {
      auto connections = result.AsPointer<int32_t>();

      for (size_t channel = 0; channel < size(PoseTopology); channel++)
      {
        auto topology = PoseTopology[channel];

        auto rowCount = peaks[batch][topology.PartAffinityA].size();
        auto columnCount = peaks[batch][topology.PartAffinityB].size();

        CostGraph costGraph{ rowCount, columnCount };
        auto costs = costGraph.AsSpan();
        auto pafs = partAffinityScores.AsSubSpan<float>(batch, channel);
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

  void ConnectParts(const Tensor& connectionMap, const PosePeakCandidateCollection& peaks)
  {
    auto batchCount = connectionMap.Shape[0];
    Tensor results{ TensorType::Int32, batchCount, PoseMaxBodyCount, PoseChannelCount };

    vector<bool> visitations;
    visitations.resize(PoseChannelCount * PoseMaxBodyCount);

    for (size_t batch = 0; batch < batchCount; batch++)
    {
      auto connections = connectionMap.AsSubSpan<int32_t>(batch);

      auto objects = results.AsSubSpan<int32_t>(batch);
      ranges::fill(objects, -1);

      auto objectCount = 0;
      for (size_t channel = 0; channel < PoseChannelCount && objectCount < PoseMaxBodyCount; channel++)
      {
        auto count = peaks[batch][channel].size();

        for (size_t i = 0; i < count && objectCount < PoseMaxBodyCount; i++)
        {
          bool isNewObject = false;

          queue<pair<int, int>> queue;
          queue.push({ channel, i });

          while (!queue.empty())
          {
            auto [nodeC, nodeI] = queue.front();
            queue.pop();

            auto visitationIndex = nodeC * PoseMaxBodyCount + nodeI;
            if (visitations[visitationIndex]) continue;

            visitations[visitationIndex] = true;
            isNewObject = true;
            objects[objectCount * PoseChannelCount + nodeC] = nodeI;

            for (size_t k = 0; k < size(PoseTopology); k++)
            {
              auto topology = PoseTopology[k];
              auto connection = &connections[k * 2 * PoseMaxBodyCount];

              if (topology.ConfidenceA == nodeC)
              {
                auto i_b = connection[nodeI];
                if (i_b >= 0) queue.push({ topology.ConfidenceB, i_b });
              }

              if (topology.ConfidenceB == nodeC)
              {
                auto i_a = connection[M + nodeI];
                if (i_a >= 0) queue.push({ topology.ConfidenceA, i_a });
              }
            }
          }

          if (isNewObject) objectCount++;
        }
      }


    }


  }
}
