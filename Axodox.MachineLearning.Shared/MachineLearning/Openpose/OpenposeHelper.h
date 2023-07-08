#pragma once
#include "../../includes.h"
#include "MachineLearning/Tensor.h"

namespace Axodox::MachineLearning
{
  const size_t PoseChannelCount = 18;
  const size_t PoseMaxBodyCount = 100;

  typedef std::array<std::vector<DirectX::XMFLOAT2>, PoseChannelCount> PosePeakCandidates;
  typedef std::vector<PosePeakCandidates> PosePeakCandidateCollection;

  PosePeakCandidateCollection FindPeaks(const Tensor& confidenceMap);

  struct PoseElement
  {
    size_t PartAffinityA, PartAffinityB, ConfidenceA, ConfidenceB;
  };

  const PoseElement PoseTopology[] = {
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

  Tensor CalculatePartAffinityScores(const Tensor& partAffinityMap, const PosePeakCandidateCollection& peaks);
  Tensor CalculatePartAssignments(const Tensor& partAffinityScores, const PosePeakCandidateCollection& peaks);
}