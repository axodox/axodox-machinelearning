#pragma once
#include "CostGraph.h"
#include "PairGraph.h"

namespace Axodox::MachineLearning::Munkres
{
  void SolveMunkres(CostGraph& costGraph, PairGraph& starGraph);
}