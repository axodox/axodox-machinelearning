#pragma once
#include "CostGraph.h"
#include "PairGraph.h"

namespace Axodox::MachineLearning::Solvers::Munkres
{
  void SolveMunkres(CostGraph& costGraph, PairGraph& starGraph);
}