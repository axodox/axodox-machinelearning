#include "pch.h"
#include "MunkresSolver.h"
#include "CoverTable.h"

using namespace std;

namespace Axodox::MachineLearning::Solvers::Munkres
{
  void SubtractMinimumFromRows(CostGraph& costGraph)
  {
    for (size_t row = 0; row < costGraph.RowCount(); row++)
    {
      auto min = INFINITY;
      for (size_t column = 0; column < costGraph.ColumnCount(); column++)
      {
        auto value = costGraph.At(row, column);
        if (value < min) min = value;
      }

      for (size_t column = 0; column < costGraph.ColumnCount(); column++)
      {
        costGraph.At(row, column) -= min;
      }
    }
  }

  void SubtractMinimumFromColumns(CostGraph& costGraph)
  {
    for (size_t column = 0; column < costGraph.ColumnCount(); column++)
    {
      auto min = INFINITY;
      for (size_t row = 0; row < costGraph.RowCount(); row++)
      {
        auto value = costGraph.At(row, column);
        if (value < min) min = value;
      }

      for (size_t row = 0; row < costGraph.RowCount(); row++)
      {
        costGraph.At(row, column) -= min;
      }
    }
  }

  void MunkresStep1(const CostGraph& costGraph, PairGraph& starGraph)
  {
    for (size_t row = 0; row < costGraph.RowCount(); row++)
    {
      for (size_t column = 0; column < costGraph.ColumnCount(); column++)
      {
        if (!starGraph.IsRowSet(row) && !starGraph.IsColumnSet(column) && costGraph.At(row, column) == 0)
        {
          starGraph.Set(row, column);
        }
      }
    }
  }

  bool MunkresStep2(const PairGraph& starGraph, CoverTable& coverTable)
  {
    auto k = min(starGraph.RowCount(), starGraph.ColumnCount());
    auto count = 0;
    for (size_t column = 0; column < starGraph.ColumnCount(); column++)
    {
      if (starGraph.IsColumnSet(column))
      {
        coverTable.CoverColumn(column);
        count++;
      }
    }

    return count >= k;
  }

  bool MunkresStep3(const CostGraph& costGraph, const PairGraph& starGraph, PairGraph& primeGraph, CoverTable& coverTable, pair<size_t, size_t>& p)
  {
    for (size_t row = 0; row < costGraph.RowCount(); row++)
    {
      for (size_t column = 0; column < costGraph.ColumnCount(); column++)
      {
        if (costGraph.At(row, column) == 0 && !coverTable.IsCovered(row, column))
        {
          primeGraph.Set(row, column);
          if (starGraph.IsRowSet(row))
          {
            coverTable.CoverRow(row);
            coverTable.UncoverColumn(starGraph.GetColumn(row));
          }
          else
          {
            p = { row, column };
            return true;
          }
        }
      }
    }

    return false;
  }

  void MunkresStep4(PairGraph& starGraph, PairGraph& primeGraph, CoverTable& coverTable, pair<size_t, size_t> p)
  {
    while (starGraph.IsColumnSet(p.second))
    {
      pair s{ starGraph.GetRow(p.second), p.second };
      starGraph.Reset(s.first, s.second);
      starGraph.Set(p.first, p.second);
      p = { s.first, primeGraph.GetColumn(s.first) };
    }

    starGraph.Set(p.first, p.second);
    coverTable.Clear();
    primeGraph.Clear();
  }

  void MunkresStep5(CostGraph& costGraph, const CoverTable& coverTable)
  {
    float min = INFINITY;
    for (size_t row = 0; row < costGraph.RowCount(); row++)
    {
      for (size_t column = 0; column < costGraph.ColumnCount(); column++)
      {
        if (!coverTable.IsCovered(row, column))
        {
          auto value = costGraph.At(row, column);
          if (value < min)
          {
            min = value;
          }
        }
      }
    }

    for (size_t row = 0; row < costGraph.RowCount(); row++)
    {
      if (!coverTable.IsRowCovered(row)) continue;

      for (size_t column = 0; column < costGraph.ColumnCount(); column++)
      {
        costGraph.At(row, column) += min;
      }
    }

    for (size_t column = 0; column < costGraph.ColumnCount(); column++)
    {
      if (coverTable.IsColumnCovered(column)) continue;

      for (size_t row = 0; row < costGraph.RowCount(); row++)
      {
        costGraph.At(row, column) -= min;
      }
    }
  }

  void SolveMunkres(CostGraph& costGraph, PairGraph& starGraph)
  {
    PairGraph primeGraph{ costGraph.RowCount(), costGraph.ColumnCount() };
    CoverTable coverTable{ costGraph.RowCount(), costGraph.ColumnCount() };

    if (costGraph.ColumnCount() >= costGraph.RowCount())
    {
      SubtractMinimumFromRows(costGraph);
    }

    auto step = costGraph.ColumnCount() > costGraph.RowCount() ? 1 : 0;

    pair p{ size_t(-1), size_t(-1) };
    auto done = false;
    while (!done)
    {
      switch (step)
      {
      case 0:
        SubtractMinimumFromColumns(costGraph);
        [[fallthrough]];
      case 1:
        MunkresStep1(costGraph, starGraph);
        [[fallthrough]];
      case 2:
        if (MunkresStep2(starGraph, coverTable))
        {
          done = true;
          break;
        }
        [[fallthrough]];
      case 3:
        if (!MunkresStep3(costGraph, starGraph, primeGraph, coverTable, p))
        {
          step = 5;
          break;
        }
        [[fallthrough]];
      case 4:
        MunkresStep4(starGraph, primeGraph, coverTable, p);
        step = 2;
        break;
      case 5:
        MunkresStep5(costGraph, coverTable);
        step = 3;
        break;
      }
    }
  }
}