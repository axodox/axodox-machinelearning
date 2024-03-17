#include "pch.h"
#include "CostGraph.h"

namespace Axodox::MachineLearning::Solvers::Munkres
{
  CostGraph::CostGraph(size_t rows, size_t columns) :
    _rowCount(rows),
    _columnCount(columns),
    _values(rows * columns)
  { }

  size_t CostGraph::RowCount() const
  {
    return _rowCount;
  }

  size_t CostGraph::ColumnCount() const
  {
    return _columnCount;
  }

  float& CostGraph::At(size_t row, size_t column)
  {
    return _values[row * _columnCount + column];
  }

  float CostGraph::At(size_t row, size_t column) const
  {
    return _values[row * _columnCount + column];
  }

  std::span<float> CostGraph::AsSpan()
  {
    return _values;
  }
}