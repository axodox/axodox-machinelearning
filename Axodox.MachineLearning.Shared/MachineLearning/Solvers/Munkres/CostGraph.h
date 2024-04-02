#pragma once
#include "pch.h"

namespace Axodox::MachineLearning::Solvers::Munkres
{
  class CostGraph
  {
  public:
    CostGraph(size_t rows, size_t columns);

    size_t RowCount() const;
    size_t ColumnCount() const;

    float& At(size_t row, size_t column);
    float At(size_t row, size_t column) const;

    std::span<float> AsSpan();

  private:
    size_t _rowCount, _columnCount;
    std::vector<float> _values;
  };
}