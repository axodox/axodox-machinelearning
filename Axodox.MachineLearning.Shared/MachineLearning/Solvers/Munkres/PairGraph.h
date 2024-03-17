#pragma once
#include "pch.h"

namespace Axodox::MachineLearning::Solvers::Munkres
{
  class PairGraph
  {
  public:
    PairGraph(size_t rowCount, size_t columnCount);

    size_t RowCount() const;
    size_t ColumnCount() const;

    bool IsRowSet(size_t row) const;
    bool IsColumnSet(size_t column) const;

    void Set(size_t row, size_t column);
    void Reset(size_t row, size_t column);

    size_t GetRow(size_t column) const;
    size_t GetColumn(size_t row) const;

    bool IsPair(size_t row, size_t column) const;

    void Clear();

  private:
    static const size_t _invalidValue = size_t(-1);
    std::vector<size_t> _rows, _columns;
  };
}