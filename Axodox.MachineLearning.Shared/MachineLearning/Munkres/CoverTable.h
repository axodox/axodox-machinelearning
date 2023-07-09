#pragma once
#include "pch.h"

namespace Axodox::MachineLearning::Munkres
{
  class CoverTable
  {
  public:
    CoverTable(size_t rows, size_t columns);

    size_t RowCount() const;
    size_t ColumnCount() const;

    void CoverRow(size_t row);
    void CoverColumn(size_t column);

    void UncoverRow(size_t row);
    void UncoverColumn(size_t column);

    bool IsCovered(size_t row, size_t column) const;
    bool IsRowCovered(size_t row) const;
    bool IsColumnCovered(size_t column) const;

    void Clear();

  private:
    std::vector<bool> _rows, _columns;
  };
}