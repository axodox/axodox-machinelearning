#include "pch.h"
#include "PairGraph.h"

using namespace std;

namespace Axodox::MachineLearning::Munkres
{
  PairGraph::PairGraph(size_t rowCount, size_t columnCount) :
    _rows(rowCount),
    _columns(columnCount)
  {
    Clear();
  }

  size_t PairGraph::RowCount() const
  {
    return _rows.size();
  }

  size_t PairGraph::ColumnCount() const
  {
    return _columns.size();
  }

  bool PairGraph::IsRowSet(size_t row) const
  {
    return _rows[row] != _invalidValue;
  }

  bool PairGraph::IsColumnSet(size_t column) const
  {
    return _columns[column] != _invalidValue;
  }

  void PairGraph::Set(size_t row, size_t column)
  {
    _rows[row] = column;
    _columns[column] = row;
  }

  void PairGraph::Reset(size_t row, size_t column)
  {
    _rows[row] = _invalidValue;
    _columns[column] = _invalidValue;
  }

  size_t PairGraph::GetRow(size_t column) const
  {
    return _columns[column];
  }

  size_t PairGraph::GetColumn(size_t row) const
  {
    return _rows[row];
  }

  bool PairGraph::IsPair(size_t row, size_t column) const
  {
    return _rows[row] == column;
  }

  void PairGraph::Clear()
  {
    ranges::fill(_rows, _invalidValue);
    ranges::fill(_columns, _invalidValue);
  }
}