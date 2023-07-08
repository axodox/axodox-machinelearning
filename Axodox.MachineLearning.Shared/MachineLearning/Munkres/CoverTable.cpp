#include "pch.h"
#include "CoverTable.h"

using namespace std;

namespace Axodox::MachineLearning::Munkres
{
  CoverTable::CoverTable(size_t rows, size_t columns) :
    _rows(rows),
    _columns(columns)
  {
    Clear();
  }

  size_t CoverTable::RowCount() const
  {
    return _rows.size();
  }

  size_t CoverTable::ColumnCount() const
  {
    return _columns.size();
  }

  void CoverTable::CoverRow(size_t row)
  {
    _rows[row] = true;
  }

  void CoverTable::CoverColumn(size_t column)
  {
    _columns[column] = true;
  }

  void CoverTable::UncoverRow(size_t row)
  {
    _rows[row] = false;
  }

  void CoverTable::UncoverColumn(size_t column)
  {
    _columns[column] = false;
  }

  bool CoverTable::IsCovered(size_t row, size_t column) const
  {
    return _rows[row] || _columns[column];
  }

  bool CoverTable::IsRowCovered(size_t row) const
  {
    return _rows[row];
  }

  bool CoverTable::IsColumnCovered(size_t column) const
  {
    return _columns[column];
  }

  void CoverTable::Clear()
  {
    fill(_rows.begin(), _rows.end(), false);
    fill(_columns.begin(), _columns.end(), false);
  }
}