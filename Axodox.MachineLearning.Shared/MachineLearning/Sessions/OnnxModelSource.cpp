#include "pch.h"
#include "OnnxModelSource.h"

using namespace Axodox::Storage;
using namespace std;

namespace Axodox::MachineLearning::Sessions
{
  OnnxModelSource::OnnxModelSource(std::function<std::vector<uint8_t>()>&& source) :
    _source(move(source))
  { }

  std::vector<uint8_t> OnnxModelSource::GetModelData() const
  {
    return _source();
  }

  OnnxModelSource OnnxModelSource::FromFilePath(const std::filesystem::path& path)
  {
    return OnnxModelSource([=] { return read_file(path); });
  }

#ifdef WINRT_Windows_Storage_H
  OnnxModelSource OnnxModelSource::FromStorageFile(const winrt::Windows::Storage::StorageFile& file)
  {
    return OnnxModelSource([=] { return read_file(file); });
  }
#endif
}