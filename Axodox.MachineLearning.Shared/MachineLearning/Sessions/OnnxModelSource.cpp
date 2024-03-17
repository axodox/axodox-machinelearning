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

  std::unique_ptr<OnnxModelSource> OnnxModelSource::FromFilePath(const std::filesystem::path& path)
  {
    return make_unique<OnnxModelSource>([=] { return read_file(path.lexically_normal()); });
  }

#ifdef WINRT_Windows_Storage_H
  std::unique_ptr<OnnxModelSource> OnnxModelSource::FromStorageFile(const winrt::Windows::Storage::StorageFile& file)
  {
    return make_unique<OnnxModelSource>([=] { return read_file(file); });
  }
#endif
}