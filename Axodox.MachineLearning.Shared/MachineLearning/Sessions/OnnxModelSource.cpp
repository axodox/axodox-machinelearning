#include "pch.h"
#include "OnnxModelSource.h"

using namespace Axodox::Storage;
using namespace std;

namespace
{
  bool check_file_access(const winrt::Windows::Storage::StorageFile& file)
  {
    FILE* fileHandle = nullptr;
    auto canReachLocation = _wfopen_s(&fileHandle, file.Path().c_str(), L"rb") == 0;
    if (canReachLocation) fclose(fileHandle);
    return canReachLocation;
  }
}

namespace Axodox::MachineLearning::Sessions
{
  OnnxModelSource::OnnxModelSource(std::function<std::vector<uint8_t>()>&& source, const std::filesystem::path& pathHint) :
    _source(move(source)),
    _pathHint(pathHint)
  { }

  const std::filesystem::path& OnnxModelSource::PathHint() const
  {
    return _pathHint;
  }

  std::vector<uint8_t> OnnxModelSource::GetModelData() const
  {
    return _source();
  }

  std::unique_ptr<OnnxModelSource> OnnxModelSource::FromFilePath(const std::filesystem::path& path)
  {
    error_code ec;
    return filesystem::exists(path, ec) ? make_unique<OnnxModelSource>([=] { return read_file(path.lexically_normal()); }, path) : nullptr;
  }

#ifdef WINRT_Windows_Storage_H
  std::unique_ptr<OnnxModelSource> OnnxModelSource::FromStorageFile(const winrt::Windows::Storage::StorageFile& file)
  {
    return file ? make_unique<OnnxModelSource>([=] { return read_file(file); }, check_file_access(file) ? file.Path().c_str() : L"") : nullptr;
  }
#endif
}