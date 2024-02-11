#include "pch.h"
#include "Tensor.h"

using namespace Axodox::Graphics;
using namespace DirectX;
using namespace DirectX::PackedVector;
using namespace std;
using namespace Ort;

namespace Axodox::MachineLearning
{
  const MemoryInfo Tensor::_ortMemoryInfo = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Tensor::Tensor() :
    Type(TensorType::Unknown),
    Shape({ 0, 0, 0, 0 })
  { }

  Tensor::Tensor(TensorType type, size_t x, size_t y, size_t z, size_t w) :
    Type(type),
    Shape({ x, y, z, w })
  {
    AllocateBuffer();
  }

  Tensor::Tensor(TensorType type, TensorShape shape) :
    Type(type),
    Shape(shape)
  {
    AllocateBuffer();
  }

  Tensor::Tensor(Tensor&& other) noexcept
  {
    *this = move(other);
  }

  Tensor& Tensor::operator=(Tensor&& other) noexcept
  {
    Buffer = move(other.Buffer);
    Type = other.Type;
    Shape = other.Shape;

    other.Reset();

    return *this;
  }

  void Tensor::Reset() noexcept
  {
    Type = TensorType::Unknown;
    Shape = { 0, 0, 0, 0 };
    Buffer.clear();
  }

  void Tensor::AllocateBuffer()
  {
    Buffer.resize(ByteCount());
  }

  size_t Tensor::ByteCount() const
  {
    size_t result = Shape[0] > 0 ? GetElementSize(Type) : 0;

    for (auto dimension : Shape)
    {
      if (dimension != 0) result *= dimension;
    }

    return result;
  }

  bool Tensor::IsValid() const
  {
    return Type != TensorType::Unknown && !Buffer.empty() && Buffer.size() == ByteCount();
  }

  void Tensor::ThrowIfInvalid() const
  {
    if (IsValid()) throw runtime_error("The tensor is invalid.");
  }

  Tensor::operator bool() const
  {
    return IsValid();
  }

  size_t Tensor::Size(size_t index) const
  {
    size_t result = 1;

    for (auto i = index; i < Shape.size(); i++)
    {
      if (Shape[i] > 0) result *= Shape[i];
    }

    return result;
  }

  Tensor Tensor::CreateRandom(TensorShape shape, std::span<std::minstd_rand> randoms, float scale)
  {
    static uniform_real_distribution<float> floatDistribution(0.f, 1.f);

    Tensor result{ TensorType::Single, shape };

    for (size_t i = 0; i < shape[0]; i++)
    {
      for (auto& value : result.AsSubSpan<float>(i))
      {
        auto u1 = floatDistribution(randoms[i]);
        auto u2 = floatDistribution(randoms[i]);
        auto radius = sqrt(-2.f * log(u1));
        auto theta = 2.f * XM_PI * u2;
        auto standardNormalRand = radius * cos(theta);

        value = standardNormalRand * scale;
      }
    }

    return result;
  }

  Tensor Tensor::Concat(const Tensor& other) const
  {
    if (!AreShapesEqual(Shape, other.Shape, 1)) throw invalid_argument("other");
    if (Type != other.Type) throw invalid_argument("other");

    auto shape = Shape;
    shape[0] += other.Shape[0];

    Tensor tensor{ Type, shape };

    auto baseByteCount = ByteCount();

    auto pTarget = tensor.AsPointer();
    memcpy(pTarget, AsPointer(), baseByteCount);
    memcpy(pTarget + baseByteCount, other.AsPointer(), other.ByteCount());

    return tensor;
  }

  bool Tensor::operator==(const Tensor& other) const
  {
    return Type == other.Type && Shape == other.Shape &&
      0 == memcmp(AsPointer(), other.AsPointer(), ByteCount());
  }

  bool Tensor::operator!=(const Tensor& other) const
  {
    return !(*this == other);
  }

  Tensor Tensor::ToSingle(bool isEnabled) const
  {
    if (!isEnabled || Type == TensorType::Single) return *this;
    if (Type != TensorType::Half) throw bad_cast();

    auto size = Size();

    Tensor result{ TensorType::Single, Shape };
    XMConvertHalfToFloatStream(reinterpret_cast<float*>(result.Buffer.data()), 4, reinterpret_cast<const HALF*>(Buffer.data()), 2, size);

    return result;
  }

  Tensor Tensor::ToHalf(bool isEnabled) const
  {
    if (!isEnabled || Type == TensorType::Half) return *this;
    if (Type != TensorType::Single) throw bad_cast();

    auto size = Size();

    Tensor result{ TensorType::Half, Shape };
    XMConvertFloatToHalfStream(reinterpret_cast<HALF*>(result.Buffer.data()), 2, reinterpret_cast<const float*>(Buffer.data()), 4, size);

    return result;
  }

  Tensor Tensor::ToInt64(bool isEnabled) const
  {
    if (!isEnabled || Type == TensorType::Int64) return *this;
    if (Type != TensorType::Int32) throw bad_cast();

    auto size = Size();

    Tensor result{ TensorType::Int64, Shape };
    auto pResult = result.AsPointer<int64_t>();
    for (auto value : AsSpan<int32_t>())
    {
      *pResult++ = value;
    }    

    return result;
  }

  size_t Tensor::GetDimensionFromIndex(size_t& x, size_t& y, size_t& z, size_t& w)
  {
    auto dimension = 4;

    if (w == ~0u)
    {
      dimension = 3;
      w = 0;
    }

    if (z == ~0u)
    {
      dimension = 2;
      z = 0;
    }

    if (y == ~0u)
    {
      dimension = 1;
      y = 0;
    }

    if (x == ~0u)
    {
      dimension = 0;
      x = 0;
    }

    return dimension;
  }

  bool Tensor::AreShapesEqual(TensorShape a, TensorShape b, size_t startDimension)
  {
    for (size_t i = startDimension; i < TensorDimension; i++)
    {
      if (a[i] != b[i]) return false;
    }

    return true;
  }

  size_t Tensor::ElementCount(TensorShape shape)
  {
    size_t result = shape[0] > 0 ? 1 : 0;
    for (auto item : shape)
    {
      if (item) result *= item;
      else break;
    }
    return result;
  }

  Tensor Tensor::FromTextureDataRgba8(const Graphics::TextureData& texture, ColorNormalization normalization)
  {
    Tensor result(TensorType::Single, 1, 3, texture.Height, texture.Width);

    if (texture.Format != DXGI_FORMAT_B8G8R8A8_UNORM && texture.Format != DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) throw logic_error("Unsupported texture format.");

    auto rTarget = result.AsPointer<float>(0, 0);
    auto gTarget = result.AsPointer<float>(0, 1);
    auto bTarget = result.AsPointer<float>(0, 2);

    XMFLOAT4A color;
    for (uint32_t y = 0u; y < texture.Height; y++)
    {
      auto pSource = texture.Row<XMUBYTEN4>(y);
      for (uint32_t x = 0u; x < texture.Width; x++)
      {
        switch (normalization)
        {
        case ColorNormalization::LinearPlusMinusOne:
          XMStoreFloat4A(&color, XMVectorScale(XMLoadUByteN4(pSource++) - XMVectorReplicate(0.5f), 2.f));
          break;
        case ColorNormalization::LinearZeroToOne:
          XMStoreFloat4A(&color, XMLoadUByteN4(pSource++));
          break;
        default:
          throw logic_error("Normalization mode not implemented.");
        }

        *rTarget++ = color.z;
        *gTarget++ = color.y;
        *bTarget++ = color.x;
      }
    }

    return result;
  }

  Tensor Tensor::FromTextureDataGray8(const Graphics::TextureData& texture, ColorNormalization normalization)
  {
    Tensor result(TensorType::Single, 1, 1, texture.Height, texture.Width);

    if (texture.Format != DXGI_FORMAT_R8_UNORM) throw logic_error("Unsupported texture format.");

    auto pTarget = result.AsPointer<float>();
    for (uint32_t y = 0u; y < texture.Height; y++)
    {
      auto pSource = texture.Row<uint8_t>(y);
      for (uint32_t x = 0u; x < texture.Width; x++)
      {
        switch (normalization)
        {
        case ColorNormalization::LinearPlusMinusOne:
          *pTarget++ = 2.f * (*pSource++ / 255.f - 0.5f);
          break;
        case ColorNormalization::LinearZeroToOne:
          *pTarget++ = *pSource++ / 255.f;
          break;
        default:
          throw logic_error("Normalization mode not implemented.");
        }
      }
    }

    return result;
  }

  std::vector<Graphics::TextureData> Tensor::ToTextureDataRgba8(ColorNormalization normalization) const
  {
    vector<TextureData> results;
    results.reserve(Shape[0]);

    auto width = uint32_t(Shape[3]);
    auto height = uint32_t(Shape[2]);
    for (size_t i = 0u; i < Shape[0]; i++)
    {
      TextureData result{ width, height, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB };

      auto rSource = AsPointer<float>(i, 0);
      auto gSource = AsPointer<float>(i, 1);
      auto bSource = AsPointer<float>(i, 2);
      for (uint32_t y = 0u; y < height; y++)
      {
        auto pTarget = result.Row<XMUBYTEN4>(y);
        for (uint32_t x = 0u; x < width; x++)
        {
          XMVECTOR color;
          switch (normalization)
          {
          case ColorNormalization::LinearPlusMinusOne:
            color = XMVectorSet(*bSource++, *gSource++, *rSource++, 1.f) / 2.f + XMVectorReplicate(0.5f);
            break;
          case ColorNormalization::LinearZeroToOne:
            color = XMVectorSet(*bSource++, *gSource++, *rSource++, 1.f);
            break;
          default:
            throw logic_error("Normalization mode not implemented.");
          }
          XMStoreUByteN4(pTarget++, XMVectorSaturate(color));
        }
      }

      results.push_back(move(result));
    }

    return results;
  }

  std::vector<Graphics::TextureData> Tensor::ToTextureDataGray8(ColorNormalization normalization) const
  {
    vector<TextureData> results;
    results.reserve(Shape[0]);

    auto width = uint32_t(Shape[3]);
    auto height = uint32_t(Shape[2]);
    for (size_t i = 0u; i < Shape[0]; i++)
    {
      TextureData result{ width, height, DXGI_FORMAT_R8_UNORM };

      auto pSource = AsPointer<float>(i);
      for (uint32_t y = 0u; y < height; y++)
      {
        auto pTarget = result.Row<uint8_t>(y);
        for (uint32_t x = 0u; x < width; x++)
        {
          float color;
          switch (normalization)
          {
          case ColorNormalization::LinearPlusMinusOne:
            color = *pSource++ / 2.f + 0.5f;
            break;
          case ColorNormalization::LinearZeroToOne:
            color = *pSource++;
            break;
          default:
            throw logic_error("Normalization mode not implemented.");
          }
          *pTarget++ = uint8_t(clamp(color, 0.f, 1.f) * 255.f);
        }
      }

      results.push_back(move(result));
    }

    return results;
  }

  Tensor Tensor::FromTextureData(const Graphics::TextureData& texture, ColorNormalization normalization)
  {
    if (!texture) return {};

    switch (texture.Format)
    {
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
      return FromTextureDataRgba8(texture, normalization);
    case DXGI_FORMAT_R8_UNORM:
      return FromTextureDataGray8(texture, normalization);
    default:
      throw logic_error("Unsupported texture format.");
    }
  }

  std::vector<Graphics::TextureData> Tensor::ToTextureData(ColorNormalization normalization) const
  {
    if (Type != TensorType::Single) throw bad_cast();

    switch (Shape[1])
    {
    case 1:
      return ToTextureDataGray8(normalization);
    case 3:
      return ToTextureDataRgba8(normalization);
    default:
      throw logic_error("Unsupported channel count.");
    }
  }

  const uint8_t* Tensor::AsPointer(size_t x, size_t y, size_t z, size_t w) const
  {
    TensorShape index{ x, y, z, w };

    auto elementSize = GetElementSize(Type);
    size_t offset = 0;
    for (size_t i = 0; i < Shape.size(); i++)
    {
      offset += index[i] * Size(i + 1) * elementSize;
    }

    if (offset > Buffer.size()) throw out_of_range("Tensor index out of range.");

    return Buffer.data() + offset;
  }

  uint8_t* Tensor::AsPointer(size_t x, size_t y, size_t z, size_t w)
  {
    return const_cast<uint8_t*>(static_cast<const Tensor*>(this)->AsPointer(x, y, z, w));
  }

  Tensor Tensor::DuplicateToSize(size_t instances) const
  {
    if (Shape[0] == instances) return *this;

    if (instances % Shape[0] != 0) throw out_of_range("The instance count must be a multiple of the current shape[0].");

    return Duplicate(instances / Shape[0]);
  }

  Tensor Tensor::Duplicate(size_t instances) const
  {
    Tensor tensor{ Type, Shape[0] * instances, Shape[1], Shape[2], Shape[3] };

    for (size_t i = 0; i < instances; i++)
    {
      memcpy(tensor.AsPointer(i * Shape[0]), AsPointer(), ByteCount());
    }

    return tensor;
  }

  Tensor Tensor::Swizzle(size_t blockCount) const
  {
    Tensor result{ Type, Shape };

    auto blockByteCount = ByteCount() / Shape[0];
    auto blockSize = Shape[0] / blockCount;
    for (size_t i = 0; i < blockCount; i++)
    {
      for (size_t j = 0; j < blockSize; j++)
      {
        memcpy(result.AsPointer(i * blockSize + j), AsPointer(j * blockCount + i), blockByteCount);
      }
    }

    return result;
  }

  Tensor Tensor::Reshape(TensorShape shape) const
  {
    if (ElementCount(shape) != ElementCount(Shape)) throw bad_cast();

    auto result{ *this };
    result.Shape = shape;
    return result;
  }

  std::vector<Tensor> Tensor::Split(size_t instances) const
  {
    if (Shape[0] % instances != 0) throw invalid_argument("instances");

    auto newShape = Shape;
    newShape[0] /= instances;

    vector<Tensor> results;
    results.resize(instances);
    for (size_t i = 0; auto & result : results)
    {
      result = Tensor(Type, newShape);
      memcpy(result.AsPointer(), AsPointer(i), result.ByteCount());

      i += newShape[0];
    }

    return results;
  }

  Tensor Tensor::FromOrtValue(const Ort::Value& value)
  {
    Tensor result;

    //Set type and shape
    auto info = TensorInfo::FromTypeAndShapeInfo(value.GetTensorTypeAndShapeInfo());
    result.Type = info.Type;
    result.Shape = info.Shape;
    result.AllocateBuffer();

    //Copy data
    memcpy(result.Buffer.data(), value.GetTensorRawData(), result.Buffer.size());

    return result;
  }

  Ort::Value Tensor::ToOrtValue() const
  {
    std::vector<int64_t> shape;
    for (auto dimension : Shape)
    {
      if (dimension != 0) shape.push_back(int64_t(dimension));
    }

    return Value::CreateTensor(_ortMemoryInfo, const_cast<uint8_t*>(Buffer.data()), Buffer.size(), shape.data(), shape.size(), ToTensorType(Type));
  }

  void Tensor::UpdateOrtValue(Ort::Value& value)
  {
    auto info = TensorInfo::FromTypeAndShapeInfo(value.GetTensorTypeAndShapeInfo());
    if (info.Type != Type || info.Shape != Shape) throw bad_cast();

    auto data = value.GetTensorMutableRawData();
    memcpy(data, AsPointer(), ByteCount());
  }
}