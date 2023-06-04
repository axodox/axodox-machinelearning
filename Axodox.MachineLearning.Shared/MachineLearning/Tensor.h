#pragma once
#include "TensorType.h"
#include "Graphics/Textures/TextureData.h"
#include "Collections/AlignedAllocator.h"

namespace Axodox::MachineLearning
{
  struct AXODOX_MACHINELEARNING_API Tensor
  {
    static const size_t shape_dimension = 4;
    typedef std::array<size_t, shape_dimension> shape_t;

    TensorType Type;
    shape_t Shape;
    std::vector<uint8_t, Collections::aligned_allocator<uint8_t>> Buffer;

    Tensor();
    explicit Tensor(TensorType type, size_t x = 0, size_t y = 0, size_t z = 0, size_t w = 0);
    Tensor(TensorType type, shape_t shape);

    template<typename T>
    explicit Tensor(T value) :
      Type(ToTensorType<T>()),
      Shape(1, 0, 0, 0)
    {
      AllocateBuffer();
      *AsPointer<T>() = value;
    }

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    void Reset() noexcept;

    void AllocateBuffer();

    size_t ByteCount() const;
    bool IsValid() const;
    void ThrowIfInvalid() const;

    explicit operator bool() const;

    size_t Size(size_t dimension = 0) const;

    static Tensor FromTextureData(const Graphics::TextureData& texture);
    std::vector<Graphics::TextureData> ToTextureData() const;    

    const uint8_t* AsPointer(size_t x = 0, size_t y = 0, size_t z = 0, size_t w = 0) const;
    uint8_t* AsPointer(size_t x = 0, size_t y = 0, size_t z = 0, size_t w = 0);

    template<typename T>
    const T* AsPointer(size_t x = 0, size_t y = 0, size_t z = 0, size_t w = 0) const
    {
      if (ToTensorType<T>() != Type) throw std::bad_cast();
      return reinterpret_cast<const T*>(AsPointer(x, y, z, w));
    }

    template<typename T>
    T* AsPointer(size_t x = 0, size_t y = 0, size_t z = 0, size_t w = 0)
    {
      if (ToTensorType<T>() != Type) throw std::bad_cast();
      return reinterpret_cast<T*>(AsPointer(x, y, z, w));
    }

    template<typename T>
    std::span<const T> AsSpan() const
    {
      if (ToTensorType<T>() != Type) throw std::bad_cast();
      return std::span<const T>(reinterpret_cast<const T*>(Buffer.data()), Size());
    }

    template<typename T>
    std::span<T> AsSpan()
    {
      if (ToTensorType<T>() != Type) throw std::bad_cast();
      return std::span<T>(reinterpret_cast<T*>(Buffer.data()), Size());
    }

    template<typename T>
    std::span<const T> AsSubSpan(size_t x = ~0u, size_t y = ~0u, size_t z = ~0u, size_t w = ~0u) const
    {
      if (ToTensorType<T>() != Type) throw std::bad_cast();

      auto dimension = GetDimensionFromIndex(x, y, z, w);
      return std::span<const T>(AsPointer<T>(x, y, z, w), Size(dimension));
    }

    template<typename T>
    std::span<T> AsSubSpan(size_t x = ~0u, size_t y = ~0u, size_t z = ~0u, size_t w = ~0u)
    {
      if (ToTensorType<T>() != Type) throw std::bad_cast();

      auto dimension = GetDimensionFromIndex(x, y, z, w);
      return std::span<T>(AsPointer<T>(x, y, z, w), Size(dimension));
    }

    template<typename T>
    Tensor operator*(T value) const
    {
      Tensor result(*this);
      for (auto& item : result.AsSpan<T>())
      {
        item *= value;
      }
      return result;
    }

    template<typename T>
    Tensor operator/(T value) const
    {
      return *this * (T(1) / value);
    }

    Tensor DuplicateToSize(size_t instances) const;

    Tensor Duplicate(size_t instances = 2) const;

    Tensor Swizzle(size_t blockCount = 2) const;

    Tensor Reshape(shape_t shape) const;

    std::vector<Tensor> Split(size_t instances = 2) const;

    template<typename T>
    Tensor BinaryOperation(const Tensor& other, const std::function<T(T, T)>& operation) const
    {
      if (Shape != other.Shape) throw std::logic_error("Incompatible tensor shapes.");
      if (Type != other.Type) throw std::logic_error("Incompatible tensor types.");

      Tensor result{ Type, Shape };

      auto size = Size();
      auto a = AsPointer<T>();
      auto b = other.AsPointer<T>();
      auto c = result.AsPointer<T>();
      for (size_t i = 0; i < size; i++)
      {
        *c++ = operation(*a++, *b++);
      }

      return result;
    }

    template<typename T>
    void UnaryOperation(const Tensor& other, const std::function<T(T, T)>& operation)
    {
      if (Shape != other.Shape) throw std::logic_error("Incompatible tensor shapes.");
      if (Type != other.Type) throw std::logic_error("Incompatible tensor types.");
      
      auto size = Size();
      auto a = AsPointer<T>();
      auto b = other.AsPointer<T>();
      for (size_t i = 0; i < size; i++)
      {
        *a++ = operation(*a, *b++);
      }
    }

    static Tensor CreateRandom(shape_t shape, std::span<std::minstd_rand> randoms, float scale = 1.f);

    template<typename T>
    void Fill(T value)
    {
      for (auto& item : AsSpan<T>())
      {
        item = value;
      }
    }

    Tensor Concat(const Tensor& tensor) const;

    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

    Tensor ToSingle() const;
    Tensor ToHalf() const;

    static Tensor FromOrtValue(const Ort::Value& value);
    Ort::Value ToOrtValue(Ort::MemoryInfo& memoryInfo) const;

    void UpdateOrtValue(Ort::Value& value);

    static std::pair<TensorType, Tensor::shape_t> ToTypeAndShape(const Ort::TensorTypeAndShapeInfo& info);

  private:
    static size_t GetDimensionFromIndex(size_t& x, size_t& y, size_t& z, size_t& w);
    static bool AreShapesEqual(shape_t a, shape_t b, size_t startDimension = 0);
    static size_t ElementCount(shape_t shape);

    static Tensor FromTextureDataRgba8(const Graphics::TextureData& texture);
    static Tensor FromTextureDataGray8(const Graphics::TextureData& texture);
  };
}