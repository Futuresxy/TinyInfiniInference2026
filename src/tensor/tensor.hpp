#pragma once  //防止头文件被重复包含
#include "../core/llaisys_core.hpp"

#include <vector>
namespace llaisys {  //用于防止命名冲突
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;   //智能指针“引用计数”机制。当没有变量再指向这个张量时，它会自动释放内存

struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;  //size_t 是无符号整数类型，适合表示大小和索引
    std::vector<ptrdiff_t> strides; //ptrdiff_t 是有符号整数INT类型，适合表示指针之间的差值
};

class Tensor {
private:
    TensorMeta _meta;
    core::storage_t _storage;
    size_t _offset;
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);  //外部不能直接使用 Tensor t(...) 来实例化

public:
    static tensor_t create(  //静态工厂，提供统一的创建入口。这在框架设计中很常见，便于在创建对象前进行设备检查、内存分配等逻辑。
        const std::vector<size_t> &shape,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        int device = 0);
    ~Tensor() = default;
    // Info
    std::byte *data();  //字节指针,代替 unsigned char 表示纯粹的内存数据
    const std::byte *data() const;
    size_t ndim() const;
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    llaisysDataType_t dtype() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    size_t numel() const;
    size_t elementSize() const;

    std::string info() const;
    void debug() const;

    bool isContiguous() const;

    // Meta Transform
    tensor_t permute(const std::vector<size_t> &order) const;
    tensor_t slice(size_t dim, size_t start, size_t end) const;
    tensor_t view(const std::vector<size_t> &shape) const;

    // Load data from host memory
    void load(const void *src);

    // Challenging features
    tensor_t contiguous() const;
    tensor_t reshape(const std::vector<size_t> &shape) const;
    tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;
};

} // namespace llaisys
