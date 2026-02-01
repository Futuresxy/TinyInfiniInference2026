#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {  // 高纬度 ---> 低纬度
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}


void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}
//内存中是否连续
bool Tensor::isContiguous() const {
    size_t ndim = this->ndim();
    // 0维或1维张量（如果最后一个步长是1）通常认为是连续的
    if (ndim == 0) return true;

    // 检查最后一维步长是否为 1
    if (static_cast<size_t>(this->strides()[ndim - 1]) != 1) {
        return false;
    }

    // 比较当前维度的步长是否等于 后一维步长 * 后一维形状
    for (size_t i = ndim - 1; i > 0; i--) {
        size_t current_stride = static_cast<size_t>(this->strides()[i - 1]);
        size_t next_stride = static_cast<size_t>(this->strides()[i]);
        size_t next_shape = this->shape()[i];

        if (current_stride != next_stride * next_shape) {
            return false;
        }
    }
    return true;}
//创建一个新张量，改变原始张量维度的顺序。不涉及数据传输
//例如，将形状为(2, 3, 4)的张量的维度顺序更改为(4, 2, 3)。
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim_ = this->ndim();
    
    // 1. 校验输入合法性
    CHECK_ARGUMENT(order.size() == ndim_, "Permute order size must match tensor ndim");
    
    std::vector<bool> used(ndim_, false);
    for (auto d : order) {
        CHECK_ARGUMENT(d < ndim_, "Permute axis out of range");
        CHECK_ARGUMENT(!used[d], "Permute order must not contain duplicate axes");
        used[d] = true;
    }

    // 2. 构造新的元数据
    std::vector<size_t> new_shape(ndim_);
    std::vector<ptrdiff_t> new_strides(ndim_);

    for (size_t i = 0; i < ndim_; i++) {
        // 将旧的维度信息映射到新位置
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }

    TensorMeta new_meta{this->dtype(), new_shape, new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage,this->_offset));
}
//创建一个新张量，通过拆分或合并原始维度将原始张量重塑为给定形状。不涉及数据传输
//例如，通过合并最后两个维度，将形状为(2, 3, 5)的张量更改为(2, 15)。
tensor_t Tensor::view(const std::vector<size_t> &new_shape) const {
    // 1. 校验元素总数是否匹配
    size_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    ASSERT(new_numel == this->numel(), "Total elements must remain the same in view");

    // 2. 只有连续存储的张量才能直接 view (通常逻辑如此)
    ASSERT(this->isContiguous(), "View only supports contiguous tensors");

    // 3. 计算新步长 (Row-major)
    std::vector<ptrdiff_t> new_strides(new_shape.size());
    size_t st = 1;
    for (int i = new_shape.size() - 1; i >= 0; --i) {
        new_strides[i] = st;
        st *= new_shape[i];
    }

    // 4. 构造新 Meta
    TensorMeta new_meta{this->dtype(), new_shape, new_strides};

    // 5. 创建新 Tensor 对象，共享 _storage，传递当前的 _offset
    // 注意：这里调用的是私有构造函数
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}
//创建一个新张量，沿给定维度，start（包含）和end（不包含）索引对原始张量进行切片操作。
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    size_t ndim = this->ndim();
    CHECK_ARGUMENT(dim < ndim, "Slice dimension out of range");
    CHECK_ARGUMENT(start < end && end <= this->shape()[dim], "Slice indices out of range");

    //如果start不是0，说明有偏移
    size_t new_offset = _offset + start * _meta.strides[dim] * this->elementSize();
    std::vector<size_t> new_shape = this->shape();
    new_shape[dim] = end - start;
    TensorMeta new_meta{this->dtype(), new_shape, this->strides()};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, new_offset));
}
//将主机（cpu）数据加载到张量（可以在设备上）
void Tensor::load(const void *src_) {  //src_ 指向主机内存
    core::context().setDevice(this->deviceType(), this->deviceId());
    llaisysMemcpyKind_t Kind = (this->deviceType() == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;
    core::context().runtime().api()->memcpy_sync(
        this->data(),
        src_,
        this->numel() * this->elementSize(),
        Kind);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
