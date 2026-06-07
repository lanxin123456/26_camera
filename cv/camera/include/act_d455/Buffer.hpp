#ifndef BUFFER_HPP
#define BUFFER_HPP
/*
 * @file Buffer.hpp
 * @brief 线程安全的缓冲区库
 * @author 澜汐善溟渊
 * @version 1.1
 * @date 2025-12-16
 * 
 * 使用示例：
 * @code
 * // 创建单缓冲池（编译期定大小）
 * buffer::StaticBuffer<int, 1024> buffer_pool("buffer_name"); // 静态缓冲区
 * buffer::StaticRingBuffer<int, 1024> buffer_pool("buffer_name", 1024); // 静态环形缓冲区
 * // 创建单缓冲池（运行期定大小）
 * buffer::DynamicBuffer<int> buffer_pool("buffer_name"); // 动态缓冲区
 * buffer::DynamicRingBuffer<int> buffer_pool("buffer_name", 1024); // 动态环形缓冲区
 * 
 * // 单缓冲池的操作方法
 * // - 写入的数据类型必须可析构
 * buffer_pool.write(data); // 写入数据，返回是否成功，支持移动语义
 * buffer_pool.read(data); // 读取数据，返回是否成功，支持移动语义
 * buffer_pool.try_write(data); // 尝试写入数据，返回是否成功
 * buffer_pool.try_read(data); // 尝试读取数据，返回是否成功
 * buffer_pool.write_with_timeout(data, timeout); // 带超时的写入数据，返回是否成功
 * buffer_pool.read_with_timeout(data, timeout); // 带超时的读取数据，返回是否成功
 * buffer_pool.is_empty(); // 判断缓冲区是否为空
 * buffer_pool.has_space(); // 判断缓冲区是否有空间
 * buffer_pool.capacity(); // 获取缓冲区容量
 * buffer_pool.size(); // 获取缓冲区当前大小
 * buffer_pool.clear(); // 清空缓冲区
 * 
 * std::vector<int> data = {1, 2, 3, 4, 5};
 * size_t written = buffer_pool.write_batch(data.begin(), data.end()); // 批量写入数据，返回写入成功的元素数量
 * size_t read_count = buffer_pool.read_batch(data.begin(), data.end()); // 批量读取数据，返回读取成功的元素数量
 * 
 * // 创建双缓冲池（编译期定大小）
 * buffer::StaticDoubleBuffer<int, 1024> buffer_pool("buffer_name"); // 静态双缓冲区
 * // 创建双缓冲池（运行期定大小）
 * buffer::DynamicDoubleBuffer<int> buffer_pool("buffer_name", 1024); // 动态双缓冲区
 * 
 * // 双缓冲池使用示例，仅可在多线程下使用，否则可能会抛出死锁的异常
 * // - 设计用于单生产者 + 单消费者模型，不支持单线程下的读写操作
 * // - 判断是否停止需要外部信号，例如使用 std::atomic<bool> 标志位，双缓冲的write和read都只返回是否成功写入或读取
 * // - 设计思路是优先保证实时性，所以在write时如果缓冲区已满，会直接交换缓冲区，让消费者消费“更新”的数据，而不是阻塞等待读写消费完上一个缓冲区
 * // - 没有设置主动flush机制，如果需要，需要手动调用flush()方法保证最后池中的数据被消费
 * 
 * @code
 * #include <iostream>
 * #include <csignal>
 * #include <atomic>
 * #include <thread>
 * #include <Buffer.hpp>
 * 
 * std::atomic<bool> stop_flag;
 * void signal_handler(int signum)
 * {
 *     stop_flag.store(true);
 * }
 * 
 * void produce(buffer::DynamicDoubleBuffer<int>& buffer, int size)
 * {
 *     for (int i = 0; i < size; ++i)
 *     {
 *         buffer.write(i);
 *         std::cout << i << std::endl;
 *     }
 *     // buffer.flush(); // 手动刷新缓冲区，确保所有数据被消费
 * }
 * 
 * void consume(buffer::DynamicDoubleBuffer<int>& buffer, int size, std::atomic<bool>& stop_flag)
 * {
 *     int consumed = 0;
 *     while(!stop_flag)
 *     {
 *         if(buffer.read(consumed))
 *         {
 *             std::cout << consumed << std::endl;
 *         }
 *     }
 * }
 * 
 * int main()
 * {
 *     signal(SIGINT, signal_handler);
 *     int size;
 *     std::cin >> size;
 *     buffer::DynamicDoubleBuffer<int> buffer("test_", 16); // 缓冲区大小不要设置太大，缓冲区大小≈生产与消费的延迟
 * 
 *     std::thread prod_thread(produce, std::ref(buffer), size);
 *     std::thread cons_thread(consume, std::ref(buffer), size, std::ref(stop_flag));
 *         
 *     // 等待线程完成
 *     prod_thread.join();
 *     cons_thread.join();
 * }
 * 
 * @endcode
 */


#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <deque>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/types.h>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#if __cplusplus >= 201703L
    #include <optional>
#endif

// #define IS_DEBUG
// #define BUFFER_ENABLE_LOGGING

namespace buffer
{
    #ifdef DEBUG
    #pragma push_macro("DEBUG")
    #undef DEBUG
    #endif
    /*
     * @brief 辅助工具类系
     */
    // monostate
    #if __cplusplus < 201703L
        struct monostate {};
    #else
        using std::monostate;
    #endif
    // 策略类
    enum class OverwritePolicy
    {
        RING, // 环形缓冲区
        LINE, // 线性缓冲区
    };
    // 互斥锁类
    class Mutex
    {
    public:
        // RALL阻塞式加锁类
        class ScopedLock
        {
        public:
            explicit ScopedLock(Mutex& mutex) : mutex_(mutex)
            {
                mutex_.lock();
            }
            ~ScopedLock()
            {
                mutex_.unlock();
            }
        private:
            Mutex& mutex_;
        };
        // RALL非阻塞式加锁类
        class TryLock
        {
        public:
            explicit TryLock(Mutex& mutex) : mutex_(mutex), acquired_(mutex_.try_lock()) {}
            ~TryLock()
            {
                if(acquired_)
                {
                    mutex_.unlock();
                }
            }
            bool acquired() const { return acquired_; }
            explicit operator bool() const { return acquired_; }
        private:
            Mutex& mutex_;
            bool acquired_;
        };
        // RALL延时加锁类
        class DelayedLock
        {
        public:
            explicit DelayedLock(Mutex& mutex, std::chrono::milliseconds timeout) : mutex_(mutex), acquired_(mutex_.try_lock_for(timeout)) {}
            ~DelayedLock()
            {
                if(acquired_)
                {
                    mutex_.unlock();
                }
            }
            bool acquired() const { return acquired_; }
            explicit operator bool() const { return acquired_; }
        private:
            Mutex& mutex_;
            bool acquired_;
        };
        // RALL双锁类
        class DoubleLock
        {
        public:
            explicit DoubleLock(Mutex& m1, Mutex& m2) : mutex1_(m1), mutex2_(m2)
            {
                if(&mutex1_ == &mutex2_)
                {
                    mutex1_.lock();
                    same_ = true;
                }
                else
                {
                    if(std::addressof(mutex1_) < std::addressof(mutex2_))
                    {
                        mutex1_.lock();
                        mutex2_.lock();
                    }
                    else
                    {
                        mutex2_.lock();
                        mutex1_.lock();
                    }
                }
            }
            ~DoubleLock()
            {
                if(same_)
                {
                    mutex1_.unlock();
                }
                else
                {
                    mutex1_.unlock();
                    mutex2_.unlock();
                }
            }
            DoubleLock(const DoubleLock&) = delete;
            DoubleLock& operator=(const DoubleLock&) = delete;
            DoubleLock(DoubleLock&&) = delete;
            DoubleLock& operator=(DoubleLock&&) = delete;
        private:
            Mutex& mutex1_;
            Mutex& mutex2_;
            bool same_ = false;
        };
        
        Mutex() = default;
        ~Mutex()
        {
            #ifdef IS_DEBUG
                if(is_locked())
                {
                    std::cerr << "警告：Mutex 析构时仍被锁定，可能导致死锁，当前线程ID：" << owner_id_ << std::endl;
                    while(lock_count_ > 0)
                    {
                        mutex_.unlock();
                        lock_count_--;
                    }
                }
            #endif
        }
        Mutex(const Mutex&) = delete;
        Mutex& operator=(const Mutex&) = delete;
        Mutex(Mutex&&) = delete;
        Mutex& operator=(Mutex&&) = delete;

    private:
        void lock()
        {
            #ifdef IS_DEBUG
                if(owner_id_ == std::this_thread::get_id())
                {
                    throw std::runtime_error("不能在同一线程中重复加锁");
                }
                owner_id_ = std::this_thread::get_id();
                lock_count_++;
                lock_time_ = std::chrono::system_clock::now();
            #endif
            mutex_.lock();
        }
        void unlock()
        {
            #ifdef IS_DEBUG
                if (lock_count_ == 0)
                {
                    throw std::runtime_error("解锁次数超过加锁次数");
                }
                if (owner_id_ != std::this_thread::get_id())
                {
                    std::cerr << "警告：线程 " << std::this_thread::get_id() 
                            << " 尝试解锁由线程 " << owner_id_ << " 持有的锁" << std::endl;
                }
                lock_count_--;
                if(lock_count_ == 0)
                {
                    owner_id_ = std::thread::id();
                }
            #endif
            mutex_.unlock();
        }
        bool try_lock()
        {
            if(mutex_.try_lock())
            {
                #ifdef IS_DEBUG
                    owner_id_ = std::this_thread::get_id();
                    lock_count_++;
                    lock_time_ = std::chrono::system_clock::now();
                #endif
                return true;
            }
            return false;
        }
        bool try_lock_for(std::chrono::milliseconds timeout)
        {
            auto start = std::chrono::system_clock::now();
            while(std::chrono::system_clock::now() - start < timeout)
            {
                if(try_lock()) return true;
                std::this_thread::yield();
            }
            return false;
        }
        #ifdef IS_DEBUG
            bool is_locked() const { return lock_count_ > 0; }
            bool is_locked_by_current_thread() const { return owner_id_ == std::this_thread::get_id(); }
            std::thread::id get_owner() const { return owner_id_; }
            size_t get_lock_count() const { return lock_count_; }
        #endif
        std::mutex mutex_;
        #ifdef IS_DEBUG
            std::atomic<std::thread::id> owner_id_{std::thread::id()};
            std::atomic<size_t> lock_count_{0};
            std::chrono::system_clock::time_point lock_time_{};
        #endif
    };

    // 时间戳类
    class Timestamp
    {
    public:
        using Clock = std::chrono::system_clock;
        static uint64_t now() // 微秒级时间戳
        {
            return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now().time_since_epoch()).count();
        }
        static uint64_t now_ns() // 纳秒级时间戳
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
        }
        static uint64_t now_ms() // 毫秒级时间戳
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
        }
        static int64_t diff(uint64_t newer, uint64_t older)
        {
            if(newer >= older)
            {
                return static_cast<int64_t>(newer - older);
            }
            else
            {
                return static_cast<int64_t>(newer) - static_cast<int64_t>(older);
            }
        }
        static bool is_within_threshold(uint64_t ts1, uint64_t ts2, uint64_t threshold)
        {
            auto difference = diff(ts1 > ts2 ? ts1 : ts2, ts1 > ts2 ? ts2 : ts1);
            return difference <= static_cast<int64_t>(threshold);
        }
        static uint64_t average(uint64_t ts1, uint64_t ts2, float weight = 0.5f)
        {
            return static_cast<uint64_t>(ts1 * weight + ts2 * (1 - weight));
        }
    };

    // 日志类
    class Logger
    {
    public:
        enum class Level
        {
            NONE = 0,
            ERROR, WARNING, INFO, DEBUG
        };
        static Logger& instance()
        {
            static Logger instance_;
            return instance_;
        }
        static void set_level(Level level)
        {
            instance().level_ = level;
        }
        static void set_output_to_cout()
        {
            instance().output_ = &std::cout;
        }
        static void set_output_to_cerr()
        {
            instance().output_ = &std::cerr;
        }
        template<typename... Args>
        static void error(std::string_view fmt, Args&&... args)
        {
            log_impl(Level::ERROR, fmt, std::forward<Args>(args)...);
        }
        template<typename... Args>
        static void warning(std::string_view fmt, Args&&... args)
        {
            log_impl(Level::WARNING, fmt, std::forward<Args>(args)...);
        }
        template<typename... Args>
        static void info(std::string_view fmt, Args&&... args)
        {
            log_impl(Level::INFO, fmt, std::forward<Args>(args)...);
        }
        template<typename... Args>
        static void debug(std::string_view fmt, Args&&... args)
        {
            log_impl(Level::DEBUG, fmt, std::forward<Args>(args)...);
        }

        static void buffer_write(std::string_view buffer_name, size_t size, size_t capacity)
        {
            #ifdef BUFFER_ENABLE_LOGGING
                log_impl(Level::DEBUG, "缓冲区 {} 写入: {}/{} ({:.1f}%)", buffer_name, size, capacity, capacity > 0 ? (size * 100.0 / capacity) : 0.0);
            #endif
        }
        
        static void buffer_read(std::string_view buffer_name, size_t size, size_t capacity)
        {
            #ifdef BUFFER_ENABLE_LOGGING
                log_impl(Level::DEBUG, "缓冲区 {} 读取: {}/{} ({:.1f}%)", buffer_name, size, capacity, capacity > 0 ? (size * 100.0 / capacity) : 0.0);
            #endif
        }
        
        static void buffer_drop(std::string_view buffer_name, std::string_view reason)
        {
            #ifdef BUFFER_ENABLE_LOGGING
                log_impl(Level::WARNING, "缓冲区 {} 丢弃: {}", buffer_name, reason);
            #endif
        }
        
        static void buffer_full(std::string_view buffer_name)
        {
            #ifdef BUFFER_ENABLE_LOGGING
                log_impl(Level::WARNING, "缓冲区 {} 已满", buffer_name);
            #endif
        }
        
        static void buffer_empty(std::string_view buffer_name)
        {
            #ifdef BUFFER_ENABLE_LOGGING
                log_impl(Level::DEBUG, "缓冲区 {} 为空", buffer_name);
            #endif
        }
    private:
        Logger() = default;
        ~Logger() = default;
        
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;
        Logger(Logger&&) = delete;
        Logger& operator=(Logger&&) = delete;

        template<typename... Args>
        static void log_impl(Level level, std::string_view format, Args&&... args)
        {
            #ifdef BUFFER_ENABLE_LOGGING
            if (instance().should_log(level))
            {
                std::ostringstream oss;
                oss << format;
                ((oss << " " << std::forward<Args>(args)), ...);
                instance().write_log(level, oss.str());
            }
            #endif
        }
        bool should_log(Level level) const
        {
            return level <= level_;
        }
        void write_log(Level level, const std::string& message)
        {
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            
            std::tm tm_buf{};
            #ifdef _WIN32
            localtime_s(&tm_buf, &time);
            #else
            localtime_r(&time, &tm_buf);
            #endif
            std::ostringstream oss;
            oss << "[" << std::put_time(&tm_buf, "%H:%M:%S");
            oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
            oss << level_to_string(level) << ": " << message; 
            *output_ << oss.str() << std::endl;
        }
        static constexpr std::string_view level_to_string(Level level)
        {
            switch(level)
            {
                case Level::NONE: return "NONE";
                case Level::ERROR: return "ERROR";
                case Level::WARNING: return "WARNING";
                case Level::INFO: return "INFO";
                case Level::DEBUG: return "DEBUG";
                default: return "UNKNOWN";
            }
        }
        
        Level level_ = Level::INFO;
        std::ostream* output_ = &std::cout;
    };

    /*
     * @brief 缓冲区分配策略
     */
    // 编译期分配策略
    template<typename T, std::size_t N>
    class StaticStorage
    {
    public:
        static_assert(N > 0, "缓冲区容量必须大于0");
        static_assert(std::is_destructible_v<T>, "缓冲区元素类型必须是可析构的");
        constexpr size_t capacity() const noexcept { return N; }
        T* data() noexcept
        {
            return reinterpret_cast<T*>(buffer_);
        }
        const T* data() const noexcept
        {
            return reinterpret_cast<const T*>(buffer_);
        }
    private:
        alignas(T) std::byte buffer_[N * sizeof(T)];
    };
    // 运行期分配策略
    template<typename T>
    class DynamicStorage
    {
    public:
        using Alloc = std::allocator<T>;
        explicit DynamicStorage(std::size_t capacity, const Alloc& alloc = {}) : capacity_(capacity), alloc_(alloc)
        {
            if (capacity == 0) {
                throw std::invalid_argument("缓冲区容量必须大于0");
            }
            buffer_ = alloc_.allocate(capacity_);
        }
        ~DynamicStorage()
        {
            if(buffer_)
            {
                alloc_.deallocate(buffer_, capacity_);
            }
        }

        DynamicStorage(const DynamicStorage&) = delete;
        DynamicStorage& operator=(const DynamicStorage&) = delete;

        DynamicStorage(DynamicStorage&& other) noexcept : capacity_(other.capacity_), buffer_(other.buffer_), alloc_(std::move(other.alloc_))
        {
            other.capacity_ = 0;
            other.buffer_ = nullptr;
        }

        DynamicStorage& operator=(DynamicStorage&& other) noexcept
        {
            if(this != &other)
            {
                if(buffer_)
                {
                    alloc_.deallocate(buffer_, capacity_);
                }
                capacity_ = other.capacity_;
                buffer_ = other.buffer_;
                alloc_ = std::move(other.alloc_);
                other.capacity_ = 0;
                other.buffer_ = nullptr;
            }
            return *this;
        }

        std::size_t capacity() const noexcept { return capacity_; }

        T* data() noexcept { return buffer_; }
        const T* data() const noexcept { return buffer_; }

    private:
        std::size_t capacity_ = 0;
        T* buffer_ = nullptr;
        Alloc alloc_;
    };

    /*
    * @brief 缓冲区类
    */
    // 单缓存池基类
    template<typename T, typename StoragePolicy, OverwritePolicy Policy = OverwritePolicy::RING, bool UseMutex = true>
    class BufferBase
    {
    public:
        using storage_type = StoragePolicy; 
        template<typename... StorageArgs>
        explicit BufferBase(std::string_view name,StorageArgs&&... args) : name_(name), storage_(std::forward<StorageArgs>(args)...), read_index_(0), write_index_(0), size_(0)
        {
            #ifdef BUFFER_ENABLE_LOGGING
                Logger::info("缓冲区 {} 已初始化，容量: {}", name_, storage_.capacity());
            #endif
        }
        ~BufferBase()
        {
            destroy_elements();
        }
        bool write(const T& data)
        {
            if constexpr (UseMutex)
            {
                Mutex::ScopedLock lock(mutex_);
                return write_impl(data);
            }
            else
            {
                return write_impl(data);
            }
        }
        bool write(T&& data)
        {   
            if constexpr (UseMutex)
            {
                Mutex::ScopedLock lock(mutex_);
                return write_impl_move(std::move(data));
            }
            else
            {
                return write_impl_move(std::move(data));
            }
        }
        bool read(T& data)
        {
            if constexpr (UseMutex)
            {
                Mutex::ScopedLock lock(mutex_);
                return read_impl(data);
            }
            else
            {
                return read_impl(data);
            }
        }
        bool try_write(const T& data)
        {
            if constexpr (UseMutex)
            {
                Mutex::TryLock lock(mutex_);
                if(lock.acquired())
                {
                    return write_impl(data);
                }
            }
            else
            {
                return write_impl(data);
            }
            return false;
        }
        bool try_read(T& data)
        {
            if constexpr (UseMutex)
            {
                Mutex::TryLock lock(mutex_);
                if(lock.acquired())
                {
                    return read_impl(data);
                }
            }
            else
            {
                return read_impl(data);
            }
            return false;
        }
        bool write_with_timeout(const T& data, std::chrono::milliseconds timeout)
        {
            if constexpr (UseMutex)
            {
                Mutex::DelayedLock lock(mutex_, timeout);
                if(lock.acquired())
                {
                    return write_impl(data);
                }
            }
            else
            {
                return write_impl(data);
            }
            return false;
        }
        bool read_with_timeout(T& data, std::chrono::milliseconds timeout)
        {
            if constexpr (UseMutex)
            {
                Mutex::DelayedLock lock(mutex_, timeout);
                if(lock.acquired())
                {
                    return read_impl(data);
                }   
            }
            else
            {
                return read_impl(data);
            }
            return false;
        }
        size_t size() const
        {
            if constexpr (UseMutex)
            {
                Mutex::ScopedLock lock(mutex_);
                return size_;
            }
            else
            {
                return size_;
            }
        }
        size_t capacity() const
        {
            return storage_.capacity();
        }
        bool is_empty() const
        {
            if constexpr (UseMutex)
            {
                Mutex::ScopedLock lock(mutex_);
                return size_ == 0;
            }
            else
            {
                return size_ == 0;
            }
        }
        bool has_space() const
        {
            if constexpr (UseMutex)
            {
                Mutex::ScopedLock lock(mutex_);
                return size_ < storage_.capacity();
            }
            else
            {
                return size_ < storage_.capacity();
            }
        }
        #if __cplusplus >= 202002L
            template<std::input_iterator Iterator>
        #else
            template<typename Iterator>
        #endif
        size_t write_batch(Iterator begin, Iterator end)
        {
            using iter_traits = std::iterator_traits<Iterator>;
            static_assert(std::is_same_v<typename iter_traits::iterator_category, std::input_iterator_tag> ||
                          std::is_convertible_v<typename iter_traits::iterator_category, std::input_iterator_tag>,
                          "迭代器必须是输入迭代器");
            if constexpr (UseMutex)
            {
                Mutex::ScopedLock lock(mutex_);
                size_t count = 0;
                for(auto it = begin; it != end; it++)
                {
                    if(!write_impl(*it)) break;
                    count++;
                }
                return count;
            }
            else
            {
                size_t count = 0;
                for(auto it = begin; it != end; it++)
                {
                    if(!write_impl(*it)) break;
                    count++;
                }
                return count;
            }
        }
        #if __cplusplus >= 202002L
            template<std::output_iterator<const T&> OutputIt>
        #else
            template<typename OutputIt>
        #endif
        size_t read_batch(OutputIt output, size_t max_count) // 返回读取的元素数量
        {
            if constexpr (UseMutex)
            {
                Mutex::ScopedLock lock(mutex_);
                size_t count = 0;
                for(size_t i = 0; i < max_count; i++)
                {
                    T data;
                    if(!read_impl(data)) break;
                    *output++ = data;
                    count++;
                }
                return count;
            }
            else
            {
                size_t count = 0;
                for(size_t i = 0; i < max_count; i++)
                {
                    T data;
                    if(!read_impl(data)) break;
                    *output++ = data;
                    count++;
                }
                return count;
            }
        }
        void clear() // 清空缓冲区
        {
            if constexpr (UseMutex)
            {
                Mutex::ScopedLock lock(mutex_);
                if constexpr (!std::is_trivially_destructible_v<T>)
                {
                    destroy_elements();
                }
                write_index_ = 0;
                read_index_ = 0;
                size_ = 0;
                #ifdef BUFFER_ENABLE_LOGGING
                    Logger::info("缓冲区 {} 已清空", name_);
                #endif
            }
            else
            {
                if constexpr (!std::is_trivially_destructible_v<T>)
                {
                    destroy_elements();
                }
                write_index_ = 0;
                read_index_ = 0;
                size_ = 0;
                #ifdef BUFFER_ENABLE_LOGGING
                    Logger::info("缓冲区 {} 已清空", name_);
                #endif
            }
        }
        #if __cplusplus >= 201703L
        std::optional<T> peek(size_t offset = 0) const
        {
            if constexpr (UseMutex)
            {
                Mutex::ScopedLock lock(mutex_);
                return unsafe_peek(offset);
            }
            else
            {
                return unsafe_peek(offset);
            }
        }
        #else
        bool peek(T&out, size_t offset = 0) const
        {
            if constexpr (UseMutex)
            {
                Mutex::ScopedLock lock(mutex_);
                return unsafe_peek(out, offset);
            }
            else
            {
                return unsafe_peek(out, offset);
            }
        }
        #endif
    private:
        BufferBase(const BufferBase&) = delete;
        BufferBase& operator=(const BufferBase&) = delete;
        BufferBase(BufferBase&&) = default;
        BufferBase& operator=(BufferBase&&) = default;

        bool write_impl(const T& data)
        {
            if(size_ >= storage_.capacity())
            {
                if(Policy == OverwritePolicy::LINE)
                {
                    #ifdef BUFFER_ENABLE_LOGGING
                        Logger::buffer_drop(name_, "写入缓冲区已满，新数据被丢弃");
                    #endif
                    return false;
                }
                else
                {
                    std::destroy_at(data_at(read_index_));
                    read_index_ = (read_index_ + 1) % storage_.capacity();
                    #ifdef BUFFER_ENABLE_LOGGING
                        Logger::warning("缓冲区 {} 已溢出，旧数据被覆盖", name_);
                    #endif
                }
            }
            else
            {
                size_++;
            }
            #if __cplusplus >= 202002L
                std::construct_at(data_at(write_index_), data);
            #else
                new (data_at(write_index_)) T(data);
            #endif
            write_index_ = (write_index_ + 1) % storage_.capacity();
            #ifdef BUFFER_ENABLE_LOGGING
                Logger::buffer_write(name_, size_, storage_.capacity());
            #endif
            return true;
        }
        bool write_impl_move(T&& data)
        {
            if(size_ >= storage_.capacity())
            {
                if(Policy == OverwritePolicy::LINE)
                {
                    #ifdef BUFFER_ENABLE_LOGGING
                        Logger::buffer_drop(name_, "写入缓冲区已满，新数据被丢弃");
                    #endif
                    return false;
                }
                else
                {
                    std::destroy_at(data_at(read_index_));
                    read_index_ = (read_index_ + 1) % storage_.capacity();
                    #ifdef BUFFER_ENABLE_LOGGING
                        Logger::warning("缓冲区 {} 已溢出，旧数据被覆盖", name_);
                    #endif
                }
            }
            else
            {
                size_++;
            }
            #if __cplusplus >= 202002L
                std::construct_at(data_at(write_index_), std::move(data));
            #else
                new (data_at(write_index_)) T(std::move(data));
            #endif
            write_index_ = (write_index_ + 1) % storage_.capacity();
            #ifdef BUFFER_ENABLE_LOGGING
                Logger::buffer_write(name_, size_, storage_.capacity());
            #endif
            return true;
        }
        bool read_impl(T& data)
        {
            if(size_ == 0)
            {
                #ifdef BUFFER_ENABLE_LOGGING
                    Logger::buffer_empty(name_);
                #endif
                return false;
            }
            T* element = data_at(read_index_);
            data = std::move(*element);
            std::destroy_at(element);
            read_index_ = (read_index_ + 1) % storage_.capacity();
            size_--;
            #ifdef BUFFER_ENABLE_LOGGING
                Logger::buffer_read(name_, size_, storage_.capacity());
            #endif
            return true;
        }
    private:
        T* data_at(size_t index)
        {
            return &storage_.data()[index % storage_.capacity()];
        }
        const T* data_at(size_t index) const
        {
            return &storage_.data()[index % storage_.capacity()];
        }
        void destroy_elements()
        {
            for(size_t i = 0; i < size_; i++)
            {
                size_t index = (read_index_ + i) % storage_.capacity();
                std::destroy_at(&storage_.data()[index]);
            }
        }
        #if __cplusplus >= 201703L
        std::optional<T> unsafe_peek(size_t offset) const
        {
            if(offset >= size_)
            {
                return std::nullopt;
            }
            size_t idx = (read_index_ + offset) % storage_.capacity();
            return storage_.data()[idx];
        }
        #else
        bool unsafe_peek(T& out, size_t offset) const
        {
            if(offset >= size_)
            {
                return false;
            }
            size_t idx = (read_index_ + offset) % storage_.capacity();
            out = storage_.data()[idx];
            return true;
        }
        #endif
        StoragePolicy storage_;
        size_t read_index_;
        size_t write_index_;
        size_t size_;
        std::string name_;
        [[no_unique_address]] mutable std::conditional_t<UseMutex, Mutex, monostate> mutex_;
    };

    /*
    * @brief 单缓冲接口命名
    */
   template<typename T, size_t N>
   using StaticRingBuffer = BufferBase<T, StaticStorage<T, N>>;
   template<typename T, size_t N>
   using StaticBuffer = BufferBase<T, StaticStorage<T, N>, OverwritePolicy::LINE>;
   template<typename T>
   using DynamicRingBuffer = BufferBase<T, DynamicStorage<T>>;
   template<typename T>
   using DynamicBuffer = BufferBase<T, DynamicStorage<T>, OverwritePolicy::LINE>;
   template<typename T, size_t N>
   using UnlockStaticBuffer = BufferBase<T, StaticStorage<T, N>, OverwritePolicy::LINE, false>;
   template<typename T>
   using UnlockDynamicBuffer = BufferBase<T, DynamicStorage<T>, OverwritePolicy::LINE, false>;
   
   // 双缓存池类
    template<typename T, typename BufferType>
    class DoubleBuffer
    {
    public:
        explicit DoubleBuffer(std::string_view name)
            : front_(std::make_unique<BufferType>(name)), back_(std::make_unique<BufferType>(name)), name_(name), swap_requested_(false), back_buffer_releasable_(true) {}

        template<typename U = BufferType>
        explicit DoubleBuffer(std::string_view name, std::size_t capacity,
            std::enable_if_t<std::is_constructible_v<U, std::string_view, std::size_t>, int> = 0)
            : front_(std::make_unique<BufferType>(name, capacity)), back_(std::make_unique<BufferType>(name, capacity)), name_(name), swap_requested_(false), back_buffer_releasable_(true) {}

        bool write(const T& item)
        {
            if(front_->has_space())
            {
                return front_->write(item);
            }
            flush();
            return front_->write(item);
        }

        bool read(T& item) // 当前读取不掌握主动权
        {
            if(!swap_requested_.load())
            {
                back_buffer_releasable_.store(false);
                if(!swap_requested_.load())
                {
                    bool read_success = back_->read(item);
                    if(read_success)
                    {
                        back_buffer_releasable_.store(true);
                        return true;
                    }
                }
                back_buffer_releasable_.store(true);
                return false;
            }
            else
            {
                back_buffer_releasable_.store(true);
                return false;
            }
        }

        void flush()
        {
            swap_requested_.store(true);
            while(!back_buffer_releasable_.load())
            {
                std::this_thread::yield();
            }
            front_.swap(back_);
            front_->clear();
            swap_requested_.store(false);
        }
    private:
        std::string name_;
        std::unique_ptr<BufferType> front_;
        std::unique_ptr<BufferType> back_;
        std::atomic<bool> swap_requested_;
        std::atomic<bool> back_buffer_releasable_;
    };

    /*
    *@brief 双缓冲区类别名
    */
    // 编译期确定缓冲区大小
    template<typename T, std::size_t N>
    using StaticDoubleBuffer = DoubleBuffer<T, UnlockStaticBuffer<T, N>>;
    // 运行时确定缓冲区大小
    template<typename T>
    using DynamicDoubleBuffer = DoubleBuffer<T, UnlockDynamicBuffer<T>>;

// 单容量无锁缓冲区
template<typename Data>
class DataChannel
{
public:
    DataChannel() = default;
    ~DataChannel()
    {
        if(read_index_.load(std::memory_order_relaxed) != write_index_.load(std::memory_order_relaxed))
        {
            destroy_elements();
        }
    }
    DataChannel(const DataChannel&) = delete;
    DataChannel& operator=(const DataChannel&) = delete;
    DataChannel(DataChannel&&) = delete;
    DataChannel& operator=(DataChannel&&) = delete;
    bool write(const Data& data)
    {
        return write_impl(data);
    }
    bool write(Data&& data)
    {
        return write_impl(std::move(data));
    }
    bool read(Data& data)
    {  
        return read_impl(data);
    }
    bool is_empty() const
    {
        return read_index_.load(std::memory_order_acquire) == write_index_.load(std::memory_order_acquire);
    }
    bool has_data() const
    {
        return !is_empty();
    }
private:
    template<typename U>
    bool write_impl(U&& data) //转发引用
    {
        size_t current_write = write_index_.load(std::memory_order_relaxed);
        size_t current_read = read_index_.load(std::memory_order_acquire);
        if(current_write != current_read)
        {
            return false;
        }
        new (data_ptr()) Data(std::forward<U>(data));                     //placement new。forward完美转发
        write_index_.store(current_write + 1, std::memory_order_release); //release“贴封条”
        return true;
    }
    bool read_impl(Data& data)
    {    
        size_t current_read = read_index_.load(std::memory_order_relaxed);
        size_t current_write = write_index_.load(std::memory_order_acquire); //“确认封条”
        if(current_read == current_write)
        {
            return false;
        }
        Data* element = data_ptr();
        data = *element;
        std::destroy_at(element);  //一块内存上，在同一时刻只能存在一个对象生命周期,以防第二次new出现UB
        read_index_.store(current_read + 1, std::memory_order_release);
        return true;
    }
private:
    Data* data_ptr()
    {
        return reinterpret_cast<Data*>(&buffer_);
    }
    const Data* data_ptr() const
    {
        return reinterpret_cast<const Data*>(&buffer_);
    }
    void destroy_elements()
    {
        size_t current_read = read_index_.load(std::memory_order_relaxed);
        size_t current_write = write_index_.load(std::memory_order_acquire);
        if(current_read != current_write)
        {
            Data* element = data_ptr();
            element->~Data();
        }
    }
    alignas(alignof(Data)) std::byte buffer_[sizeof(Data)];//创建sizeof(Data)个字节大小的字节数组buffer_，把buffer_的首地址放在以alignof(Data)为最小字节对齐的整数倍的地址上
    std::atomic<size_t> read_index_{0};
    std::atomic<size_t> write_index_{0};
};

}

#endif