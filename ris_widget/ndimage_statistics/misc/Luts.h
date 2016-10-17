// The MIT License (MIT)
//
// Copyright (c) 2016 WUSTL ZPLAB
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Authors: Erik Hvatum <ice.rikh@gmail.com>

#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

class SampleLut;
class SampleLuts;

typedef std::vector<std::uint64_t> SampleLutData;
typedef std::shared_ptr<SampleLut> SampleLutPtr;
typedef std::map<std::pair<std::uint64_t, std::uint64_t>, SampleLutPtr> SampleLutCache;
typedef SampleLutCache::iterator SampleLutCacheIt;
typedef std::list<SampleLutCacheIt> SampleLutCacheLru;
typedef SampleLutCacheLru::iterator SampleLutCacheLruIt;

class SampleLut
{
public:
    friend class SampleLuts;

    const std::uint64_t m_fromSampleCount;
    const std::uint64_t m_toSampleCount;
    const SampleLutData m_data;

    SampleLut(const std::uint64_t& fromSampleCount, const std::uint64_t& toSampleCount);

protected:
    SampleLutCacheIt m_lutCacheIt;
    SampleLutCacheLruIt m_lutCacheLruIt;
};

class SampleLuts
{
public:
    explicit SampleLuts(const std::size_t& maxCachedLuts);

    SampleLutPtr getLut(const std::uint64_t& fromSampleCount, const std::uint64_t& toSampleCount);

protected:
    SampleLutCache m_lutCache;
    SampleLutCacheLru m_lutCacheLru;
    std::mutex m_lutCacheMutex;
    std::size_t m_maxCachedLuts;
};

extern SampleLuts sampleLuts;


class PeroneCircleLut;
class PeroneCircleLuts;

typedef std::unique_ptr<const std::int32_t, void(*)(const std::int32_t*)> PeroneCircleLutData;
typedef std::shared_ptr<PeroneCircleLut> PeroneCircleLutPtr;
typedef std::map<std::uint32_t, PeroneCircleLutPtr> PeroneCircleLutCache;
typedef PeroneCircleLutCache::iterator PeroneCircleLutCacheIt;
typedef std::list<PeroneCircleLutCacheIt> PeroneCircleLutCacheLru;
typedef PeroneCircleLutCacheLru::iterator PeroneCircleLutCacheLruIt;

// The magnificent circle algorithm used by this class is a modest modification of the last algorithm listed here in the
// main post (not the comments): http://www.willperone.net/Code/codecircle.php 
class PeroneCircleLut
{
public:
    friend class PeroneCircleLuts;

    const std::uint32_t m_r;
    const PeroneCircleLutData m_y_to_x_data;

    explicit PeroneCircleLut(const std::uint32_t r);

protected:
    PeroneCircleLutCacheIt m_lutCacheIt;
    PeroneCircleLutCacheLruIt m_lutCacheLruIt;
};

class PeroneCircleLuts
{
public:
    explicit PeroneCircleLuts(const std::size_t& maxCachedLuts);

    PeroneCircleLutPtr getLut(const std::uint32_t r);

protected:
    PeroneCircleLutCache m_lutCache;
    PeroneCircleLutCacheLru m_lutCacheLru;
    std::mutex m_lutCacheMutex;
    std::size_t m_maxCachedLuts;
};

extern PeroneCircleLuts peroneCircleLuts;
