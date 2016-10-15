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


class BresenhamCircleLut;
class BresenhamCircleLuts;

typedef std::vector<std::uint64_t> BresenhamCircleLutData;
typedef std::shared_ptr<BresenhamCircleLut> BresenhamCircleLutPtr;
typedef std::map<std::pair<std::uint64_t, std::uint64_t>, BresenhamCircleLutPtr> BresenhamCircleLutCache;
typedef BresenhamCircleLutCache::iterator BresenhamCircleLutCacheIt;
typedef std::list<BresenhamCircleLutCacheIt> BresenhamCircleLutCacheLru;
typedef BresenhamCircleLutCacheLru::iterator BresenhamCircleLutCacheLruIt;

class BresenhamCircleLut
{
public:
    friend class BresenhamCircleLuts;

    const std::uint64_t m_fromBresenhamCircleCount;
    const std::uint64_t m_toBresenhamCircleCount;
    const BresenhamCircleLutData m_data;

    BresenhamCircleLut(const std::uint64_t& fromBresenhamCircleCount, const std::uint64_t& toBresenhamCircleCount);

protected:
    BresenhamCircleLutCacheIt m_lutCacheIt;
    BresenhamCircleLutCacheLruIt m_lutCacheLruIt;
};

class BresenhamCircleLuts
{
public:
    explicit BresenhamCircleLuts(const std::size_t& maxCachedLuts);

    BresenhamCircleLutPtr getLut(const std::uint64_t& fromBresenhamCircleCount, const std::uint64_t& toBresenhamCircleCount);

protected:
    BresenhamCircleLutCache m_lutCache;
    BresenhamCircleLutCacheLru m_lutCacheLru;
    std::mutex m_lutCacheMutex;
    std::size_t m_maxCachedLuts;
};

extern BresenhamCircleLuts bresenhamCircleLuts;
