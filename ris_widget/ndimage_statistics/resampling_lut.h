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
#include <stdexcept>
#include <vector>

class Lut;
class Luts;

typedef std::vector<std::uint32_t> LutData;
typedef std::shared_ptr<Lut> LutPtr;
typedef std::map<std::pair<std::uint32_t, std::uint32_t>, LutPtr> LutCache;
typedef LutCache::iterator LutCacheIt;
typedef std::list<LutCacheIt> LutCacheLru;
typedef LutCacheLru::iterator LutCacheLruIt;

class Lut
{
public:
    friend class Luts;

    const std::uint32_t m_fromSampleCount;
    const std::uint32_t m_toSampleCount;
    const LutData m_data;

    Lut(const std::uint32_t& fromSampleCount, const std::uint32_t& toSampleCount);
    ~Lut();

protected:
    LutCacheIt m_lutCacheIt;
    LutCacheLruIt m_lutCacheLruIt;
};

class Luts
{
public:
    explicit Luts(const std::size_t& maxCachedLuts);

    const LutPtr& getLut(const std::uint32_t& fromSampleCount, const std::uint32_t& toSampleCount);

protected:
    LutCache m_lutCache;
    LutCacheLru m_lutCacheLru;

    std::size_t m_maxCachedLuts;
};
