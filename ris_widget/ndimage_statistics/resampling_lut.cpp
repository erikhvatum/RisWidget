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

#include "resampling_lut.h"
#include <cmath>

Lut::Lut(const std::uint32_t& fromSampleCount, const std::uint32_t& toSampleCount)
  : m_fromSampleCount(fromSampleCount),
    m_toSampleCount(toSampleCount),
    m_data(fromSampleCount, 0)
{
    LutData& data{const_cast<LutData&>(m_data)};
    std::uint32_t* toSampleIt{data.data()};
    const double f{((double)(m_toSampleCount)) / m_fromSampleCount};
    for(std::uint32_t fromSampleNum=0; fromSampleNum < m_fromSampleCount; ++fromSampleNum, ++toSampleIt)
        *toSampleIt = fromSampleNum * f;
}

Luts::Luts(const std::size_t& maxCachedLuts)
  : m_maxCachedLuts(maxCachedLuts)
{
    if(m_maxCachedLuts <= 0)
        throw std::invalid_argument("The value supplied for maxCachedLuts must be > 0.");
}

LutPtr Luts::getLut(const std::uint32_t& fromSampleCount, const std::uint32_t& toSampleCount)
{
    std::lock_guard<std::mutex> lutCacheLock(m_lutCacheMutex);
    if(fromSampleCount == 0 || toSampleCount == 0)
        throw std::invalid_argument("The values supplied for fromSampleCount and toSampleCount must be > 0.");
    std::pair<std::uint32_t, std::uint32_t> key(fromSampleCount, toSampleCount);
    LutCacheIt lutCacheIt{m_lutCache.find(key)};
    if(lutCacheIt == m_lutCache.end())
    {
        LutPtr lut(new Lut(fromSampleCount, toSampleCount));
        lutCacheIt = lut->m_lutCacheIt = m_lutCache.insert(
            std::pair<std::pair<std::uint32_t, std::uint32_t>, LutPtr>(
                std::pair<std::uint32_t, std::uint32_t>(fromSampleCount, toSampleCount), lut
            )
        ).first;
        m_lutCacheLru.push_front(lut->m_lutCacheIt);
        lut->m_lutCacheLruIt = m_lutCacheLru.begin();
        if(m_lutCacheLru.size() > m_maxCachedLuts)
        {

            m_lutCache.erase(m_lutCacheLru.back()->second->m_lutCacheIt);
            m_lutCacheLru.pop_back();
        }
    }
    else
    {
        Lut& lut{*lutCacheIt->second};
        if(lut.m_lutCacheLruIt != m_lutCacheLru.begin())
        {
            m_lutCacheLru.erase(lut.m_lutCacheLruIt);
            m_lutCacheLru.push_front(lut.m_lutCacheIt);
            lut.m_lutCacheLruIt = m_lutCacheLru.begin();
        }
    }
    return lutCacheIt->second;
}

