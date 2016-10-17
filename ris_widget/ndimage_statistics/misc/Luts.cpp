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

#include "branch_hints.h"
#include "Luts.h"
#include <cmath>
#include <cstdlib>

SampleLut::SampleLut(const std::uint64_t& fromSampleCount, const std::uint64_t& toSampleCount)
  : m_fromSampleCount(fromSampleCount),
    m_toSampleCount(toSampleCount),
    m_data(fromSampleCount, 0)
{
    SampleLutData& data{const_cast<SampleLutData&>(m_data)};
    std::uint64_t* toSampleIt{data.data()};
    const double f{((double)(m_toSampleCount)) / m_fromSampleCount};
    for(std::uint64_t fromSampleNum=0; fromSampleNum < m_fromSampleCount; ++fromSampleNum, ++toSampleIt)
        *toSampleIt = fromSampleNum * f;
}

SampleLuts::SampleLuts(const std::size_t& maxCachedLuts)
  : m_maxCachedLuts(maxCachedLuts)
{
    if(m_maxCachedLuts <= 0)
        throw std::invalid_argument("The value supplied for maxCachedLuts must be > 0.");
}

SampleLutPtr SampleLuts::getLut(const std::uint64_t& fromSampleCount, const std::uint64_t& toSampleCount)
{
    std::lock_guard<std::mutex> lutCacheLock(m_lutCacheMutex);
    if(fromSampleCount == 0 || toSampleCount == 0)
        throw std::invalid_argument("The values supplied for fromSampleCount and toSampleCount must be > 0.");
    std::pair<std::uint64_t, std::uint64_t> key(fromSampleCount, toSampleCount);
    SampleLutCacheIt lutCacheIt{m_lutCache.find(key)};
    if(lutCacheIt == m_lutCache.end())
    {
        SampleLutPtr lut(new SampleLut(fromSampleCount, toSampleCount));
        lutCacheIt = lut->m_lutCacheIt = m_lutCache.insert(
            std::pair<std::pair<std::uint64_t, std::uint64_t>, SampleLutPtr>(
                std::pair<std::uint64_t, std::uint64_t>(fromSampleCount, toSampleCount), lut
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
        SampleLut& lut{*lutCacheIt->second};
        if(lut.m_lutCacheLruIt != m_lutCacheLru.begin())
        {
            m_lutCacheLru.erase(lut.m_lutCacheLruIt);
            m_lutCacheLru.push_front(lut.m_lutCacheIt);
            lut.m_lutCacheLruIt = m_lutCacheLru.begin();
        }
    }
    return lutCacheIt->second;
}

SampleLuts sampleLuts{256};



PeroneCircleLut::PeroneCircleLut(const std::uint32_t r)
  : m_r(r),
    m_y_to_x_data(
       static_cast<std::int32_t*>
       (
          reinterpret_cast<std::int32_t*>
          (
             malloc((r*2 + 1)*sizeof(std::int32_t))
          )
       ),
       [](const std::int32_t* v){free(const_cast<std::int32_t*>(v));}
    )
{
    std::int32_t* lut = const_cast<std::int32_t*>(m_y_to_x_data.get());
    std::uint32_t x=r, y=0;
    std::int32_t cd2=0;

    lut[0] = 0;
    lut[r] = r;
    lut[r*2] = 0;

    // NB: The lut[r-x] = y and lut[r+x] = y assignments replace previously written values a number of times. The
    // correct value is always the last written in any sequence of consecutive overwrites because y never decreases, so
    // that's OK.
    while(x > y)
    {
        cd2 -= --x - ++y;
        if(cd2 < 0)
            cd2 += x++;
//      else
//      {
//          lut[r-x] = y;
//          lut[r+x] = y;
//      }
//      lut[r-y] = x;
//      lut[r+y] = x;
        lut[r-x] = y;
        lut[r-y] = x;
        lut[r+x] = y;
        lut[r+y] = x;
    }

}

PeroneCircleLuts::PeroneCircleLuts(const std::size_t& maxCachedLuts)
  : m_maxCachedLuts(maxCachedLuts)
{
    if(m_maxCachedLuts <= 0)
        throw std::invalid_argument("The value supplied for maxCachedLuts must be > 0.");
}

PeroneCircleLutPtr PeroneCircleLuts::getLut(const std::uint32_t r)
{
    std::lock_guard<std::mutex> lutCacheLock(m_lutCacheMutex);
    PeroneCircleLutCacheIt lutCacheIt{m_lutCache.find(r)};
    if(lutCacheIt == m_lutCache.end())
    {
        PeroneCircleLutPtr lut(new PeroneCircleLut(r));
        lutCacheIt = lut->m_lutCacheIt = m_lutCache.insert(std::pair<std::uint32_t, PeroneCircleLutPtr>(r, lut)).first;
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
        PeroneCircleLut& lut{*lutCacheIt->second};
        if(lut.m_lutCacheLruIt != m_lutCacheLru.begin())
        {
            m_lutCacheLru.erase(lut.m_lutCacheLruIt);
            m_lutCacheLru.push_front(lut.m_lutCacheIt);
            lut.m_lutCacheLruIt = m_lutCacheLru.begin();
        }
    }
    return lutCacheIt->second;
}

PeroneCircleLuts peroneCircleLuts{256};
