// The MIT License (MIT)
//
// Copyright (c) 2014 Erik Hvatum
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

#include "Common.h"
#include "SharedGlObjects.h"
#include "View.h"

SharedGlObjects::SharedGlObjects()
  : histoCalcProg("histoCalcProg"),
    histoConsolidateProg("histoConsolidateProg"),
    imageDrawProg("imageDrawProg"),
    histoDrawProg("histoDrawProg"),
    image(std::numeric_limits<GLuint>::max()),
    imageSize(std::numeric_limits<int>::min(), std::numeric_limits<int>::min()),
    histogramIsStale(true),
    histogramDataStructuresAreStale(true),
    histogramBinCount(2048),
    histogramBlocks(std::numeric_limits<GLuint>::max()),
    histogram(std::numeric_limits<GLuint>::max()),
    histogramData(histogramBinCount, 0),
    histogramPmv(1.0f),
    m_inited(false)
{
}

void SharedGlObjects::init(View* imageView, View* histogramView)
{
    if(m_inited)
    {
        throw RisWidgetException("void SharedGlObjects::init(View* histogramView, View* imageView): Called multiple "
                                 "times for the same SharedGlObjects instance.");
    }

    histoCalcProg.setView(histogramView);
    histoCalcProg.build();

    histoConsolidateProg.setView(histogramView);
    histoConsolidateProg.build();

    imageDrawProg.setView(imageView);
    imageDrawProg.build();

    histoDrawProg.setView(histogramView);
    histoDrawProg.build();

    m_inited = true;
}
