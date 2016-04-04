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

#include <algorithm>
#include "_cpp_image.h"

_CppImage::_CppImage(const char* s)
{
    if(s) m_name = s;
}

const ImageStatus& _CppImage::status() const
{
    return m_status;
}

void _CppImage::setStatus(const ImageStatus& status)
{
    if(status != m_status)
    {
        m_status = status;
        call(&_CppImage::m_statusChangedCallbacks);
    }
}

void _CppImage::addStatusChangedCallback(const Callback& callback)
{
}

void _CppImage::removeStatusChangedCallback(const Callback& callback)
{
}

bool _CppImage::hasStatusChangedCallback(const Callback& callback) const
{
    bool ret{false};
    for(const Callback& currCallback : m_statusChangedCallbacks)
    {
        if(currCallback.target<void()>() == callback.target<void()>())
        {
            ret = true;
            break;
        }
    }
    return ret;
}

void _CppImage::call(const CallbackListMPtr& callbackListMPtr)
{
    for(Callback& callback : this->*callbackListMPtr) callback();
}
