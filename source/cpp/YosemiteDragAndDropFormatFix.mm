// The MIT License (MIT)
// 
// Copyright (c) 2014 WUSTL ZPLAB
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
// Authors: Erik Hvatum

#include <string>
#import <Foundation/Foundation.h>

// Apple decided that, from 10.10 on, they are no longer capable of supplying file names with paths for drag and drop
// events.  It was just too hard to do without sacrificing performance.  Assembling a list of strings is
// difficult/impossible for Apple, on a CPU that can pack a million words into a linked list in one millisecond.  This
// is yet another idiotic, totally unecessary, will-never-fix problem that Apple has _added_ to OS X over the years in
// their comprehensive effort to undo every last good thing they inherited from FreeBSD and NeXT.  They already got the
// low hanging fruit by tearing out FFS, natD, and the firewall, and by refusing to accept upstream ZFS patches in order
// to make their own ZFS implementation (which failed), and by adopting a microkernel architecture that somehow makes
// threads consume two megabytes of kernel memory rather than two kilobytes, and by trashing Samba in favor of their
// own, broken implementation made from intentionally erroneous specifications supplied by Microsoft, amongst many,
// many, many other things... Even software made entirey by Apple is not immune to the general degedration: consider the
// notorious case Final Cut Pro (ok, they did acquire and incorporate the industry best color matching and pallete
// plugin, only to regress back to something even worse than what they originally replaced.  That plugin, incidentally,
// cost some thousands of dollars per user to license before Apple bought it and threw it away.)
void deref_yosemite_annoying_useless_path_ref_string(const std::string& useless_shit_from_apple, std::string& useful_fpath_for_you)
{
    @autoreleasepool
    {
        NSString* ns_useless_shit_from_apple = [NSString stringWithUTF8String: useless_shit_from_apple.c_str()];
        NSString* ns_useful_fpath_for_you = [[[NSURL URLWithString: ns_useless_shit_from_apple] filePathURL] path];
        useful_fpath_for_you = [ns_useful_fpath_for_you UTF8String];
    }
}
