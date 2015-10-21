# The MIT License (MIT)
#
# Copyright (c) 2015 WUSTL ZPLAB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors: Erik Hvatum <ice.rikh@gmail.com>

from PyQt5 import Qt
from .signaling_list import SignalingList

class UniformSignalingList(SignalingList):
    '''
    UniformSignalingList: A SignalingList whose elements satisfy constraints imposed by
    the UniformSignalingList derivative's take_input_element method.

    For example...

    from ris_widget.signaling_list import UniformSignalingList
    class USL(UniformSignalingList):
        def take_input_element(self, obj):
            return float(obj)
    usl = USL([1,2,3,4,5])
    print(usl)
     # <__main__.USL object at 0x7f49888c4318
     # [
     #     1.0,
     #     2.0,
     #     3.0,
     #     4.0,
     #     5.0
     # ]>
    usl.append(7)
    print(usl)
     # <__main__.USL object at 0x7f49888c4318
     # [
     #     1.0,
     #     2.0,
     #     3.0,
     #     4.0,
     #     5.0,
     #     7.0
     # ]>
    usl.append('8')
    print(usl)
     # <__main__.USL object at 0x7f49888c4318
     # [
     #     1.0,
     #     2.0,
     #     3.0,
     #     4.0,
     #     5.0,
     #     7.0,
     #     8.0
     # ]>
    usl.append('?what')
     # ---------------------------------------------------------------------------
     # ValueError                                Traceback (most recent call last)
     # <ipython-input-4-ef4872d597a8> in <module>()
     # ----> 1 usl.append('?what')
     #
     # /usr/lib64/python3.4/_collections_abc.py in append(self, value)
     #     706     def append(self, value):
     #     707         'S.append(value) -- append value to the end of the sequence'
     # --> 708         self.insert(len(self), value)
     #     709 
     #     710     def clear(self):
     #
     # /home/ehvatum/zplrepo/ris_widget/ris_widget/om/signaling_list/uniform_signaling_list.py in insert(self, idx, obj)
     #      45 
     #      46     def insert(self, idx, obj):
     # ---> 47         super().insert(idx, self.take_input_element(obj))
     #      48     insert.__doc__ = SignalingList.insert.__doc__
     #      49 
     #
     # <ipython-input-2-8c035dfe6f8a> in take_input_element(self, obj)
     #       1 class USL(UniformSignalingList):
     #       2     def take_input_element(self, obj):
     # ----> 3         return float(obj)
     #       4 
     #
     #   ValueError: could not convert string to float: '?what' '''

    def __init__(self, iterable=None, parent=None):
        super().__init__(iterable=map(self.take_input_element, iterable), parent=parent)

    def take_input_element(self, obj):
        raise NotImplementedError()

    def __setitem__(self, idx_or_slice, srcs):
        super().__setitem__(idx_or_slice, map(self.take_input_element, srcs))
    __setitem__.__doc__ = SignalingList.__setitem__.__doc__

    def extend(self, srcs):
        super().extend(map(self.take_input_element, srcs))
    extend.__doc__ = SignalingList.extend.__doc__

    def insert(self, idx, obj):
        super().insert(idx, self.take_input_element(obj))
    insert.__doc__ = SignalingList.insert.__doc__

UniformSignalingList.__doc__ = UniformSignalingList.__doc__ + '\n\n' + SignalingList.__doc__
