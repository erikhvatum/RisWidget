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

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void computeBlocks(read_only image2d_t image,
                          uint binCount,
                          read_only uint2 invocationRegionSize,
                          uint paddedBlockSize,
                          global uint* zeroBlock,
                          local uint* block,
                          global uint16* blocks)
{
    // This seems to be the very best way to fill block stored in local memory with zeros.  It is essential to do so on
    // devices that do not provide zeroed local buffers (otherwise, the computed block histogram will be rubbish).
    event_t copyEvent = async_work_group_copy(block, zeroBlock, binCount, 0);

    uint2 start = invocationRegionSize * (uint2)(get_global_id(0), get_global_id(1));
    uint2 size = (uint2)(get_image_width(image), get_image_height(image));
    // Ensure that processing of remainder blocks for non-divisible images does not proceed beyond image edge
    uint2 end = (uint2)(min(start.x + invocationRegionSize.x, size.x),
                        min(start.y + invocationRegionSize.y, size.y));

    float intensity;
    uint bin;
    wait_group_events(1, &copyEvent);
    for(uint y = start.y, x; y < end.y; ++y)
    {
        for(x = start.x; x < end.x; ++x)
        {
            intensity = read_imagef(image, sampler, (int2)(x, y)).x;
            bin = (uint)(floor(intensity * (binCount - 1)));
            atomic_inc(block + bin);
        }
    }

    // Wait until all invocations in the work group have finished running the loop above
    write_mem_fence(CLK_LOCAL_MEM_FENCE);

    // Copy block histogram to global memory for reduction
    copyEvent = async_work_group_copy(((global uint*)blocks) + (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * paddedBlockSize,
                                      block,
                                      binCount,
                                      0);
    wait_group_events(1, &copyEvent);
}

kernel void reduceBlocks(global int* blah)
{
    ++(*blah);
}
