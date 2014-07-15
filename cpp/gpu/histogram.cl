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

// Stuffing these read-only arguments into a struct allows them to be passed as a pointer to magically fast global
// read-only memory
typedef struct _xxBlocksConstArgs
{
    const uint2 imageSize;
    // Used only by computeBlocks(..)
    const uint2 invocationRegionSize;
    const uint binCount;
    const uint paddedBlockSize;
}
XxBlocksConstArgs;

// Note that ComputeBlocksConstArgs must be passed as a pointer.  If not passed as a pointer, it will instead gobble up
// thread local registers (this is what happens when each thread AKA work item receiving its own private duplicate of
// the struct).  The constant qualifier in OpenCL exists for _exactly_this_situation_, where a function depends on a
// bunch of read-only variables that are read often and also requires lots of writeable registers to work efficiently
// (as most functions do!).
kernel void computeBlocks(constant XxBlocksConstArgs* args,
                          read_only image2d_t image,
                          global uint16* blocks,
                          local uint* block,
                          global uint* zeroBlock)
{
    // This seems to be the very best way to fill block stored in local memory with zeros.  It is essential to do so on
    // devices that do not provide zeroed local buffers (otherwise, the computed block histogram will be rubbish).
    event_t copyEvent = async_work_group_copy(block, zeroBlock, args->binCount, 0);

    uint2 start = args->invocationRegionSize * (uint2)(get_global_id(0), get_global_id(1));
    // Ensure that processing of remainder blocks for non-divisible images does not proceed beyond image edge
    uint2 end = (uint2)(min(start.x + args->invocationRegionSize.x, args->imageSize.x),
                        min(start.y + args->invocationRegionSize.y, args->imageSize.y));

    float intensity;
    uint bin;
    wait_group_events(1, &copyEvent);
    for(uint y = start.y, x; y < end.y; ++y)
    {
        for(x = start.x; x < end.x; ++x)
        {
            intensity = read_imagef(image, sampler, (int2)(x, y)).x;
            bin = clamp((uint)(ceil(intensity * args->binCount) - 1), (uint)0, (uint)65535);
            atom_inc(block + bin);
        }
    }

    // Wait until all invocations in the work group have finished running the loop above
    barrier(CLK_LOCAL_MEM_FENCE);

    // Copy block histogram to global memory for reduction
    const size_t idx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * args->paddedBlockSize;
    copyEvent = async_work_group_copy(((global uint*)blocks) + idx,
                                      block,
                                      args->binCount,
                                      0);
    wait_group_events(1, &copyEvent);
}

kernel void reduceBlocks(constant XxBlocksConstArgs* args,
                         const uint invocationBinCount,
                         global uint16* blocks)
{
    if(get_global_id(0) == 0 && get_global_id(1) == 0)
    {
        return;
    }
    global const uint* sbin = ((global const uint*)blocks) + (get_global_id(1) * 8 + get_global_id(0)) * args->paddedBlockSize;
//  global const uint* sbin = (global const uint*)(blocks + (get_global_id(1) * 8 + get_global_id(0)));
    for ( global uint *bin = (global uint*)blocks, *binsEnd = ((global uint*)blocks) + args->binCount;
          bin != binsEnd;
          ++bin, ++sbin )
    {
        atomic_add(bin, *sbin);
//      atomic_inc(bin);
    }
//  global uint* b = (global uint*)blocks;
//  uint o = atomic_inc(b);
//  b[o] = get_global_id(1) * 8 + get_global_id(0);
//  if(get_global_id(0) == get_global_id(1) == 0)
//  {
//      return;
//  }
//  global const uint16* sblock = blocks + (get_global_id(1) * 8 + get_global_id(0));
//  global const uint* sblockc;
//  global const uint* sblockce;
//  global const uint16* sblockEnd = sblock + (args->binCount / 16);
//  global uint16* dblock = blocks;
//  global uint* dblockc;
//  for(;;)
//  {
//      for(sblockc = (const global uint*)sblock, sblockce = (const global uint*)sblock + 16, dblockc = (global uint*)dblock; sblockc != sblockce; ++sblockc, ++dblockc)
//      {
//          atomic_add(dblockc, *sblockc);
//      }
//      ++sblock;
//      if(sblock >= sblockEnd) break;
//      ++dblock;
//  }
}
