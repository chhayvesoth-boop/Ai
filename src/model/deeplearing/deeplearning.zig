const std=@import("std");
const DBMS=@import("../../DBMS/DBMS.zig");
const kernel =@import("../../config/config.zig").loadkernel;
const matrixGpu =@import("../../config/ptx/matrix.zig");

pub fn run() !void {

    var buff_in:[1024]u8=undefined;
    var buff_out:[1024]u8=undefined;
    const io = std.Io.Threaded.global_single_threaded.io();

    var stdin_reader=std.Io.File.stdin().readerStreaming(io,&buff_in);// reader data in bus
    var stdout_writer=std.Io.File.stdout().writerStreaming(io,&buff_out);// write data to bus
    const stdin: *std.Io.Reader = &stdin_reader.interface;//take reader fn
    const stdout: *std.Io.Writer = &stdout_writer.interface;//take writer fn

    try stdout.print("what in your mind today:  ",.{});
    try stdout.flush();

    const rawdata= try stdin.takeDelimiterExclusive('\n');
    const inputFromUser=stdin.
    try stdout.print("{s} hello",.{inputFromUser});
    try stdout.flush();

    
    //load gpu
    // var loadGpuMem =try kernel.memorymenagement.load();
    // defer loadGpuMem.deinit();
    // var loadGpuPoc =try kernel.processergpu.load();
    // defer loadGpuPoc.deinit();
    // var loadGpuAv = try kernel.checkaviable.load();
    // defer loadGpuAv.deinit();

    // var device: kernel.CUdevice = 0;
    // detectresult(loadGpuAv.cuDeviceGet(&device, 0));

    // var context: kernel.CUcontext = null;
    // detectresult(loadGpuMem.cuCtxCreate(&context, 0, device));
    // defer _ = loadGpuMem.cuCtxDestroy(context);

    // const shape = [_]usize{ 2, 3 };
    // const mat = matrixGpu.NDMatrix.init(&shape);
    // const total_elements: u32 = @intCast(mat.totalElements()); // 6
    // const byte_size: usize = total_elements * @sizeOf(f32);
    // var ptx_buff: [4096]u8 = @splat(0);
    // // var CUresult:kernel.CUresult=undefined;

    // var host_A = [2][3]f32{
    //     .{ 1.0, 2.0, 3.0 },
    //     .{ 4.0, 5.0, 6.0 },
    // };
    // var host_B = [2][3]f32{
    //     .{ 10.0, 20.0, 30.0 },
    //     .{ 40.0, 50.0, 60.0 },
    // };
    // var host_out = [2][3]f32{
    //     .{ 0.0, 0.0, 0.0 },
    //     .{ 0.0, 0.0, 0.0 },
    // };

    // var dev_A: kernel.CUdeviceptr = 0;
    // var dev_B: kernel.CUdeviceptr = 0;
    // var dev_out: kernel.CUdeviceptr = 0;

    // _ = loadGpuMem.cuMemAlloc(&dev_A, byte_size);
    // defer _ = loadGpuMem.cuMemFree(dev_A);
    // _ = loadGpuMem.cuMemAlloc(&dev_B, byte_size);
    // defer _ = loadGpuMem.cuMemFree(dev_B);
    // _ = loadGpuMem.cuMemAlloc(&dev_out, byte_size);
    // defer _ = loadGpuMem.cuMemFree(dev_out);

    // _ = loadGpuMem.cuMemcpyHtoD(dev_A, @ptrCast(&host_A), byte_size);
    // _ = loadGpuMem.cuMemcpyHtoD(dev_B, @ptrCast(&host_B), byte_size);

    // const buildkernel=try mat.buildDynamicKernel(&ptx_buff,.mul);
    // // ptx_buff[buildkernel.len] = 0;

    // var module: kernel.CUmodule = null;

    // detectresult(loadGpuPoc.cuModuleLoadData(&module, buildkernel.ptx.ptr));
    // defer _ = loadGpuPoc.cuModuleUnload(module);

    // var kernel_fn: kernel.CUfunction = null;
    // detectresult(loadGpuPoc.cuModuleGetFunction(&kernel_fn, module, buildkernel.kername));

    // var kernel_params = [_]?*anyopaque{
    //     @ptrCast(&dev_A),          // -> .param .u64 param_in_A
    //     @ptrCast(&dev_B),          // -> .param .u64 param_in_B
    //     @ptrCast(&dev_out),        // -> .param .u64 param_out
    //     @constCast(&total_elements), // -> .param .u32 total_elements
    // };

    // const threads_per_block: c_uint = 256;
    // const blocks_per_grid: c_uint = (total_elements + threads_per_block - 1) / threads_per_block;

    // detectresult(loadGpuPoc.cuLaunchKernel(
    //     kernel_fn,
    //     blocks_per_grid, 1, 1,   // Grid X, Y, Z
    //     threads_per_block, 1, 1, // Block X, Y, Z
    //     0,                       // Shared memory bytes
    //     null,                    // Default stream
    //     @ptrCast(&kernel_params[0]), // Matches ?*?*anyopaque
    //     null,                    // Extra options
    // ));

    // detectresult(loadGpuMem.cuMemcpyDtoH(@ptrCast(&host_out), dev_out, byte_size));

    // std.debug.print("Result from GPU:\n{any}\n", .{host_out});


}

pub fn detectresult(res: anytype) void {
        if (res == .CUDA_SUCCESS) { // or res == 0 if CUresult is an integer
            std.debug.print("CUDA Success!\n", .{});
        } else {
            std.debug.print("CUDA Error: {any}\n", .{res});
        }
}
pub fn addweight( ) void {
    
}
