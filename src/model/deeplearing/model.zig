const std = @import("std");
const DBMS = @import("../../DBMS/DBMS.zig");
const kernel = @import("../../config/config.zig").loadkernel;
const op_mod = @import("../../config/ptx/operation.zig");

pub const AnswerModel = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    db: *DBMS.RelationDB,
    learning_rate: f32 = 0.00005,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, db: *DBMS.RelationDB) AnswerModel {
        return .{
            .allocator = allocator,
            .io = io,
            .db = db,
        };
    }

    /// Loads relation data from disk, trains weights on the GPU, and persists learned weights back to disk
    pub fn trainRelation(self: *AnswerModel, relation_id: u64, feature_len: usize, epochs: usize) !void {
        const total_elements: u32 = @intCast(feature_len);
        const byte_size: usize = feature_len * @sizeOf(f32);

        // 1. Load Input A, Target B, and current Weights W from Matrix DB
        const host_A = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(host_A);
        const host_B = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(host_B);
        const host_W = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(host_W);

        try self.db.inputs.getRow(self.io, relation_id, host_A, 0.0);
        try self.db.targets.getRow(self.io, relation_id, host_B, 0.0);
        try self.db.weights.getRow(self.io, relation_id, host_W, 0.1);

        // 2. Setup GPU Context
        var gpu_av = try kernel.checkaviable.load();
        defer gpu_av.deinit();
        var gpu_mem = try kernel.memorymenagement.load();
        defer gpu_mem.deinit();
        var gpu_proc = try kernel.processergpu.load();
        defer gpu_proc.deinit();

        checkCuda(gpu_av.cuInit(0));
        var device: kernel.CUdevice = 0;
        checkCuda(gpu_av.cuDeviceGet(&device, 0));
        var context: kernel.CUcontext = null;
        checkCuda(gpu_mem.cuCtxCreate(&context, 0, device));
        defer _ = gpu_mem.cuCtxDestroy(context);

        // 3. Allocate & Upload to GPU VRAM
        var dev_A: kernel.CUdeviceptr = 0;
        var dev_B: kernel.CUdeviceptr = 0;
        var dev_W: kernel.CUdeviceptr = 0;
        var dev_out: kernel.CUdeviceptr = 0;

        checkCuda(gpu_mem.cuMemAlloc(&dev_A, byte_size));
        defer _ = gpu_mem.cuMemFree(dev_A);
        checkCuda(gpu_mem.cuMemAlloc(&dev_B, byte_size));
        defer _ = gpu_mem.cuMemFree(dev_B);
        checkCuda(gpu_mem.cuMemAlloc(&dev_W, byte_size));
        defer _ = gpu_mem.cuMemFree(dev_W);
        checkCuda(gpu_mem.cuMemAlloc(&dev_out, byte_size));
        defer _ = gpu_mem.cuMemFree(dev_out);

        checkCuda(gpu_mem.cuMemcpyHtoD(dev_A, @ptrCast(host_A.ptr), byte_size));
        checkCuda(gpu_mem.cuMemcpyHtoD(dev_B, @ptrCast(host_B.ptr), byte_size));
        checkCuda(gpu_mem.cuMemcpyHtoD(dev_W, @ptrCast(host_W.ptr), byte_size));

        // 4. Generate & Load Gradient Descent PTX Kernel
        const gen = op_mod.KernelGenerator.init(32);
        var ptx_buff: [4096]u8 = @splat(0);
        const built = try gen.emitTrainKernel(&ptx_buff, "model_rel", self.learning_rate);

        var module: kernel.CUmodule = null;
        checkCuda(gpu_proc.cuModuleLoadData(&module, built.ptx.ptr));
        defer _ = gpu_proc.cuModuleUnload(module);

        var kernel_fn: kernel.CUfunction = null;
        checkCuda(gpu_proc.cuModuleGetFunction(&kernel_fn, module, built.kername.ptr));

        var params = [_]?*anyopaque{
            @ptrCast(&dev_A),
            @ptrCast(&dev_B),
            @ptrCast(&dev_W),
            @ptrCast(&dev_out),
            @constCast(&total_elements),
        };

        const threads: c_uint = 256;
        const blocks: c_uint = (total_elements + threads - 1) / threads;

        for (0..epochs) |_| {
            checkCuda(gpu_proc.cuLaunchKernel(
                kernel_fn,
                blocks, 1, 1,
                threads, 1, 1,
                0, null,
                @ptrCast(&params[0]),
                null,
            ));
        }

        // 5. Copy Learned Weights back from GPU and Save to Disk Matrix!
        checkCuda(gpu_mem.cuMemcpyDtoH(@ptrCast(host_W.ptr), dev_W, byte_size));
        try self.db.weights.setRow(self.io, relation_id, host_W);
    }

    /// Trains `num_rows` simultaneously on the GPU (num_rows * cols total elements)
    pub fn trainAllRowsBatch(self: *AnswerModel, num_rows: u64, epochs: usize) !void {
        if (num_rows == 0) return;

        const cols: usize = @intCast(self.db.cols);
        const total_floats: usize = @as(usize, @intCast(num_rows)) * cols;
        const total_elements: u32 = @intCast(total_floats);
        const byte_size: usize = total_floats * @sizeOf(f32);

        // 1. Allocate flat host buffers for the entire dataset
        const host_A = try self.allocator.alloc(f32, total_floats);
        defer self.allocator.free(host_A);
        const host_B = try self.allocator.alloc(f32, total_floats);
        defer self.allocator.free(host_B);
        const host_W = try self.allocator.alloc(f32, total_floats);
        defer self.allocator.free(host_W);

        // Read all rows from disk into contiguous arrays
        for (0..@as(usize, @intCast(num_rows))) |r| {
            const start = r * cols;
            try self.db.inputs.getRow(self.io, r, host_A[start .. start + cols], 32.0);
            try self.db.targets.getRow(self.io, r, host_B[start .. start + cols], 32.0);
            try self.db.weights.getRow(self.io, r, host_W[start .. start + cols], 0.1);
        }

        // 2. Initialize GPU Context ONCE for the entire dataset
        var gpu_av = try kernel.checkaviable.load();
        defer gpu_av.deinit();
        var gpu_mem = try kernel.memorymenagement.load();
        defer gpu_mem.deinit();
        var gpu_proc = try kernel.processergpu.load();
        defer gpu_proc.deinit();

        checkCuda(gpu_av.cuInit(0));
        var device: kernel.CUdevice = 0;
        checkCuda(gpu_av.cuDeviceGet(&device, 0));
        var context: kernel.CUcontext = null;
        checkCuda(gpu_mem.cuCtxCreate(&context, 0, device));
        defer _ = gpu_mem.cuCtxDestroy(context);

        // 3. Allocate & Upload entire dataset to GPU VRAM
        var dev_A: kernel.CUdeviceptr = 0;
        var dev_B: kernel.CUdeviceptr = 0;
        var dev_W: kernel.CUdeviceptr = 0;
        var dev_out: kernel.CUdeviceptr = 0;

        checkCuda(gpu_mem.cuMemAlloc(&dev_A, byte_size));
        defer _ = gpu_mem.cuMemFree(dev_A);
        checkCuda(gpu_mem.cuMemAlloc(&dev_B, byte_size));
        defer _ = gpu_mem.cuMemFree(dev_B);
        checkCuda(gpu_mem.cuMemAlloc(&dev_W, byte_size));
        defer _ = gpu_mem.cuMemFree(dev_W);
        checkCuda(gpu_mem.cuMemAlloc(&dev_out, byte_size));
        defer _ = gpu_mem.cuMemFree(dev_out);

        checkCuda(gpu_mem.cuMemcpyHtoD(dev_A, @ptrCast(host_A.ptr), byte_size));
        checkCuda(gpu_mem.cuMemcpyHtoD(dev_B, @ptrCast(host_B.ptr), byte_size));
        checkCuda(gpu_mem.cuMemcpyHtoD(dev_W, @ptrCast(host_W.ptr), byte_size));

        // 4. Compile & Launch Gradient Descent Kernel across ALL rows at once
        const gen = op_mod.KernelGenerator.init(32);
        var ptx_buff: [4096]u8 = @splat(0);
        const built = try gen.emitTrainKernel(&ptx_buff, "batch_train", self.learning_rate);

        var module: kernel.CUmodule = null;
        checkCuda(gpu_proc.cuModuleLoadData(&module, built.ptx.ptr));
        defer _ = gpu_proc.cuModuleUnload(module);

        var kernel_fn: kernel.CUfunction = null;
        checkCuda(gpu_proc.cuModuleGetFunction(&kernel_fn, module, built.kername.ptr));

        var params = [_]?*anyopaque{
            @ptrCast(&dev_A),
            @ptrCast(&dev_B),
            @ptrCast(&dev_W),
            @ptrCast(&dev_out),
            @constCast(&total_elements),
        };

        const threads: c_uint = 256;
        const blocks: c_uint = (total_elements + threads - 1) / threads;

        for (0..epochs) |_| {
            checkCuda(gpu_proc.cuLaunchKernel(
                kernel_fn,
                blocks, 1, 1,
                threads, 1, 1,
                0, null,
                @ptrCast(&params[0]),
                null,
            ));
        }

        // 5. Copy all learned weights back to CPU and save to weights.mat
        checkCuda(gpu_mem.cuMemcpyDtoH(@ptrCast(host_W.ptr), dev_W, byte_size));

        for (0..@as(usize, @intCast(num_rows))) |r| {
            const start = r * cols;
            try self.db.weights.setRow(self.io, r, host_W[start .. start + cols]);
        }
    }

    /// Uses the saved weights from `weights.mat` on the GPU to generate an answer for `user_input`
    pub fn answer(self: *AnswerModel, relation_id: u64, user_input: []const u8, out_text: []u8) !void {
        const feature_len = user_input.len;
        const total_elements: u32 = @intCast(feature_len);
        const byte_size: usize = feature_len * @sizeOf(f32);

        const host_A = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(host_A);
        const host_W = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(host_W);
        const host_out = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(host_out);

        for (user_input, 0..) |ch, i| {
            host_A[i] = @floatFromInt(ch);
        }

        // Load the previously learned weights from the Matrix file
        try self.db.weights.getRow(self.io, relation_id, host_W, 0.1);

        // Run Inference on GPU: Answer = Input_A * Learned_Weight_W
        var gpu_av = try kernel.checkaviable.load();
        defer gpu_av.deinit();
        var gpu_mem = try kernel.memorymenagement.load();
        defer gpu_mem.deinit();
        var gpu_proc = try kernel.processergpu.load();
        defer gpu_proc.deinit();

        checkCuda(gpu_av.cuInit(0));
        var device: kernel.CUdevice = 0;
        checkCuda(gpu_av.cuDeviceGet(&device, 0));
        var context: kernel.CUcontext = null;
        checkCuda(gpu_mem.cuCtxCreate(&context, 0, device));
        defer _ = gpu_mem.cuCtxDestroy(context);

        var dev_A: kernel.CUdeviceptr = 0;
        var dev_W: kernel.CUdeviceptr = 0;
        var dev_out: kernel.CUdeviceptr = 0;

        checkCuda(gpu_mem.cuMemAlloc(&dev_A, byte_size));
        defer _ = gpu_mem.cuMemFree(dev_A);
        checkCuda(gpu_mem.cuMemAlloc(&dev_W, byte_size));
        defer _ = gpu_mem.cuMemFree(dev_W);
        checkCuda(gpu_mem.cuMemAlloc(&dev_out, byte_size));
        defer _ = gpu_mem.cuMemFree(dev_out);

        checkCuda(gpu_mem.cuMemcpyHtoD(dev_A, @ptrCast(host_A.ptr), byte_size));
        checkCuda(gpu_mem.cuMemcpyHtoD(dev_W, @ptrCast(host_W.ptr), byte_size));

        // Emit forward inference PTX kernel (.mul)
        const gen = op_mod.KernelGenerator.init(32);
        var ptx_buff: [4096]u8 = @splat(0);
        const built = try gen.emitArbitraryNDKernel(&ptx_buff, "infer_answer", .mul);

        var module: kernel.CUmodule = null;
        checkCuda(gpu_proc.cuModuleLoadData(&module, built.ptx.ptr));
        defer _ = gpu_proc.cuModuleUnload(module);

        var kernel_fn: kernel.CUfunction = null;
        checkCuda(gpu_proc.cuModuleGetFunction(&kernel_fn, module, built.kername.ptr));

        var params = [_]?*anyopaque{
            @ptrCast(&dev_A),
            @ptrCast(&dev_W),
            @ptrCast(&dev_out),
            @constCast(&total_elements),
        };

        const threads: c_uint = 256;
        const blocks: c_uint = (total_elements + threads - 1) / threads;

        checkCuda(gpu_proc.cuLaunchKernel(
            kernel_fn,
            blocks, 1, 1,
            threads, 1, 1,
            0, null,
            @ptrCast(&params[0]),
            null,
        ));

        checkCuda(gpu_mem.cuMemcpyDtoH(@ptrCast(host_out.ptr), dev_out, byte_size));

        // Convert predicted floats into readable ASCII text
        for (host_out, 0..) |val, i| {
            const clamped = std.math.clamp(@round(val), 32.0, 126.0);
            out_text[i] = @intFromFloat(clamped);
        }
    }
};

fn checkCuda(res: anytype) void {
    if (res != .CUDA_SUCCESS) {
        std.debug.print("CUDA Error: {any}\n", .{res});
    }
}