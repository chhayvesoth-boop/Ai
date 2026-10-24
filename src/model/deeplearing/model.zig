const std = @import("std");
const DBMS = @import("../../DBMS/DBMS.zig");
const kernel = @import("../../config/config.zig").loadkernel;
const op_mod = @import("../../config/ptx/operation.zig");

pub const AnswerModel = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    db: *DBMS.RelationDB,
    // Scaled learning rate so raw GPU gradient descent converges without the B/A division cheat
    learning_rate: f32 = 0.0005,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, db: *DBMS.RelationDB) AnswerModel {
        return .{
            .allocator = allocator,
            .io = io,
            .db = db,
        };
    }

    /// Normalizes an ASCII character + rolling letter context into a stable range [1.0 .. 4.5]
    /// so GPU gradient descent (out = A * W) converges naturally without exploding gradients.
    inline fn encodeCharContext(curr_char: f32, prev_state: f32, pos: usize) f32 {
        const c = std.math.clamp(curr_char, 32.0, 126.0);
        const p = std.math.clamp(prev_state, 32.0, 126.0);
        const norm_c = c / 32.0; // [1.0 .. 3.9375]
        const norm_p = (p - 32.0) / 256.0; // [0.0 .. 0.367]
        const pos_mod = @as(f32, @floatFromInt((pos % 8) + 1)) * 0.02;
        return norm_c + norm_p + pos_mod;
    }

    /// Builds the causal (letter_t -> letter_{t+1}) chain for a row
    fn buildNextLetterChain(
        user_seed: []const f32,
        raw_target: []const f32,
        chain_A: []f32,
        chain_B: []f32,
    ) void {
        const len = chain_A.len;
        var prev_state: f32 = 32.0;
        for (user_seed) |ch| {
            if (ch > 32.0) {
                prev_state = prev_state * 0.7 + ch * 0.3;
            }
        }

        const first_char = if (user_seed.len > 0 and user_seed[0] >= 32.0) user_seed[0] else 32.0;

        for (0..len) |i| {
            const current_letter = if (i == 0)
                first_char
            else
                (if (raw_target[i - 1] >= 32.0) raw_target[i - 1] else 32.0);

            chain_A[i] = encodeCharContext(current_letter, prev_state, i);
            chain_B[i] = if (raw_target[i] >= 32.0) raw_target[i] else 32.0;
            prev_state = current_letter;
        }
    }

    /// CHARACTER-LEVEL FUZZY MATCHER:
    /// Finds the relation row whose input letters best match `user_query` (e.g. "hellow" -> matches "hello").
    /// Call this from your CLI when exact relation lookup fails!
    pub fn findClosestRelation(self: *AnswerModel, user_query: []const u8, total_rows: u64) !?u64 {
        if (total_rows == 0 or user_query.len == 0) return null;

        const cols: usize = @intCast(self.db.cols);
        const row_buf = try self.allocator.alloc(f32, cols);
        defer self.allocator.free(row_buf);

        var best_row: ?u64 = null;
        var best_score: f32 = 0.0;

        for (0..@as(usize, @intCast(total_rows))) |r| {
            try self.db.inputs.getRow(self.io, r, row_buf, 32.0);

            // Extract actual non-space length of stored input key
            var key_len: usize = 0;
            for (0..cols) |i| {
                const v = @round(row_buf[i]);
                if (v > 32.0 and v <= 126.0) {
                    key_len = i + 1;
                }
            }
            if (key_len == 0) continue;

            // Score character-by-character overlap + bigram similarity
            var matches: f32 = 0.0;
            const min_len = @min(user_query.len, key_len);
            const max_len = @max(user_query.len, key_len);

            for (0..min_len) |i| {
                const q_c = std.ascii.toLower(user_query[i]);
                const k_c = std.ascii.toLower(@as(u8, @intFromFloat(std.math.clamp(@round(row_buf[i]), 32.0, 126.0))));
                if (q_c == k_c) {
                    matches += 1.0;
                } else if (i + 1 < user_query.len and std.ascii.toLower(user_query[i + 1]) == k_c) {
                    matches += 0.6; // Tolerate 1-char insertion/deletion
                } else if (i + 1 < key_len) {
                    const next_k = std.ascii.toLower(@as(u8, @intFromFloat(std.math.clamp(@round(row_buf[i + 1]), 32.0, 126.0))));
                    if (q_c == next_k) matches += 0.6;
                }
            }

            const score = matches / @as(f32, @floatFromInt(max_len));
            if (score > best_score and score >= 0.45) {
                best_score = score;
                best_row = @intCast(r);
            }
        }

        return best_row;
    }

    /// Trains weights purely via GPU Gradient Descent (NO B/A division cheat!)
    pub fn trainRelation(self: *AnswerModel, relation_id: u64, feature_len: usize, epochs: usize) !void {
        const total_elements: u32 = @intCast(feature_len);
        const byte_size: usize = feature_len * @sizeOf(f32);

        const raw_A = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(raw_A);
        const raw_B = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(raw_B);

        const host_A = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(host_A);
        const host_B = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(host_B);
        const host_W = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(host_W);

        try self.db.inputs.getRow(self.io, relation_id, raw_A, 32.0);
        try self.db.targets.getRow(self.io, relation_id, raw_B, 32.0);
        try self.db.weights.getRow(self.io, relation_id, host_W, 1.0);

        // Build normalized next-letter transition chain
        buildNextLetterChain(raw_A, raw_B, host_A, host_B);

        // Initialize unlearned weights to neutral baseline (NO B/A division!)
        for (0..feature_len) |i| {
            if (@abs(host_W[i]) < 0.001 or !std.math.isFinite(host_W[i])) {
                host_W[i] = 20.0; // Neutral starting guess (~space/ASCII baseline)
            }
        }

        // Setup GPU Context
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

        // Ensure enough gradient descent steps for true convergence without B/A pre-division
        const actual_epochs = @max(epochs, 15000);
        for (0..actual_epochs) |_| {
            checkCuda(gpu_proc.cuLaunchKernel(
                kernel_fn,
                blocks, 1, 1,
                threads, 1, 1,
                0, null,
                @ptrCast(&params[0]),
                null,
            ));
        }

        checkCuda(gpu_mem.cuMemcpyDtoH(@ptrCast(host_W.ptr), dev_W, byte_size));
        try self.db.weights.setRow(self.io, relation_id, host_W);
    }

    /// Trains `num_rows` simultaneously on the GPU using pure gradient descent
    pub fn trainAllRowsBatch(self: *AnswerModel, num_rows: u64, epochs: usize) !void {
        if (num_rows == 0) return;

        const cols: usize = @intCast(self.db.cols);
        const total_floats: usize = @as(usize, @intCast(num_rows)) * cols;
        const total_elements: u32 = @intCast(total_floats);
        const byte_size: usize = total_floats * @sizeOf(f32);

        const raw_row_A = try self.allocator.alloc(f32, cols);
        defer self.allocator.free(raw_row_A);
        const raw_row_B = try self.allocator.alloc(f32, cols);
        defer self.allocator.free(raw_row_B);

        const host_A = try self.allocator.alloc(f32, total_floats);
        defer self.allocator.free(host_A);
        const host_B = try self.allocator.alloc(f32, total_floats);
        defer self.allocator.free(host_B);
        const host_W = try self.allocator.alloc(f32, total_floats);
        defer self.allocator.free(host_W);

        for (0..@as(usize, @intCast(num_rows))) |r| {
            const start = r * cols;
            try self.db.inputs.getRow(self.io, r, raw_row_A, 32.0);
            try self.db.targets.getRow(self.io, r, raw_row_B, 32.0);
            try self.db.weights.getRow(self.io, r, host_W[start .. start + cols], 20.0);

            buildNextLetterChain(
                raw_row_A,
                raw_row_B,
                host_A[start .. start + cols],
                host_B[start .. start + cols],
            );

            for (0..cols) |i| {
                const w = host_W[start + i];
                if (@abs(w) < 0.001 or !std.math.isFinite(w) or w < 5.0) {
                    host_W[start + i] = 20.0;
                }
            }
        }

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

        const actual_epochs = @max(epochs, 15000);
        for (0..actual_epochs) |_| {
            checkCuda(gpu_proc.cuLaunchKernel(
                kernel_fn,
                blocks, 1, 1,
                threads, 1, 1,
                0, null,
                @ptrCast(&params[0]),
                null,
            ));
        }

        checkCuda(gpu_mem.cuMemcpyDtoH(@ptrCast(host_W.ptr), dev_W, byte_size));

        for (0..@as(usize, @intCast(num_rows))) |r| {
            const start = r * cols;
            try self.db.weights.setRow(self.io, r, host_W[start .. start + cols]);
        }
    }

    /// Generates the answer ONE LETTER AT A TIME directly from `user_input` (never touches inputs.mat!)
    pub fn answer(self: *AnswerModel, relation_id: u64, user_input: []const u8, out_text: []u8) !void {
        const feature_len: usize = @intCast(self.db.cols);
        const byte_size: usize = feature_len * @sizeOf(f32);

        const host_A = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(host_A);
        const host_W = try self.allocator.alloc(f32, feature_len);
        defer self.allocator.free(host_W);

        @memset(host_A, 1.0);

        // Build rolling character state directly from the user's live prompt (`user_input`)
        var prev_state: f32 = 32.0;
        for (user_input) |ch| {
            const cf: f32 = @floatFromInt(ch);
            if (cf > 32.0) {
                prev_state = prev_state * 0.7 + cf * 0.3;
            }
        }
        var current_letter: f32 = if (user_input.len > 0) @floatFromInt(user_input[0]) else 32.0;

        // Load only the learned weights from `weights.mat`
        try self.db.weights.getRow(self.io, relation_id, host_W, 20.0);

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

        checkCuda(gpu_mem.cuMemcpyHtoD(dev_W, @ptrCast(host_W.ptr), byte_size));

        const gen = op_mod.KernelGenerator.init(32);
        var ptx_buff: [4096]u8 = @splat(0);
        const built = try gen.emitArbitraryNDKernel(&ptx_buff, "infer_next_char", .mul);

        var module: kernel.CUmodule = null;
        checkCuda(gpu_proc.cuModuleLoadData(&module, built.ptx.ptr));
        defer _ = gpu_proc.cuModuleUnload(module);

        var kernel_fn: kernel.CUfunction = null;
        checkCuda(gpu_proc.cuModuleGetFunction(&kernel_fn, module, built.kername.ptr));

        const out_len = @min(out_text.len, feature_len);
        const single_elem: u32 = 1;

        for (0..out_len) |i| {
            host_A[i] = encodeCharContext(current_letter, prev_state, i);

            const offset_bytes = i * @sizeOf(f32);
            var step_dev_A: kernel.CUdeviceptr = dev_A + offset_bytes;
            var step_dev_W: kernel.CUdeviceptr = dev_W + offset_bytes;
            var step_dev_out: kernel.CUdeviceptr = dev_out + offset_bytes;

            checkCuda(gpu_mem.cuMemcpyHtoD(step_dev_A, @ptrCast(&host_A[i]), @sizeOf(f32)));

            var params = [_]?*anyopaque{
                @ptrCast(&step_dev_A),
                @ptrCast(&step_dev_W),
                @ptrCast(&step_dev_out),
                @constCast(&single_elem),
            };

            checkCuda(gpu_proc.cuLaunchKernel(
                kernel_fn,
                1, 1, 1,
                1, 1, 1,
                0, null,
                @ptrCast(&params[0]),
                null,
            ));

            var predicted_val: f32 = 32.0;
            checkCuda(gpu_mem.cuMemcpyDtoH(@ptrCast(&predicted_val), step_dev_out, @sizeOf(f32)));

            const clamped = std.math.clamp(@round(predicted_val), 32.0, 126.0);
            out_text[i] = @intFromFloat(clamped);

            prev_state = current_letter;
            current_letter = clamped;
        }
    }
    /// Scans `inputs.mat` on disk at startup to see how many rows are already saved
    /// so the .exe immediately remembers everything from previous runs without pressing [1] or [3]!
    pub fn countStoredRows(self: *AnswerModel, max_scan_rows: u64) !u64 {
        const cols: usize = @intCast(self.db.cols);
        const row_buf = try self.allocator.alloc(f32, cols);
        defer self.allocator.free(row_buf);

        var count: u64 = 0;
        for (0..@as(usize, @intCast(max_scan_rows))) |r| {
            self.db.inputs.getRow(self.io, r, row_buf, 0.0) catch break;

            // Check if this row on disk has actual saved characters (not empty/zero)
            var has_data = false;
            for (row_buf) |v| {
                if (v > 32.0 and v <= 126.0) {
                    has_data = true;
                    break;
                }
            }
            if (!has_data) break;
            count = @intCast(r + 1);
        }
        return count;
    }

    /// Saves a single new (input_str -> target_str) pair directly to disk matrices AND trains GPU weights immediately
    pub fn learnAndPersist(
        self: *AnswerModel,
        relation_id: u64,
        input_str: []const u8,
        target_str: []const u8,
        epochs: usize,
    ) !void {
        const cols: usize = @intCast(self.db.cols);
        const in_row = try self.allocator.alloc(f32, cols);
        defer self.allocator.free(in_row);
        const tgt_row = try self.allocator.alloc(f32, cols);
        defer self.allocator.free(tgt_row);

        @memset(in_row, 32.0);
        @memset(tgt_row, 32.0);

        for (0..@min(input_str.len, cols)) |i| {
            in_row[i] = @floatFromInt(input_str[i]);
        }
        for (0..@min(target_str.len, cols)) |i| {
            tgt_row[i] = @floatFromInt(target_str[i]);
        }

        // 1. Immediately write Input & Target to disk (.mat files) so it is NEVER lost
        try self.db.inputs.setRow(self.io, relation_id, in_row);
        try self.db.targets.setRow(self.io, relation_id, tgt_row);

        // 2. Train on GPU and immediately write learned Weights to `weights.mat`
        try self.trainRelation(relation_id, cols, epochs);
    }
};

fn checkCuda(res: anytype) void {
    if (res != .CUDA_SUCCESS) {
        std.debug.print("CUDA Error: {any}\n", .{res});
    }
}