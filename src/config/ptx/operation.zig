const std = @import("std");
const RegisterAllocator = @import("ManRegister.zig").RegisterAllocator;

pub const OpKind = enum {
    add,
    sub,
    mul,
    div,

    pub fn ptxMnemonic(self: OpKind) []const u8 {
        return switch (self) {
            .add => "add.f32",
            .sub => "sub.f32",
            .mul => "mul.f32",
            .div => "div.approx.f32",
        };
    }

    pub fn asString(self: OpKind) []const u8 {
        return @tagName(self);
    }
};

pub const BuiltKernel = struct {
    kername: [:0]const u8,
    ptx: [:0]const u8,
};

pub const KernelGenerator = struct {
    allocator: RegisterAllocator,

    pub fn init(reg_count: u32) KernelGenerator {
        return .{ .allocator = RegisterAllocator.init(reg_count) };
    }

    /// Emits PTX for a single operation and returns both the generated kernel name and PTX
    pub fn emitArbitraryNDKernel(
        self: *const KernelGenerator,
        buf: []u8,
        base_name: []const u8,
        op: OpKind,
    ) !BuiltKernel {
        return self.emitFusedKernel(buf, base_name, &[_]OpKind{op});
    }

    /// Emits PTX for chained operations across N+1 distinct input tensors in registers:
    /// ops = &[_]OpKind{ .mul, .add } -> (param_in_0 * param_in_1) + param_in_2
    pub fn emitFusedKernel(
        self: *const KernelGenerator,
        buf: []u8,
        base_name: []const u8,
        ops: []const OpKind,
    ) !BuiltKernel {
        if (ops.len == 0) return error.EmptyOperationChain;
        if (buf.len <= 256) return error.BufferTooSmall;

        const name_space = buf[0..128];
        const ptx_space = buf[128..];

        // 1. Build dynamic kernel name: e.g., "ndmat_1d_mul_add"
        var name_writer: std.Io.Writer = .fixed(name_space[0 .. name_space.len - 1]);
        try name_writer.print("{s}", .{base_name});
        for (ops) |op| {
            try name_writer.print("_{s}", .{op.asString()});
        }
        const raw_name = name_writer.buffered();
        name_space[raw_name.len] = 0;

        var reg_buf: [256]u8 = undefined;
        const reg_directives = try self.allocator.injectRegDirectives(&reg_buf);

        var ptx_writer: std.Io.Writer = .fixed(ptx_space[0 .. ptx_space.len - 1]);
        const num_inputs = ops.len + 1; // 2 ops (.mul, .add) = 3 inputs (A, B, D)

        // 2. Emit Header with dynamic input parameters (param_in_0, param_in_1, ...)
        try ptx_writer.print(
            \\.version 7.0
            \\.target sm_75
            \\.address_size 64
            \\
            \\.visible .entry {s}(
            \\
        , .{raw_name});

        for (0..num_inputs) |i| {
            try ptx_writer.print("    .param .u64 param_in_{d},\n", .{i});
        }

        try ptx_writer.print(
            \\    .param .u64 param_out,
            \\    .param .u32 total_elements
            \\) {{
            \\{s}
            \\    ld.param.u32    %r0,  [total_elements];
            \\
            \\    mov.u32         %r1, %ctaid.x;
            \\    mov.u32         %r2, %ntid.x;
            \\    mov.u32         %r3, %tid.x;
            \\    mad.lo.s32      %r4, %r1, %r2, %r3;
            \\
            \\    setp.ge.s32     %p0, %r4, %r0;
            \\    @%p0 bra        L_EXIT;
            \\
            \\    // Byte offset = idx * 4 (kept in %rd1)
            \\    mul.wide.s32    %rd1, %r4, 4;
            \\
            \\    // Load first tensor (A / param_in_0) into running answer register %f1
            \\    ld.param.u64    %rd2, [param_in_0];
            \\    add.s64         %rd3, %rd2, %rd1;
            \\    ld.global.f32   %f1, [%rd3];
            \\
        , .{reg_directives});

        // 3. For each operation, load the next input tensor into %f2 and update %f1
        for (ops, 0..) |op, i| {
            try ptx_writer.print(
                \\    // Step {d}: Load param_in_{d} into %f2 and compute {s}
                \\    ld.param.u64    %rd2, [param_in_{d}];
                \\    add.s64         %rd3, %rd2, %rd1;
                \\    ld.global.f32   %f2, [%rd3];
                \\    {s}         %f1, %f1, %f2;
                \\
            , .{ i + 1, i + 1, op.asString(), i + 1, op.ptxMnemonic() });
        }

        // 4. Write final answer (%f1) to param_out only once
        try ptx_writer.print(
            \\
            \\    ld.param.u64    %rd2, [param_out];
            \\    add.s64         %rd3, %rd2, %rd1;
            \\    st.global.f32   [%rd3], %f1;
            \\
            \\L_EXIT:
            \\    ret;
            \\}}
        , .{});

        const raw_ptx = ptx_writer.buffered();
        ptx_space[raw_ptx.len] = 0;

        return .{
            .kername = name_space[0..raw_name.len :0],
            .ptx = ptx_space[0..raw_ptx.len :0],
        };
    }
    /// Emits a Fused Gradient Descent Training Kernel:
    ///   guess = input_A * weight
    ///   error = guess - target_B
    ///   weight_new = weight - (learning_rate * error * input_A)
    pub fn emitTrainKernel(
        self: *const KernelGenerator,
        buf: []u8,
        base_name: []const u8,
        learning_rate: f32,
    ) !BuiltKernel {
        if (buf.len <= 256) return error.BufferTooSmall;

        const name_space = buf[0..128];
        const ptx_space = buf[128..];

        var name_writer: std.Io.Writer = .fixed(name_space[0 .. name_space.len - 1]);
        try name_writer.print("{s}_gd_train", .{base_name});
        const raw_name = name_writer.buffered();
        name_space[raw_name.len] = 0;

        var reg_buf: [256]u8 = undefined;
        const reg_directives = try self.allocator.injectRegDirectives(&reg_buf);

        // Encode -learning_rate as an exact IEEE-754 hex float literal for PTX (0fXXXXXXXX)
        const neg_lr_bits: u32 = @bitCast(-learning_rate);

        var ptx_writer: std.Io.Writer = .fixed(ptx_space[0 .. ptx_space.len - 1]);
        try ptx_writer.print(
            \\.version 7.0
            \\.target sm_75
            \\.address_size 64
            \\
            \\.visible .entry {s}(
            \\    .param .u64 param_in_A,     // Input A (what you give, e.g. "hello")
            \\    .param .u64 param_in_B,     // Input B (target answer, e.g. "world")
            \\    .param .u64 param_weight,   // Learned Weights (updated in-place!)
            \\    .param .u64 param_out,      // Current Guess (input_A * weight_new)
            \\    .param .u32 total_elements
            \\) {{
            \\{s}
            \\    ld.param.u64    %rd1, [param_in_A];
            \\    ld.param.u64    %rd2, [param_in_B];
            \\    ld.param.u64    %rd3, [param_weight];
            \\    ld.param.u64    %rd4, [param_out];
            \\    ld.param.u32    %r0,  [total_elements];
            \\
            \\    mov.u32         %r1, %ctaid.x;
            \\    mov.u32         %r2, %ntid.x;
            \\    mov.u32         %r3, %tid.x;
            \\    mad.lo.s32      %r4, %r1, %r2, %r3;
            \\
            \\    setp.ge.s32     %p0, %r4, %r0;
            \\    @%p0 bra        L_EXIT;
            \\
            \\    // Byte offset = idx * 4
            \\    mul.wide.s32    %rd5, %r4, 4;
            \\    add.s64         %rd6, %rd1, %rd5; // &input_A[idx]
            \\    add.s64         %rd7, %rd2, %rd5; // &target_B[idx]
            \\    add.s64         %rd8, %rd3, %rd5; // &weight[idx]
            \\    add.s64         %rd9, %rd4, %rd5; // &out[idx]
            \\
            \\    // 1. Load Input A, Target B, and current Weight into registers
            \\    ld.global.f32   %f1, [%rd6];      // %f1 = input_A
            \\    ld.global.f32   %f2, [%rd7];      // %f2 = target_B
            \\    ld.global.f32   %f3, [%rd8];      // %f3 = weight
            \\
            \\    // 2. Forward Pass in Register: guess = input_A * weight
            \\    mul.f32         %f4, %f1, %f3;    // %f4 = guess
            \\
            \\    // 3. Error in Register: (guess - target_B)
            \\    sub.f32         %f5, %f4, %f2;    // %f5 = guess - target_B
            \\
            \\    // 4. Gradient in Register: (guess - target_B) * input_A
            \\    mul.f32         %f5, %f5, %f1;    // %f5 = error * input_A
            \\
            \\    // 5. Weight Update: weight_new = (-learning_rate * grad) + weight
            \\    mov.f32         %f6, 0f{X:0>8};   // %f6 = -learning_rate
            \\    fma.rn.f32      %f3, %f6, %f5, %f3;
            \\
            \\    // 6. Save learned weight & updated prediction to VRAM
            \\    mul.f32         %f4, %f1, %f3;    // updated guess = input_A * weight_new
            \\    st.global.f32   [%rd8], %f3;      // overwrite weight in VRAM
            \\    st.global.f32   [%rd9], %f4;      // store learned answer in VRAM
            \\
            \\L_EXIT:
            \\    ret;
            \\}}
        , .{ raw_name, reg_directives, neg_lr_bits });

        const raw_ptx = ptx_writer.buffered();
        ptx_space[raw_ptx.len] = 0;

        return .{
            .kername = name_space[0..raw_name.len :0],
            .ptx = ptx_space[0..raw_ptx.len :0],
        };
    }
};

test "operator generation with kernel name" {
    const gen = KernelGenerator.init(32);
    var buf: [2048]u8 = undefined;

    // 1. Test Single Operation
    const single = try gen.emitArbitraryNDKernel(&buf, "test_tensor", .mul);
    try std.testing.expectEqualStrings("test_tensor_mul", single.kername);
    try std.testing.expect(std.mem.indexOf(u8, single.ptx, "mul.f32") != null);
    std.debug.print("Kernel Name: {s}\n{s}\n\n", .{ single.kername, single.ptx });

    // 2. Test Fused Operations: (A * B) + B
    const ops = [_]OpKind{ .mul, .add };
    const fused = try gen.emitFusedKernel(&buf, "test_tensor", &ops);
    try std.testing.expectEqualStrings("test_tensor_mul_add", fused.kername);
    try std.testing.expect(std.mem.indexOf(u8, fused.ptx, "mul.f32         %f3, %f1, %f2;") != null);
    try std.testing.expect(std.mem.indexOf(u8, fused.ptx, "add.f32         %f3, %f3, %f2;") != null);
    std.debug.print("Fused Kernel Name: {s}\n{s}\n", .{ fused.kername, fused.ptx });
}