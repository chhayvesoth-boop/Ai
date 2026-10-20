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

pub const KernelGenerator = struct {
    allocator: RegisterAllocator,

    pub fn init(reg_count: u32) KernelGenerator {
        return .{ .allocator = RegisterAllocator.init(reg_count) };
    }

    /// Emits PTX for N-dimensional tensor operations mapped over total elements
    pub fn emitArbitraryNDKernel(
        self: *const KernelGenerator,
        buf: []u8,
        kernel_name: []const u8,
        op: OpKind,
    ) ![]const u8 {
        var reg_buf: [256]u8 = undefined;
        const reg_directives = try self.allocator.injectRegDirectives(&reg_buf);

        return try std.fmt.bufPrint(buf,
            \\.version 8.0
            \\.target sm_80
            \\.address_size 64
            \\
            \\.visible .entry {s}(
            \\    .param .u64 param_in_A,
            \\    .param .u64 param_in_B,
            \\    .param .u64 param_out,
            \\    .param .u32 total_elements
            \\) {{
            \\{s}
            \\    // Load pointers & total flattened count
            \\    ld.param.u64    %rd1, [param_in_A];
            \\    ld.param.u64    %rd2, [param_in_B];
            \\    ld.param.u64    %rd3, [param_out];
            \\    ld.param.u32    %r0,  [total_elements];
            \\
            \\    // Calculate global linear thread index
            \\    mov.u32         %r1, %ctaid.x;
            \\    mov.u32         %r2, %ntid.x;
            \\    mov.u32         %r3, %tid.x;
            \\    mad.lo.s32      %r4, %r1, %r2, %r3;
            \\
            \\    // Bounds guard: if (idx >= total_elements) return;
            \\    setp.ge.s32     %p0, %r4, %r0;
            \\    @%p0 bra        L_EXIT;
            \\
            \\    // Byte offset = idx * 4 (sizeof(f32))
            \\    mul.wide.s32    %rd4, %r4, 4;
            \\    add.s64         %rd5, %rd1, %rd4;
            \\    add.s64         %rd6, %rd2, %rd4;
            \\    add.s64         %rd7, %rd3, %rd4;
            \\
            \\    // Load elements from input tensors
            \\    ld.global.f32   %f1, [%rd5];
            \\    ld.global.f32   %f2, [%rd6];
            \\
            \\    // Dynamic operation: {s}
            \\    {s}         %f3, %f1, %f2;
            \\
            \\    // Store result into designated output
            \\    st.global.f32   [%rd7], %f3;
            \\
            \\L_EXIT:
            \\    ret;
            \\}}
        , .{ kernel_name, reg_directives, op.asString(), op.ptxMnemonic() });
    }
};

test "operator generation" {
    const gen = KernelGenerator.init(32);
    var buf: [2048]u8 = undefined;
    const ptx = try gen.emitArbitraryNDKernel(&buf, "test_tensor", .mul);
    try std.testing.expect(std.mem.indexOf(u8, ptx, "mul.f32") != null);
}