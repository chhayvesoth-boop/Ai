const std = @import("std");
const op_mod = @import("operation.zig");
const KernelGenerator = op_mod.KernelGenerator;
pub const OpKind = op_mod.OpKind;

pub const NDMatrix = struct {
    shape: []const usize,

    pub fn init(shape: []const usize) NDMatrix {
        return .{ .shape = shape };
    }

    /// Computes N = d0 * d1 * ... * d(n-1)
    pub fn totalElements(self: NDMatrix) usize {
        if (self.shape.len == 0) return 0;
        var total: usize = 1;
        for (self.shape) |dim| {
            total *= dim;
        }
        return total;
    }
    pub const BuiltKernel = struct {
        kername: [:0]const u8,
        ptx: [:0]const u8,
    };

    /// Dynamically injects shape metadata into kernel name and emits PTX
    pub fn buildDynamicKernel(self: NDMatrix, ptx_buffer: []u8, op: OpKind) !BuiltKernel {
        const name_space = ptx_buffer[0..128];
        const ptx_space = ptx_buffer[128..];

        var writer: std.Io.Writer = .fixed(name_space);

        try writer.print("ndmat_{s}_{d}d", .{ op.asString(), self.shape.len });
        for (self.shape) |dim| {
            try writer.print("x{d}", .{dim});
        }
        const raw_name = writer.buffered();
        name_space[raw_name.len] = 0; // Null-terminate kernel name

        const total = self.totalElements();
        const reg_count: u32 = if (total > 2048) 64 else 32;

        const generator = KernelGenerator.init(reg_count);
        const raw_ptx = try generator.emitArbitraryNDKernel(ptx_space, raw_name, op);
        ptx_space[raw_ptx.len] = 0; // Null-terminate PTX string

        return .{
            .kername = name_space[0..raw_name.len :0],
            .ptx = ptx_space[0..raw_ptx.len :0],
        };
    }
};

test "arbitrary n-dimensional matrix kernel" {
    // Arbitrary 5D matrix: 2 x 3 x 4 x 5 x 6 = 720 elements
    const shape_5d = [_]usize{ 2, 3, 4, 5, 6 };
    const mat5d = NDMatrix.init(&shape_5d);

    try std.testing.expectEqual(@as(usize, 720), mat5d.totalElements());

    var buffer: [4096]u8 = undefined;
    const ptx = try mat5d.buildDynamicKernel(&buffer, .mul);

    // Verify name generation and single result parameter
    try std.testing.expect(std.mem.indexOf(u8, ptx.ptx, "ndmat_mul_5dx2x3x4x5x6") != null);
    try std.testing.expect(std.mem.indexOf(u8, ptx.ptx, ".param .u64 param_out") != null);
    try std.testing.expect(std.mem.indexOf(u8, ptx.ptx, "mul.f32") != null);

    std.debug.print("\n--- Generated N-D PTX Kernel ---\n{s}\n", .{ptx.ptx});
}