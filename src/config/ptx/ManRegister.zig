const std = @import("std");

pub const RegisterAllocator = struct {
    num_regs: u32,

    pub fn init(num_regs: u32) RegisterAllocator {
        return .{ .num_regs = num_regs };
    }

    pub fn injectRegDirectives(self: *const RegisterAllocator, buf: []u8) ![]const u8 {
        var writer: std.Io.Writer = .fixed(buf);
        try writer.print(
            \\    .reg .pred %p<2>;
            \\    .reg .b32  %r<{d}>;
            \\    .reg .b64  %rd<16>;
            \\    .reg .f32  %f<16>;
        , .{self.num_regs});
        return writer.buffered();
    }
};

test "register allocation" {
    const reg_alloc = RegisterAllocator.init(32);
    var buf: [256]u8 = undefined;
    const res = try reg_alloc.injectRegDirectives(&buf);
    try std.testing.expect(std.mem.indexOf(u8, res, "%r<32>") != null);
}