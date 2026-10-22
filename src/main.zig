const std = @import("std");
const Io = std.Io;
const model = @import("model/model.zig");
const Ai = @import("Ai");

pub fn main(init: std.process.Init) !void {

    // This is appropriate for anything that lives as long as the process.
    const arena: std.mem.Allocator = init.arena.allocator();


    // Accessing command line arguments:
    const args = try init.minimal.args.toSlice(arena);
    for (args) |arg| {
        std.log.info("arg: {s}", .{arg});
    }

        try model.trash.run();

}

