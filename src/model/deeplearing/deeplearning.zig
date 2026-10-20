const std=@import("std");
const DBMS=@import("../../DBMS/DBMS.zig");
const kernel =@import("../../config/config.zig").loadkernel;

pub fn run() !void {

    var buff_in:[1024]u8=undefined;
    var buff_out:[1024]u8=undefined;
    const io = std.Io.Threaded.global_single_threaded.io();


    var stdin_reader=std.Io.File.stdin().readerStreaming(io,&buff_in);// reader data in bus
    const stdin: *std.Io.Reader = &stdin_reader.interface;//take reader fn
    _=stdin;

    var stdout_writer=std.Io.File.stdout().writerStreaming(io,&buff_out);// write data to bus
    const stdout: *std.Io.Writer = &stdout_writer.interface;//take writer fn

    var loadgpu = try kernel.checkaviable.load();
    defer loadgpu.deinit();
    std.debug.print("{}\n",.{loadgpu.cuInit(0)});

    try stdout.writeAll("hello");
    try stdout.flush();


    

}