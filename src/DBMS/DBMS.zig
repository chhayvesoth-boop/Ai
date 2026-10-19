const std = @import("std");

pub const Matrix = struct {
    file: std.Io.File,
    cols: u64, // number of columns (features per row)

    pub const Cell = f32;
    const empty: Cell = std.math.nan(Cell);

    pub fn open(io: std.Io, path: []const u8, cols: u64) !Matrix {
        if (std.fs.path.dirname(path)) |dir| {
            try std.Io.Dir.cwd().createDirPath(io, dir);
        }

        const file = try std.Io.Dir.cwd().createFile(io, path, .{
            .read = true,
            .truncate = false,
        });
        return .{ .file = file, .cols = cols };
    }

    pub fn close(self: Matrix, io: std.Io) void {
        self.file.close(io);
    }

    // row-major, 0-based: cell number in the flat array
    fn index(self: Matrix, i: u64, j: u64) u64 {
        return i * self.cols + j;
    }

    // byte position in the file
    fn offset(self: Matrix, i: u64, j: u64) u64 {
        return self.index(i, j) * @sizeOf(Cell);
    }

    pub fn set(self: Matrix, io: std.Io, i: u64, j: u64, value: Cell) !void {
        std.debug.assert(j < self.cols);
        try self.file.writePositionalAll(io, std.mem.asBytes(&value), self.offset(i, j));
    }

    pub fn get(self: Matrix, io: std.Io, i: u64, j: u64) !Cell {
        std.debug.assert(j < self.cols);
        var value: Cell = 0; // stays 0 if the cell was never written   
        const got = try self.file.readPositionalAll(io, std.mem.asBytes(&value), self.offset(i, j));
        if (got < @sizeOf(Cell)) return 0; // past the end of the file
        return value;
    }

    pub fn delete(self: Matrix, io: std.Io, i: u64, j: u64) !void {
        try self.set(io, i, j, empty);
    }

    // returns null if the cell was deleted
    pub fn getOpt(self: Matrix, io: std.Io, i: u64, j: u64) !?Cell {
        const v = try self.get(io, i, j);
        if (std.math.isNan(v)) return null;
        return v;
    }
};