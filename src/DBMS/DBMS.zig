const std = @import("std");

pub const Matrix = struct {
    file: std.Io.File,
    cols: u64, // number of columns (max characters/features per row)

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

    fn index(self: Matrix, i: u64, j: u64) u64 {
        return i * self.cols + j;
    }

    fn offset(self: Matrix, i: u64, j: u64) u64 {
        return self.index(i, j) * @sizeOf(Cell);
    }

    pub fn set(self: Matrix, io: std.Io, i: u64, j: u64, value: Cell) !void {
        std.debug.assert(j < self.cols);
        try self.file.writePositionalAll(io, std.mem.asBytes(&value), self.offset(i, j));
    }

    pub fn get(self: Matrix, io: std.Io, i: u64, j: u64) !Cell {
        std.debug.assert(j < self.cols);
        var value: Cell = 0;
        const got = try self.file.readPositionalAll(io, std.mem.asBytes(&value), self.offset(i, j));
        if (got < @sizeOf(Cell)) return 0;
        return value;
    }

    pub fn delete(self: Matrix, io: std.Io, i: u64, j: u64) !void {
        try self.set(io, i, j, empty);
    }

    pub fn getOpt(self: Matrix, io: std.Io, i: u64, j: u64) !?Cell {
        const v = try self.get(io, i, j);
        if (std.math.isNan(v)) return null;
        return v;
    }

    // --- Fast Row Operations for GPU Slice Transfers ---

    /// Writes an entire f32 slice into row `row_id`
    pub fn setRow(self: Matrix, io: std.Io, row_id: u64, values: []const Cell) !void {
        std.debug.assert(values.len <= self.cols);
        const bytes = std.mem.sliceAsBytes(values);
        try self.file.writePositionalAll(io, bytes, self.offset(row_id, 0));
    }

    /// Reads row `row_id` directly into `out_buf` (fills unwritten cells with `default_val`)
    pub fn getRow(self: Matrix, io: std.Io, row_id: u64, out_buf: []Cell, default_val: Cell) !void {
        std.debug.assert(out_buf.len <= self.cols);
        @memset(out_buf, default_val);
        const bytes = std.mem.sliceAsBytes(out_buf);
        _ = try self.file.readPositionalAll(io, bytes, self.offset(row_id, 0));
    }
};

/// Relational Database linking Input (A) -> Target (B) -> Learned Model Weights (W) by row_id
pub const RelationDB = struct {
    inputs: Matrix,  // Table 1: Stores input strings as f32 (e.g., "hello")
    targets: Matrix, // Table 2: Stores target answers as f32 (e.g., "world")
    weights: Matrix, // Table 3: Stores learned GPU weights (W) for each relation
    cols: u64,

    pub fn open(io: std.Io, base_dir: []const u8, cols: u64) !RelationDB {
        var path_buf: [256]u8 = undefined;

        const in_path = try std.fmt.bufPrint(&path_buf, "{s}/inputs.mat", .{base_dir});
        const inputs = try Matrix.open(io, in_path, cols);

        const tgt_path = try std.fmt.bufPrint(&path_buf, "{s}/targets.mat", .{base_dir});
        const targets = try Matrix.open(io, tgt_path, cols);

        const w_path = try std.fmt.bufPrint(&path_buf, "{s}/weights.mat", .{base_dir});
        const weights = try Matrix.open(io, w_path, cols);

        return .{
            .inputs = inputs,
            .targets = targets,
            .weights = weights,
            .cols = cols,
        };
    }

    pub fn close(self: RelationDB, io: std.Io) void {
        self.inputs.close(io);
        self.targets.close(io);
        self.weights.close(io);
    }

    /// Stores a text pair ("hello" -> "world") at `relation_id` and initializes default weights
    pub fn storeTextRelation(
        self: RelationDB,
        io: std.Io,
        relation_id: u64,
        input_text: []const u8,
        target_text: []const u8,
    ) !void {
        const len = @min(input_text.len, @as(usize, @intCast(self.cols)));
        for (0..len) |j| {
            const in_val: f32 = @floatFromInt(input_text[j]);
            const tgt_val: f32 = @floatFromInt(target_text[j % target_text.len]);

            try self.inputs.set(io, relation_id, j, in_val);
            try self.targets.set(io, relation_id, j, tgt_val);

            // Initialize weight to 0.1 only if it hasn't been learned yet (0.0)
            const existing_w = try self.weights.get(io, relation_id, j);
            if (existing_w == 0.0) {
                try self.weights.set(io, relation_id, j, 0.1);
            }
        }
    }
    /// Finds the row_id that matches `input_text`, or returns null if not learned yet
    pub fn findRelation(self: RelationDB, io: std.Io, input_text: []const u8, max_rows: u64) !?u64 {
        const len = @min(input_text.len, @as(usize, @intCast(self.cols)));
        for (0..max_rows) |row_id| {
            const first_cell = try self.inputs.get(io, row_id, 0);
            if (first_cell == 0.0) break; // Reached unwritten rows

            var matches = true;
            for (0..len) |j| {
                const stored_val = try self.inputs.get(io, row_id, j);
                const input_val: f32 = @floatFromInt(input_text[j]);
                if (@abs(stored_val - input_val) > 0.5) {
                    matches = false;
                    break;
                }
            }
            if (matches) return row_id;
        }
        return null;
    }

    /// Finds an existing row for `input_text`, or allocates the next empty row_id
    pub fn findOrNextRow(self: RelationDB, io: std.Io, input_text: []const u8, max_rows: u64) !u64 {
        if (try self.findRelation(io, input_text, max_rows)) |existing_row| {
            return existing_row;
        }
        for (0..max_rows) |row_id| {
            const first_cell = try self.inputs.get(io, row_id, 0);
            if (first_cell == 0.0) return row_id;
        }
        return 0; // Fallback overwrite row 0 if full
    }
};