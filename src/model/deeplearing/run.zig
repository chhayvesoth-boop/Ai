const std = @import("std");
const DBMS = @import("../../DBMS/DBMS.zig");
const ModelMod = @import("model.zig");

const learn_files = [_][]const u8{
    "data/learn data/all_ai_models.csv",
    "data/learn data/frontier_ai_models.csv",
    "data/learn data/large_scale_ai_models.csv",
    "data/learn data/notable_ai_models.csv",
    "data/learn data/README.md",
};

fn promptLine(stdout: *std.Io.Writer, stdin: *std.Io.Reader, prompt_msg: []const u8) ![]const u8 {
    try stdout.print("{s}", .{prompt_msg});
    try stdout.flush();
    const raw = try stdin.takeDelimiterInclusive('\n');
    return std.mem.trim(u8, raw, "\r\n\x00 ");
}

/// Splits a CSV or Markdown line into (Input_A, Target_B) automatically
fn splitLineIntoRelation(line: []const u8) ?struct { input: []const u8, target: []const u8 } {
    // 1. If CSV line (has comma), split at first comma: ModelName -> Details
    if (std.mem.indexOfScalar(u8, line, ',')) |comma_idx| {
        const key = std.mem.trim(u8, line[0..comma_idx], " \"'\t");
        const val = std.mem.trim(u8, line[comma_idx + 1 ..], " \"'\t\r\n");
        if (key.len > 0 and val.len > 0) {
            return .{ .input = key, .target = val };
        }
    }
    // 2. If Markdown/text with ':', split at ':'
    if (std.mem.indexOfScalar(u8, line, ':')) |colon_idx| {
        const key = std.mem.trim(u8, line[0..colon_idx], "#*- \"\t");
        const val = std.mem.trim(u8, line[colon_idx + 1 ..], " \"\t\r\n");
        if (key.len > 0 and val.len > 0) {
            return .{ .input = key, .target = val };
        }
    }
    // 3. Fallback: split at first space
    if (std.mem.indexOfScalar(u8, line, ' ')) |space_idx| {
        const key = std.mem.trim(u8, line[0..space_idx], "#*- \t");
        const val = std.mem.trim(u8, line[space_idx + 1 ..], " \t\r\n");
        if (key.len > 0 and val.len > 0) {
            return .{ .input = key, .target = val };
        }
    }
    return null;
}

/// Reads all files in `data/learn data/`, stores relations in Matrix DB, and returns total rows loaded
fn ingestAllLearnData(
    allocator: std.mem.Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
    db: *DBMS.RelationDB,
    max_rows: u64,
) !u64 {
    var current_row: u64 = 0;
    const cols: usize = @intCast(db.cols);

    const padded_in = try allocator.alloc(u8, cols);
    defer allocator.free(padded_in);
    const padded_tgt = try allocator.alloc(u8, cols);
    defer allocator.free(padded_tgt);

    for (learn_files) |file_path| {
        const file = std.Io.Dir.cwd().openFile(io, file_path, .{}) catch |err| {
            try stdout.print("Skipping {s} ({any})\n", .{ file_path, err });
            try stdout.flush();
            continue;
        };
        defer file.close(io);

        // Read file content into memory (up to 16 MB per file)
        const stat = try file.stat(io);
        const file_size: usize = @intCast(@min(stat.size, 16 * 1024 * 1024));
        if (file_size == 0) continue;

        const content = try allocator.alloc(u8, file_size);
        defer allocator.free(content);
        _ = try file.readPositionalAll(io, content, 0);

        var line_iter = std.mem.splitScalar(u8, content, '\n');
        var line_num: usize = 0;
        var file_rows: usize = 0;

        while (line_iter.next()) |raw_line| : (line_num += 1) {
            if (current_row >= max_rows) break;
            const line = std.mem.trim(u8, raw_line, "\r\n\t ");
            if (line.len < 3) continue;

            // Skip CSV header row (line 0 in .csv files)
            if (line_num == 0 and std.mem.endsWith(u8, file_path, ".csv")) continue;

            if (splitLineIntoRelation(line)) |rel| {
                @memset(padded_in, ' ');
                @memset(padded_tgt, ' ');

                const in_len = @min(rel.input.len, cols);
                const tgt_len = @min(rel.target.len, cols);
                @memcpy(padded_in[0..in_len], rel.input[0..in_len]);
                @memcpy(padded_tgt[0..tgt_len], rel.target[0..tgt_len]);

                try db.storeTextRelation(io, current_row, padded_in, padded_tgt);
                current_row += 1;
                file_rows += 1;
            }
        }

        try stdout.print("Loaded {d} rows from \"{s}\"\n", .{ file_rows, file_path });
        try stdout.flush();
    }

    return current_row;
}

pub fn run() !void {
    var buff_in: [1024]u8 = undefined;
    var buff_out: [1024]u8 = undefined;
    const io = std.Io.Threaded.global_single_threaded.io();

    var stdin_reader = std.Io.File.stdin().readerStreaming(io, &buff_in);
    const stdin: *std.Io.Reader = &stdin_reader.interface;

    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &buff_out);
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    const allocator = std.heap.page_allocator;

    // 128 columns per row (fits full CSV descriptions), up to 10,000 rows
    const max_cols: u64 = 128;
    const max_rows: u64 = 10000;
    var db = try DBMS.RelationDB.open(io, "data/relations", max_cols);
    defer db.close(io);

    var model = ModelMod.AnswerModel.init(allocator, io, &db);

    while (true) {
        try stdout.print("\n============================================================\n", .{});
        try stdout.print("[1] Learn Text | [2] Ask Answer | [3] Learn ALL 'data/learn data' | [q] Quit\n", .{});
        const mode = try promptLine(stdout, stdin, "> ");

        if (std.mem.eql(u8, mode, "q") or std.mem.eql(u8, mode, "quit")) break;

        if (std.mem.eql(u8, mode, "3")) {
            try stdout.print("\nReading all files from 'data/learn data/'...\n", .{});
            try stdout.flush();

            const total_rows = try ingestAllLearnData(allocator, io, stdout, &db, max_rows);
            if (total_rows == 0) {
                try stdout.print("No rows found to train!\n", .{});
                try stdout.flush();
                continue;
            }

            try stdout.print("Training GPU on ALL {d} rows in parallel (5,000 epochs)...\n", .{total_rows});
            try stdout.flush();

            try model.trainAllRowsBatch(total_rows, 15000);

            try stdout.print("Done! Learned {d} rows and saved weights to 'data/relations/weights.mat'!\n", .{total_rows});
            try stdout.flush();
        } else if (std.mem.eql(u8, mode, "1")) {
            const raw_line = try promptLine(stdout, stdin, "Feed text to learn (e.g. 'GPT-4 OpenAI,2023'): ");
            if (splitLineIntoRelation(raw_line)) |rel| {
                const cols: usize = @intCast(max_cols);
                const padded_in = try allocator.alloc(u8, cols);
                defer allocator.free(padded_in);
                const padded_tgt = try allocator.alloc(u8, cols);
                defer allocator.free(padded_tgt);
                @memset(padded_in, ' ');
                @memset(padded_tgt, ' ');

                const in_len = @min(rel.input.len, cols);
                const tgt_len = @min(rel.target.len, cols);
                @memcpy(padded_in[0..in_len], rel.input[0..in_len]);
                @memcpy(padded_tgt[0..tgt_len], rel.target[0..tgt_len]);

                const row_id = try db.findOrNextRow(io, rel.input, max_rows);
                try db.storeTextRelation(io, row_id, padded_in, padded_tgt);
                try model.trainRelation(row_id, cols, 5000);

                try stdout.print("Learned Row #{d}: \"{s}\" -> \"{s}\"\n", .{ row_id, rel.input, rel.target });
                try stdout.flush();
            }
        } else if (std.mem.eql(u8, mode, "2")) {
            const user_q = try promptLine(stdout, stdin, "Ask Model (e.g. model name from CSV): ");
            if (user_q.len == 0) continue;

            const cols: usize = @intCast(max_cols);
            const padded_q = try allocator.alloc(u8, cols);
            defer allocator.free(padded_q);
            @memset(padded_q, ' ');
            const q_len = @min(user_q.len, cols);
            @memcpy(padded_q[0..q_len], user_q[0..q_len]);

            const row_id = (try db.findRelation(io, user_q, max_rows)) orelse 0;

            const reply_buf = try allocator.alloc(u8, cols);
            defer allocator.free(reply_buf);

            try model.answer(row_id, padded_q, reply_buf);
            const trimmed_reply = std.mem.trimEnd(u8, reply_buf, " ");

            try stdout.print("Model Answer (Row #{d}): {s}\n", .{ row_id, trimmed_reply });
            try stdout.flush();
        }
    }
}