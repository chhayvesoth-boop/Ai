const std = @import("std");
const DBMS = @import("../../DBMS/DBMS.zig");
const ModelMod = @import("model.zig");

const learn_files = [_][]const u8{
    "learn data/everyday_human.md",
};

const builtin_qa = [_]struct { input: []const u8, target: []const u8 }{
    .{ .input = "hello", .target = "Hello! Ask me about an AI model from my local data." },
    .{ .input = "hi", .target = "Hi! Ask me about an AI model from my local data." },
    .{ .input = "who are you", .target = "I am a local AI-model lookup assistant trained from the data files in learn data." },
    .{ .input = "help", .target = "Ask for an AI model by name, such as GPT-4, AlexNet, AlphaGo, Claude, or Gemini." },
};

/// Automatically finds the `data` directory whether Ai.exe is run from project root OR `zig-out/bin/`
fn resolveDataBasePrefix(io: std.Io) []const u8 {
    if (std.Io.Dir.cwd().openFile(io, "data/learn data/README.md", .{})) |f| {
        f.close(io);
        return "data";
    } else |_| {}

    if (std.Io.Dir.cwd().openFile(io, "../../data/learn data/README.md", .{})) |f| {
        f.close(io);
        return "../../data";
    } else |_| {}

    if (std.Io.Dir.cwd().openFile(io, "../data/learn data/README.md", .{})) |f| {
        f.close(io);
        return "../data";
    } else |_| {}

    return "data";
}

fn promptLine(stdout: *std.Io.Writer, stdin: *std.Io.Reader, prompt_msg: []const u8) ![]const u8 {
    try stdout.print("{s}", .{prompt_msg});
    try stdout.flush();
    const raw = try stdin.takeDelimiterInclusive('\n');
    return std.mem.trim(u8, raw, "\r\n\x00 ");
}

/// Splits a CSV or Markdown line into (Input_A, Target_B) using a caller-provided buffer
fn csvColumn(line: []const u8, wanted_column: usize) ?[]const u8 {
    var column: usize = 0;
    var start: usize = 0;
    var quoted = false;
    var i: usize = 0;
    while (i < line.len) : (i += 1) {
        if (line[i] == '"') {
            if (quoted and i + 1 < line.len and line[i + 1] == '"') {
                i += 1;
            } else {
                quoted = !quoted;
            }
        } else if (line[i] == ',' and !quoted) {
            if (column == wanted_column) return line[start..i];
            column += 1;
            start = i + 1;
        }
    }
    if (column == wanted_column) return line[start..];
    return null;
}

fn splitLineIntoRelation(line: []const u8, out_target_buf: []u8) ?struct { input: []const u8, target: []const u8 } {
    if (csvColumn(line, 1) == null) {
        if (std.mem.indexOfAny(u8, line, ":-=")) |sep_idx| {
            const k = std.mem.trim(u8, line[0..sep_idx], " #*\"'\t");
            const v = std.mem.trim(u8, line[sep_idx + 1 ..], " #*\"'\t");
            if (k.len == 0 or v.len == 0) return null;
            const target_str = std.fmt.bufPrint(out_target_buf, "{s}", .{v}) catch return null;
            return .{ .input = k, .target = target_str };
        }
        return null;
    }

    const model_name = csvColumn(line, 0) orelse return null;
    const task = csvColumn(line, 2) orelse return null;
    const org = csvColumn(line, 3) orelse "Unknown Org";
    const clean_name = std.mem.trim(u8, model_name, " \"'\t");
    if (clean_name.len == 0) return null;

    const clean_task = std.mem.trim(u8, task, " \"'\t");
    const clean_org = std.mem.trim(u8, org, " \"'\t");

    const target_str = std.fmt.bufPrint(
        out_target_buf,
        "Task: {s} | Org: {s}",
        .{
            if (clean_task.len > 0) clean_task else "General",
            if (clean_org.len > 0) clean_org else "Unknown",
        },
    ) catch return null;

    return .{ .input = clean_name, .target = target_str };
}

fn storePaddedPair(
    io: std.Io,
    db: *DBMS.RelationDB,
    row_id: u64,
    input: []const u8,
    target: []const u8,
    padded_in: []f32,
    padded_tgt: []f32,
) !void {
    const cols = padded_in.len;
    @memset(padded_in, 32.0);
    @memset(padded_tgt, 32.0);

    const in_len = @min(input.len, cols);
    const tgt_len = @min(target.len, cols);
    for (0..in_len) |i| padded_in[i] = @floatFromInt(input[i]);
    for (0..tgt_len) |i| padded_tgt[i] = @floatFromInt(target[i]);
    try db.inputs.setRow(io, row_id, padded_in);
    try db.targets.setRow(io, row_id, padded_tgt);
}

fn rowMatchesInput(row: []const f32, input: []const u8) bool {
    const input_len = @min(input.len, row.len);
    for (0..input_len) |i| {
        if (@abs(row[i] - @as(f32, @floatFromInt(input[i]))) > 0.5) return false;
    }
    return input.len >= row.len or row[input_len] == 32.0 or row[input_len] == 0.0;
}

fn chooseIngestRow(
    io: std.Io,
    db: *DBMS.RelationDB,
    preferred_row: u64,
    input: []const u8,
    max_rows: u64,
    row_buffer: []f32,
) !u64 {
    try db.inputs.getRow(io, preferred_row, row_buffer, 0.0);
    if (row_buffer[0] == 0.0 or rowMatchesInput(row_buffer, input)) return preferred_row;

    const row_id = try db.findOrNextRow(io, input, max_rows);
    if (row_id == 0) {
        try db.inputs.getRow(io, 0, row_buffer, 0.0);
        if (row_buffer[0] != 0.0 and !rowMatchesInput(row_buffer, input)) return error.RelationCapacityReached;
    }
    return row_id;
}

fn readStoredTarget(
    allocator: std.mem.Allocator,
    io: std.Io,
    db: *DBMS.RelationDB,
    row_id: u64,
    out_text: []u8,
) ![]const u8 {
    const values = try allocator.alloc(f32, out_text.len);
    defer allocator.free(values);
    try db.targets.getRow(io, row_id, values, 32.0);
    for (values, 0..) |value, i| {
        out_text[i] = @intFromFloat(std.math.clamp(@round(value), 32.0, 126.0));
    }
    return std.mem.trimEnd(u8, out_text, " ");
}

/// Counts how many saved rows already exist in `inputs.mat` on disk
fn countExistingRowsOnDisk(allocator: std.mem.Allocator, io: std.Io, db: *DBMS.RelationDB, max_rows: u64) !u64 {
    const cols: usize = @intCast(db.cols);
    const row_floats = try allocator.alloc(f32, cols);
    defer allocator.free(row_floats);

    var count: u64 = 0;
    var r: u64 = 0;
    while (r < max_rows) : (r += 1) {
        db.inputs.getRow(io, r, row_floats, 0.0) catch break;
        var has_char = false;
        for (row_floats) |val| {
            if (val > 32.5 and val <= 126.5) {
                has_char = true;
                break;
            }
        }
        if (!has_char) break;
        count = r + 1;
    }
    return count;
}

/// Reads all files in `data/learn data/`, stores relations in `data/relations/`, and returns total rows loaded
fn ingestAllLearnData(
    allocator: std.mem.Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
    db: *DBMS.RelationDB,
    data_prefix: []const u8,
    max_rows: u64,
) !u64 {
    // Keep source rows in a stable prefix so repeated startup loads update the same facts.
    var current_row: u64 = 0;
    var max_row_id_exclusive: u64 = 0;
    const cols: usize = @intCast(db.cols);

    const padded_in = try allocator.alloc(f32, cols);
    defer allocator.free(padded_in);
    const padded_tgt = try allocator.alloc(f32, cols);
    defer allocator.free(padded_tgt);
    const existing_input = try allocator.alloc(f32, cols);
    defer allocator.free(existing_input);

    for (builtin_qa) |qa| {
        if (current_row >= max_rows) break;
        const row_id = try chooseIngestRow(io, db, current_row, qa.input, max_rows, existing_input);
        try storePaddedPair(io, db, row_id, qa.input, qa.target, padded_in, padded_tgt);
        max_row_id_exclusive = @max(max_row_id_exclusive, row_id + 1);
        current_row += 1;
    }

    var target_buf: [512]u8 = undefined;
    var path_buf: [512]u8 = undefined;

    // 2. Load all CSV and Markdown files from `<data_prefix>/learn data/`
    for (learn_files) |rel_path| {
        const full_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ data_prefix, rel_path });

        const file = std.Io.Dir.cwd().openFile(io, full_path, .{}) catch |err| {
            try stdout.print("Skipping {s} ({any})\n", .{ full_path, err });
            try stdout.flush();
            continue;
        };
        defer file.close(io);

        const stat = try file.stat(io);
        const file_size: usize = @intCast(@min(stat.size, 32 * 1024 * 1024));
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

            if (line_num == 0 and std.mem.endsWith(u8, rel_path, ".csv")) continue;

            if (splitLineIntoRelation(line, &target_buf)) |rel| {
                const row_id = try chooseIngestRow(io, db, current_row, rel.input, max_rows, existing_input);
                try storePaddedPair(io, db, row_id, rel.input, rel.target, padded_in, padded_tgt);
                max_row_id_exclusive = @max(max_row_id_exclusive, row_id + 1);
                current_row += 1;
                file_rows += 1;
            }
        }

        try stdout.print("Loaded {d} rows from \"{s}\" into '{s}/relations'\n", .{ file_rows, full_path, data_prefix });
        try stdout.flush();
    }

    return max_row_id_exclusive;
}

/// Computes letter-by-letter similarity score [0.0 .. 1.0] so "hellow" matches "hello"!
fn charSimilarityScore(a: []const u8, b: []const u8) f32 {
    if (a.len == 0 or b.len == 0) return 0.0;
    const min_len = @min(a.len, b.len);
    const max_len = @max(a.len, b.len);

    var matches: f32 = 0.0;
    for (0..min_len) |i| {
        const ca = std.ascii.toLower(a[i]);
        const cb = std.ascii.toLower(b[i]);
        if (ca == cb) {
            matches += 1.0;
        } else if (i + 1 < a.len and std.ascii.toLower(a[i + 1]) == cb) {
            matches += 0.7;
        } else if (i + 1 < b.len and ca == std.ascii.toLower(b[i + 1])) {
            matches += 0.7;
        }
    }
    return matches / @as(f32, @floatFromInt(max_len));
}

/// Searches `data/relations/inputs.mat` for exact match, case-insensitive, substring, OR fuzzy letter match
fn findBestRelationRow(
    allocator: std.mem.Allocator,
    io: std.Io,
    db: *DBMS.RelationDB,
    query: []const u8,
    max_rows: u64,
    matched_name_out: []u8,
) !?struct { row_id: u64, name_len: usize } {
    // 1. Try DBMS native exact match first
    if (try db.findRelation(io, query, max_rows)) |exact_row| {
        const copy_len = @min(query.len, matched_name_out.len);
        @memcpy(matched_name_out[0..copy_len], query[0..copy_len]);
        return .{ .row_id = exact_row, .name_len = copy_len };
    }

    // 2. Scan inputs.mat for case-insensitive, bidirectional substring, or fuzzy letter match
    const cols: usize = @intCast(db.cols);
    const row_floats = try allocator.alloc(f32, cols);
    defer allocator.free(row_floats);
    const row_chars = try allocator.alloc(u8, cols);
    defer allocator.free(row_chars);

    var partial_match: ?u64 = null;
    var partial_len: usize = 0;
    var partial_name: [256]u8 = undefined;

    var best_fuzzy_row: ?u64 = null;
    var best_fuzzy_len: usize = 0;
    var best_fuzzy_score: f32 = 0.0;
    var best_fuzzy_name: [256]u8 = undefined;

    var r: u64 = 0;
    while (r < max_rows) : (r += 1) {
        db.inputs.getRow(io, r, row_floats, 32.0) catch break;

        var non_space: usize = 0;
        for (row_floats, 0..) |val, i| {
            const c: u8 = @intFromFloat(std.math.clamp(@round(val), 32.0, 126.0));
            row_chars[i] = c;
            if (c != ' ') non_space = i + 1;
        }

        if (non_space == 0) break; // Stop scanning once we hit unpopulated rows
        const stored_key = row_chars[0..non_space];

        // Case-insensitive exact match
        if (std.ascii.eqlIgnoreCase(stored_key, query)) {
            const copy_len = @min(stored_key.len, matched_name_out.len);
            @memcpy(matched_name_out[0..copy_len], stored_key[0..copy_len]);
            return .{ .row_id = r, .name_len = copy_len };
        }

        // Bidirectional substring match (handles both "GPT" in "GPT-4" AND "hello" in "hellow")
        if (partial_match == null and (containsIgnoreCase(stored_key, query) or containsIgnoreCase(query, stored_key))) {
            partial_match = r;
            partial_len = @min(stored_key.len, partial_name.len);
            @memcpy(partial_name[0..partial_len], stored_key[0..partial_len]);
        }

        // Character-level similarity score (so typos like "hellow" or "helo" match "hello")
        const sim = charSimilarityScore(stored_key, query);
        if (sim > best_fuzzy_score and sim >= 0.60) {
            best_fuzzy_score = sim;
            best_fuzzy_row = r;
            best_fuzzy_len = @min(stored_key.len, best_fuzzy_name.len);
            @memcpy(best_fuzzy_name[0..best_fuzzy_len], stored_key[0..best_fuzzy_len]);
        }
    }

    if (best_fuzzy_row) |fr| {
        const copy_len = @min(best_fuzzy_len, matched_name_out.len);
        @memcpy(matched_name_out[0..copy_len], best_fuzzy_name[0..copy_len]);
        return .{ .row_id = fr, .name_len = copy_len };
    }

    if (partial_match) |pm| {
        const copy_len = @min(partial_len, matched_name_out.len);
        @memcpy(matched_name_out[0..copy_len], partial_name[0..copy_len]);
        return .{ .row_id = pm, .name_len = copy_len };
    }
    return null;
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

    // 1. Resolve `data` folder path automatically even if launched from `zig-out/bin/Ai.exe`!
    const data_prefix = resolveDataBasePrefix(io);
    var rel_dir_buf: [256]u8 = undefined;
    const relations_path = try std.fmt.bufPrint(&rel_dir_buf, "{s}/relations", .{data_prefix});

    const max_cols: u64 = 256;
    const max_rows: u64 = 30000;
    var db = try DBMS.RelationDB.open(io, relations_path, max_cols);
    defer db.close(io);

    var model = ModelMod.AnswerModel.init(allocator, io, &db);
    var target_buf: [512]u8 = undefined;

    // Refresh local facts every launch so changes to the source files are picked up.
    const existing_rows = try countExistingRowsOnDisk(allocator, io, &db, max_rows);
    try stdout.print("[Auto-Load] Loading local knowledge from '{s}/learn data'...\n", .{data_prefix});
    try stdout.flush();
    const startup_rows = try ingestAllLearnData(allocator, io, stdout, &db, data_prefix, max_rows);
    if (existing_rows == 0 and startup_rows > 0) {
        try model.trainAllRowsBatch(startup_rows, 15000);
        try stdout.print("[Auto-Load] Trained and saved {d} relations to '{s}'!\n", .{ startup_rows, relations_path });
    } else {
        try stdout.print("[Auto-Load] Local knowledge refreshed; ready to answer.\n", .{});
    }
    try stdout.flush();

    while (true) {
        try stdout.print("\n============================================================\n", .{});
        try stdout.print("[1] Learn Text | [2] Ask Answer (from data/relations) | [3] Learn ALL 'data/learn data' | [q] Quit\n", .{});
        const mode = try promptLine(stdout, stdin, "> ");

        if (std.mem.eql(u8, mode, "q") or std.mem.eql(u8, mode, "quit")) break;

        if (std.mem.eql(u8, mode, "3")) {
            try stdout.print("\nReading all files from '{s}/learn data/'...\n", .{data_prefix});
            try stdout.flush();

            const total_rows = try ingestAllLearnData(allocator, io, stdout, &db, data_prefix, max_rows);
            if (total_rows == 0) {
                try stdout.print("No rows found to train!\n", .{});
                try stdout.flush();
                continue;
            }

            try stdout.print("Training GPU on ALL {d} rows in parallel (15,000 epochs)...\n", .{total_rows});
            try stdout.flush();

            try model.trainAllRowsBatch(total_rows, 15000);

            try stdout.print("Done! Learned {d} rows into '{s}/' (inputs.mat, targets.mat, weights.mat)!\n", .{ total_rows, relations_path });
            try stdout.flush();
        } else if (std.mem.eql(u8, mode, "1")) {
            const raw_line = try promptLine(stdout, stdin, "Feed text to learn (e.g. 'hello : Hi there!' or 'GPT-4,Language,Text,OpenAI'): ");
            if (splitLineIntoRelation(raw_line, &target_buf)) |rel| {
                const cols: usize = @intCast(max_cols);
                const padded_in = try allocator.alloc(f32, cols);
                defer allocator.free(padded_in);
                const padded_tgt = try allocator.alloc(f32, cols);
                defer allocator.free(padded_tgt);

                const row_id = try db.findOrNextRow(io, rel.input, max_rows);
                try storePaddedPair(io, &db, row_id, rel.input, rel.target, padded_in, padded_tgt);
                try model.trainRelation(row_id, cols, 15000);

                try stdout.print("Learned & Saved Row #{d}: \"{s}\" -> \"{s}\"\n", .{ row_id, rel.input, rel.target });
                try stdout.flush();
            } else {
                try stdout.print("Could not parse relation! Use 'Key : Value' or 'Name,Domain,Task,Org'.\n", .{});
                try stdout.flush();
            }
        } else if (std.mem.eql(u8, mode, "2")) {
            const user_q = try promptLine(stdout, stdin, "Ask Model: ");
            if (user_q.len == 0) continue;

            const cols: usize = @intCast(max_cols);
            var matched_name: [256]u8 = undefined;

            const found = (try findBestRelationRow(allocator, io, &db, user_q, max_rows, &matched_name)) orelse {
                try stdout.print("Relation for \"{s}\" not found! Use [1] to teach it or [3] to load learn data.\n", .{user_q});
                try stdout.flush();
                continue;
            };

            const reply_buf = try allocator.alloc(u8, cols);
            defer allocator.free(reply_buf);
            const trimmed_reply = try readStoredTarget(allocator, io, &db, found.row_id, reply_buf);
            try stdout.print("Model Answer (Row #{d} [{s}]): {s}\n", .{
                found.row_id,
                matched_name[0..found.name_len],
                trimmed_reply,
            });
            try stdout.flush();
        }
    }
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i <= haystack.len - needle.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) {
            return true;
        }
    }
    return false;
}

test "CSV learning relations preserve quoted commas" {
    var target_buf: [512]u8 = undefined;
    const relation = splitLineIntoRelation(
        "\"Model, Name\",\"Language,Multimodal\",\"Question answering, reasoning\",\"Example Org\"",
        &target_buf,
    ) orelse return error.RelationNotParsed;

    try std.testing.expectEqualStrings("Model, Name", relation.input);
    try std.testing.expectEqualStrings("Task: Question answering, reasoning | Org: Example Org", relation.target);
}

test "stored input matching requires the complete key" {
    const row = [_]f32{ 'G', 'P', 'T', '-', '4', ' ', ' ', ' ' };
    try std.testing.expect(rowMatchesInput(&row, "GPT-4"));
    try std.testing.expect(!rowMatchesInput(&row, "GPT"));
    try std.testing.expect(!rowMatchesInput(&row, "GPT-5"));
}