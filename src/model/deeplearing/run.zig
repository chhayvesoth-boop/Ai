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

const builtin_qa = [_]struct { input: []const u8, target: []const u8 }{
    .{ .input = "hello", .target = "Hello! I am your GPU Matrix Relation AI. Ask me about any AI model!" },
    .{ .input = "hi", .target = "Hi there! Ask me about any model in data/learn data." },
    .{ .input = "who are you", .target = "I am a Zig + CUDA PTX Neural Matrix DB trained on Epoch AI datasets." },
    .{ .input = "help", .target = "Type an AI model name (e.g. GPT-4, AlexNet, AlphaGo, Claude, Gemini)." },
};

fn promptLine(stdout: *std.Io.Writer, stdin: *std.Io.Reader, prompt_msg: []const u8) ![]const u8 {
    try stdout.print("{s}", .{prompt_msg});
    try stdout.flush();
    const raw = try stdin.takeDelimiterInclusive('\n');
    return std.mem.trim(u8, raw, "\r\n\x00 ");
}

/// Splits a CSV or Markdown line into (Input_A, Target_B) using a caller-provided buffer
fn splitLineIntoRelation(line: []const u8, out_target_buf: []u8) ?struct { input: []const u8, target: []const u8 } {
    if (std.mem.indexOfScalar(u8, line, ',') == null) {
        if (std.mem.indexOfAny(u8, line, ":-=")) |sep_idx| {
            const k = std.mem.trim(u8, line[0..sep_idx], " #*\"'\t");
            const v = std.mem.trim(u8, line[sep_idx + 1 ..], " #*\"'\t");
            if (k.len == 0 or v.len == 0) return null;
            const target_str = std.fmt.bufPrint(out_target_buf, "{s}", .{v}) catch return null;
            return .{ .input = k, .target = target_str };
        }
        return null;
    }

    var iter = std.mem.splitScalar(u8, line, ',');

    const model_name = iter.next() orelse return null;
    const clean_name = std.mem.trim(u8, model_name, " \"'\t");
    if (clean_name.len == 0) return null;

    const domain = iter.next() orelse "";
    const task = iter.next() orelse domain;
    const org = iter.next() orelse "Unknown Org";

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
    padded_in: []u8,
    padded_tgt: []u8,
) !void {
    const cols = padded_in.len;
    @memset(padded_in, ' ');
    @memset(padded_tgt, ' ');

    const in_len = @min(input.len, cols);
    const tgt_len = @min(target.len, cols);
    @memcpy(padded_in[0..in_len], input[0..in_len]);
    @memcpy(padded_tgt[0..tgt_len], target[0..tgt_len]);

    try db.storeTextRelation(io, row_id, padded_in, padded_tgt);
}

/// Reads all files in `data/learn data/`, stores relations in `data/relations/`, and returns total rows loaded
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

    // 1. Seed conversational relations ("hello", "hi", "help", etc.)
    for (builtin_qa) |qa| {
        if (current_row >= max_rows) break;
        try storePaddedPair(io, db, current_row, qa.input, qa.target, padded_in, padded_tgt);
        current_row += 1;
    }

    var target_buf: [512]u8 = undefined;

    // 2. Load all CSV and Markdown files from `data/learn data/`
    for (learn_files) |file_path| {
        const file = std.Io.Dir.cwd().openFile(io, file_path, .{}) catch |err| {
            try stdout.print("Skipping {s} ({any})\n", .{ file_path, err });
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

            if (line_num == 0 and std.mem.endsWith(u8, file_path, ".csv")) continue;

            if (splitLineIntoRelation(line, &target_buf)) |rel| {
                try storePaddedPair(io, db, current_row, rel.input, rel.target, padded_in, padded_tgt);
                current_row += 1;
                file_rows += 1;
            }
        }

        try stdout.print("Loaded {d} rows from \"{s}\" into 'data/relations'\n", .{ file_rows, file_path });
        try stdout.flush();
    }

    return current_row;
}

/// Searches `data/relations/inputs.mat` for exact match, case-insensitive match, or substring match
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

    // 2. Scan inputs.mat for case-insensitive or partial match
    const cols: usize = @intCast(db.cols);
    const row_floats = try allocator.alloc(f32, cols);
    defer allocator.free(row_floats);
    const row_chars = try allocator.alloc(u8, cols);
    defer allocator.free(row_chars);

    var partial_match: ?u64 = null;
    var partial_len: usize = 0;

    var r: u64 = 0;
    while (r < max_rows) : (r += 1) {
        db.inputs.getRow(io, r, row_floats, 32.0) catch break;

        var non_space: usize = 0;
        for (row_floats, 0..) |val, i| {
            const c: u8 = @intFromFloat(std.math.clamp(@round(val), 32.0, 126.0));
            row_chars[i] = c;
            if (c != ' ') non_space = i + 1;
        }

        if (non_space == 0) continue;
        const stored_key = row_chars[0..non_space];

        // Case-insensitive exact match
        if (std.ascii.eqlIgnoreCase(stored_key, query)) {
            const copy_len = @min(stored_key.len, matched_name_out.len);
            @memcpy(matched_name_out[0..copy_len], stored_key[0..copy_len]);
            return .{ .row_id = r, .name_len = copy_len };
        }

        // Case-insensitive substring match (keep first found)
        // Case-insensitive substring match (keep first found)
        if (partial_match == null and containsIgnoreCase(stored_key, query)) {
            partial_match = r;
            partial_len = @min(stored_key.len, matched_name_out.len);
            @memcpy(matched_name_out[0..partial_len], stored_key[0..partial_len]);
        }
    }

    if (partial_match) |pm| {
        return .{ .row_id = pm, .name_len = partial_len };
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

    const max_cols: u64 = 256;
    const max_rows: u64 = 30000; // Covers all 23,700+ rows in data/learn data
    var db = try DBMS.RelationDB.open(io, "data/relations", max_cols);
    defer db.close(io);

    var model = ModelMod.AnswerModel.init(allocator, io, &db);
    var target_buf: [512]u8 = undefined;

    while (true) {
        try stdout.print("\n============================================================\n", .{});
        try stdout.print("[1] Learn Text | [2] Ask Answer (from data/relations) | [3] Learn ALL 'data/learn data' | [q] Quit\n", .{});
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

            try stdout.print("Training GPU on ALL {d} rows in parallel (15,000 epochs)...\n", .{total_rows});
            try stdout.flush();

            try model.trainAllRowsBatch(total_rows, 15000);

            try stdout.print("Done! Learned {d} rows into 'data/relations/' (inputs.mat, targets.mat, weights.mat)!\n", .{total_rows});
            try stdout.flush();
        } else if (std.mem.eql(u8, mode, "1")) {
            const raw_line = try promptLine(stdout, stdin, "Feed text to learn (e.g. 'hello : Hi there!' or 'GPT-4,Language,Text,OpenAI'): ");
            if (splitLineIntoRelation(raw_line, &target_buf)) |rel| {
                const cols: usize = @intCast(max_cols);
                const padded_in = try allocator.alloc(u8, cols);
                defer allocator.free(padded_in);
                const padded_tgt = try allocator.alloc(u8, cols);
                defer allocator.free(padded_tgt);

                const row_id = try db.findOrNextRow(io, rel.input, max_rows);
                try storePaddedPair(io, &db, row_id, rel.input, rel.target, padded_in, padded_tgt);
                try model.trainRelation(row_id, cols, 10000);

                try stdout.print("Learned Row #{d}: \"{s}\" -> \"{s}\"\n", .{ row_id, rel.input, rel.target });
                try stdout.flush();
            } else {
                try stdout.print("Could not parse relation! Use 'Key : Value' or 'Name,Domain,Task,Org'.\n", .{});
                try stdout.flush();
            }
        } else if (std.mem.eql(u8, mode, "2")) {
            const user_q = try promptLine(stdout, stdin, "Ask Model (e.g. 'hello', 'GPT-4', 'AlphaGo', 'Claude'): ");
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
            @memset(reply_buf, ' ');

            // Run GPU inference using weights.mat via AnswerModel.answer()
            try model.answer(found.row_id, matched_name[0..found.name_len], reply_buf);

            const trimmed_reply = std.mem.trimEnd(u8, reply_buf, " ");
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