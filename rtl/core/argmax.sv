// int8_mlp: INT8 MLP accelerator on a weight-stationary systolic array (Artix-7)
//
// Description:
//   Finds the largest score in one block of Lanes scores and merges it with
//   the best score of earlier blocks, so a layer wider than the array can be
//   searched one column block at a time.
//   Ties go to the lowest index, like np.argmax: a later candidate replaces
//   an earlier one only if it is strictly larger. Lanes with lane_valid_i
//   low (padding neurons) never win.
//   Latency = log2(Lanes) + 1 cycles (4 for 8 lanes), one compare per stage:
//     1 .. log2(Lanes): one level of a tree of pairwise compares, halving
//        the candidates each stage; in each pair the left candidate has the
//        lower index. One 24-bit compare per stage keeps 150 MHz; three in
//        one cycle missed it by 1.8 ns.
//     last: compare the block's best with the earlier best (prev_*_i), which
//        always has lower indices than this block
//   With use_prev_i low (first block) the earlier best is ignored.
//   No reset: datapath only; the caller tracks which cycles are valid.

// svstyle: waive LNG-014 Vivado xsim rejects `input logic` ports under `default_nettype none
module argmax #(
    parameter int unsigned Lanes      = 8,   // Scores per block; a power of two
    parameter int unsigned ScoreWidth = 24,  // Score width (signed)
    parameter int unsigned IndexWidth = 4    // Width of a score's index (digit)
) (
    input  logic                         clk_i,         // System clock
    input  logic [Lanes-1:0][ScoreWidth-1:0] scores_i,  // This block's scores, signed
    input  logic [Lanes-1:0]             lane_valid_i,  // Lane holds a real score
    input  logic [IndexWidth-1:0]        base_index_i,  // Index of lane 0
    input  logic                         use_prev_i,    // Merge with the earlier best
    input  logic signed [ScoreWidth-1:0] prev_score_i,  // Best score of the earlier blocks
    input  logic [IndexWidth-1:0]        prev_index_i,  // Its index
    output logic signed [ScoreWidth-1:0] best_score_o,  // Best score so far, log2(Lanes)+1 later
    output logic [IndexWidth-1:0]        best_index_o   // Its index; lowest index on a tie
);
    // The pairwise tree needs a power of two.
    if (Lanes < 2 || (Lanes & (Lanes - 1)) != 0) begin : gen_lanes_check
        $error("Lanes must be a power of two, at least 2");
    end

    localparam int unsigned Levels = $clog2(Lanes);  // Tree depth: 3 for 8 lanes

    // One candidate: a score, its index, and whether it is real.
    typedef struct packed {
        logic                         valid;  // Real score, may win
        logic signed [ScoreWidth-1:0] score;  // The score
        logic [IndexWidth-1:0]        index;  // Its index
    } candidate_t;

    // The winner of a pair. right has the higher index, so it wins only if
    // it is strictly larger: ties keep the lower index.
    function automatic candidate_t pick(input candidate_t left, input candidate_t right);
        pick = left;
        if (right.valid && (!left.valid || (right.score > left.score))) begin
            pick = right;
        end
    endfunction

    // ---- Stages 1 .. Levels: the tree, one registered level per stage ----
    candidate_t [Lanes-1:0] leaf;  // One candidate per input lane

    // The inputs as candidates; lane n has index base_index_i + n.
    for (genvar lane = 0; lane < Lanes; lane++) begin : gen_leaf
        assign leaf[lane].valid = lane_valid_i[lane];
        assign leaf[lane].score = $signed(scores_i[lane]);
        assign leaf[lane].index = base_index_i + IndexWidth'(lane);
    end

    // tree_q[s] holds tree level s+1: Lanes >> (s+1) winners in the low entries.
    candidate_t [Lanes-1:0] tree_q[Levels];  // Registered tree levels

    for (genvar stage = 0; stage < Levels; stage++) begin : gen_tree
        candidate_t [Lanes-1:0] below;  // The level this stage compares

        if (stage == 0) begin : gen_from_leaves
            assign below = leaf;
        end else begin : gen_from_tree
            assign below = tree_q[stage-1];
        end

        // Keep the winner of each pair; entries past the last pair stay zero.
        always_ff @(posedge clk_i) begin
            for (int pair = 0; pair < Lanes; pair++) begin
                if (pair < (Lanes >> (stage + 1))) begin
                    tree_q[stage][pair] <= pick(below[2*pair], below[2*pair+1]);
                end else begin
                    tree_q[stage][pair] <= '0;
                end
            end
        end
    end

    // The earlier best waits beside the tree, Levels cycles.
    candidate_t prev_q[Levels];  // prev_q[s]: the earlier best, s+1 cycles old
    logic [Levels-1:0] use_prev_q;  // use_prev_i, delayed the same way

    // Shift the earlier best along with the tree.
    always_ff @(posedge clk_i) begin
        prev_q[0].valid <= use_prev_i;
        prev_q[0].score <= prev_score_i;
        prev_q[0].index <= prev_index_i;
        use_prev_q[0]   <= use_prev_i;
        for (int stage = 1; stage < Levels; stage++) begin
            prev_q[stage]     <= prev_q[stage-1];
            use_prev_q[stage] <= use_prev_q[stage-1];
        end
    end

    // ---- Last stage: merge with the earlier blocks ----
    candidate_t block_best;  // Best candidate of this block, out of the tree
    assign block_best = tree_q[Levels-1][0];

    candidate_t best_q, best_d;  // Best candidate over all blocks so far

    // The earlier best has the lower indices, so it is the left side of the pair.
    always_comb begin
        best_d = block_best;
        if (use_prev_q[Levels-1]) begin
            best_d = pick(prev_q[Levels-1], block_best);
        end
    end

    // Register the final winner.
    always_ff @(posedge clk_i) begin
        best_q <= best_d;
    end

    assign best_score_o = best_q.score;
    assign best_index_o = best_q.index;
endmodule

`resetall
