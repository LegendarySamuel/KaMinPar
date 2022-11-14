/***********************************************************************************************************************
 * @file:   colored_lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   09.11.2022
 * @brief:  Distributed label propagation refiner that moves nodes in rounds determined by a graph coloring.
 **********************************************************************************************************************/
#include "dkaminpar/refinement/colored_lp_refiner.h"

#include <kassert/kassert.hpp>

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/mpi/graph_communication.h"

#include "common/parallel/algorithm.h"
#include "common/parallel/vector_ets.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
ColoredLPRefiner::ColoredLPRefiner(const Context& ctx) : _input_ctx(ctx) {}

void ColoredLPRefiner::initialize(const DistributedGraph& graph) {
    SCOPED_TIMER("Initialize colorized label propagation refiner");

    const auto    coloring   = compute_node_coloring_sequentially(graph, _input_ctx.refinement.lp.num_chunks);
    const ColorID num_colors = *std::max_element(coloring.begin(), coloring.end());

    TIMED_SCOPE("Allocation") {
        _color_sorted_nodes.resize(graph.n());
        _color_sizes.resize(num_colors + 1);
        tbb::parallel_for<std::size_t>(0, _color_sorted_nodes.size(), [&](const std::size_t i) {
            _color_sorted_nodes[i] = 0;
        });
        tbb::parallel_for<std::size_t>(0, _color_sizes.size(), [&](const std::size_t i) { _color_sizes[i] = 0; });
    };

    TIMED_SCOPE("Count color sizes") {
        graph.pfor_nodes([&](const NodeID u) {
            const ColorID c = coloring[u];
            KASSERT(c < num_colors);
            __atomic_fetch_add(&_color_sizes[c], 1, __ATOMIC_RELAXED);
        });
        parallel::prefix_sum(_color_sizes.begin(), _color_sizes.end(), _color_sizes.begin());
    };

    TIMED_SCOPE("Sort nodes") {
        graph.pfor_nodes([&](const NodeID u) {
            const ColorID     c = coloring[u];
            const std::size_t i = __atomic_sub_fetch(&_color_sizes[c], 1, __ATOMIC_SEQ_CST);
            KASSERT(i < _color_sorted_nodes.size());
            _color_sorted_nodes[i] = u;
        });
    };

    KASSERT(_color_sizes.front() == 0u);
    KASSERT(_color_sizes.back() == graph.n());
}

void ColoredLPRefiner::refine(DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) {
    _p_ctx   = &p_ctx;
    _p_graph = &p_graph;

    [[maybe_unused]] NodeID local_num_moves = 0;

    for (ColorID c = 0; c + 1 < _color_sizes.size(); ++c) {
        local_num_moves += find_moves(c);
        perform_moves(c);
    }
}

void ColoredLPRefiner::perform_moves(const ColorID c) {
    const NodeID seq_from = _color_sizes[c];
    const NodeID seq_to   = _color_sizes[c + 1];

    const auto block_gains = TIMED_SCOPE("Gather gain values") {
        if (_input_ctx.refinement.lp.ignore_probabilities) {
            return BlockGainsContainer{};
        }

        parallel::vector_ets<EdgeWeight> block_gains_ets(_p_ctx->k);
        _p_graph->pfor_nodes_range(seq_from, seq_to, [&](const auto r) {
            auto& block_gains = block_gains_ets.local();

            for (const NodeID seq_u: r) {
                const NodeID u = _color_sorted_nodes[seq_u];
                if (_p_graph->block(u) != _next_partition[seq_u]) {
                    block_gains[_next_partition[seq_u]] += _gains[seq_u];
                }
            }
        });

        auto block_gains = block_gains_ets.combine(std::plus{});

        MPI_Allreduce(
            MPI_IN_PLACE, block_gains.data(), asserting_cast<int>(_p_ctx->k), mpi::type::get<EdgeWeight>(), MPI_SUM,
            _p_graph->communicator()
        );

        return block_gains;
    };

    TIMED_SCOPE("Perform moves") {
        for (std::size_t i = 0; i < _input_ctx.refinement.lp.num_move_attempts; ++i) {
            if (attempt_moves(c, block_gains)) {
                break;
            }
        }
        synchronize_state(c);
    };

    // Reset _next_partition for next round
    TIMED_SCOPE("Reset partition array") {
        _p_graph->pfor_nodes(seq_from, seq_to, [&](const NodeID seq_u) {
            const NodeID u         = _color_sorted_nodes[seq_u];
            _next_partition[seq_u] = _p_graph->block(u);
        });
    };
}

bool ColoredLPRefiner::attempt_moves(const ColorID c, const BlockGainsContainer& block_gains) {
    struct Move {
        Move(const NodeID seq_u, const NodeID u, const BlockID from) : seq_u(seq_u), u(u), from(from) {}
        NodeID  seq_u;
        NodeID  u;
        BlockID from;
    };

    // Keep track of the moves that we perform so that we can roll back in case the probabilistic moves made the
    // partition imbalanced
    tbb::concurrent_vector<Move> moves;

    // Track change in block weights to determine whether the partition became imbalanced
    NoinitVector<BlockWeight> block_weight_deltas(_p_ctx->k);
    tbb::parallel_for<BlockID>(0, _p_ctx->k, [&](const BlockID b) { block_weight_deltas[b] = 0; });

    const NodeID seq_from = _color_sizes[c];
    const NodeID seq_to   = _color_sizes[c + 1];

    _p_graph->pfor_nodes_range(seq_from, seq_to, [&](const auto& r) {
        auto& rand = Random::instance();

        for (const NodeID seq_u: r) {
            const NodeID u = _color_sorted_nodes[seq_u];

            // Only iterate over nodes that changed block
            if (_next_partition[seq_u] == _p_graph->block(u) || _next_partition[seq_u] == kInvalidBlockID) {
                continue;
            }

            // Compute move probability and perform it
            // Or always perform the move if move probabilities are disabled
            const BlockID to          = _next_partition[seq_u];
            const double  probability = [&] {
                if (_input_ctx.refinement.lp.ignore_probabilities) {
                    return 1.0;
                }

                const double      gain_prob = (block_gains[to] == 0) ? 1.0 : 1.0 * _gains[seq_u] / block_gains[to];
                const BlockWeight residual_block_weight =
                    _p_ctx->graph.max_block_weight(to) - _p_graph->block_weight(to);
                return gain_prob * residual_block_weight / _p_graph->node_weight(u);
            }();

            if (_input_ctx.refinement.lp.ignore_probabilities || rand.random_bool(probability)) {
                const BlockID    from     = _p_graph->block(u);
                const NodeWeight u_weight = _p_graph->node_weight(u);

                moves.emplace_back(seq_u, u, from);
                __atomic_fetch_sub(&block_weight_deltas[from], u_weight, __ATOMIC_RELAXED);
                __atomic_fetch_add(&block_weight_deltas[to], u_weight, __ATOMIC_RELAXED);
                _p_graph->set_block<false>(u, to);

                // Temporary mark that this node was actually moved
                // We will revert this during synchronization or on rollback
                _next_partition[seq_u] = kInvalidBlockID;
            }
        }
    });

    // Compute global block weights after moves
    MPI_Allreduce(
        MPI_IN_PLACE, block_weight_deltas.data(), asserting_cast<int>(_p_ctx->k), mpi::type::get<BlockWeight>(),
        MPI_SUM, _p_graph->communicator()
    );

    // Check for balance violations
    parallel::Atomic<std::uint8_t> feasible = 1;
    if (!_input_ctx.refinement.lp.ignore_probabilities) {
        _p_graph->pfor_blocks([&](const BlockID b) {
            // If blocks were already overloaded before refinement, accept it as feasible if their weight did not
            // increase (i.e., delta is <= 0) == first part of this if condition
            if (block_weight_deltas[b] > 0
                && _p_graph->block_weight(b) + block_weight_deltas[b] > _p_ctx->graph.max_block_weight(b)) {
                feasible = 0;
            }
        });
    }

    // Revert moves if resulting partition is infeasible
    // Otherwise, update block weights cached in the graph data structure
    if (!feasible) {
        tbb::parallel_for(moves.range(), [&](const auto r) {
            for (const auto& [seq_u, u, from]: r) {
                _next_partition[seq_u] = _p_graph->block(u);
                _p_graph->set_block<false>(u, from);
            }
        });
    } else {
        _p_graph->pfor_blocks([&](const BlockID b) {
            _p_graph->set_block_weight(b, _p_graph->block_weight(b) + block_weight_deltas[b]);
        });
    }

    return feasible;
}

void ColoredLPRefiner::synchronize_state(const ColorID c) {
    struct MoveMessage {
        NodeID  local_node;
        BlockID new_block;
    };

    const NodeID seq_from = _color_sizes[c];
    const NodeID seq_to   = _color_sizes[c + 1];

    mpi::graph::sparse_alltoall_interface_to_pe_custom_range<MoveMessage>(
        _p_graph->graph(), seq_from, seq_to,

        // Map sequence index to node
        [&](const NodeID seq_u) { return _color_sorted_nodes[seq_u]; },

        // We set _next_partition[] to kInvalidBlockID for nodes that were moved during perform_moves()
        [&](const NodeID u) -> bool { return _next_partition[u] == kInvalidBlockID; },

        // Send move to each ghost node adjacent to u
        [&](const NodeID u) -> MoveMessage {
            // perform_moves() marks nodes that were moved locally by setting _next_partition[] to kInvalidBlockID
            // here, we revert this mark
            _next_partition[u] = _p_graph->block(u);

            return {.local_node = u, .new_block = _p_graph->block(u)};
        },

        // Move ghost nodes
        [&](const auto recv_buffer, const PEID pe) {
            tbb::parallel_for(static_cast<std::size_t>(0), recv_buffer.size(), [&](const std::size_t i) {
                const auto [local_node_on_pe, new_block] = recv_buffer[i];
                const auto   global_node = static_cast<GlobalNodeID>(_p_graph->offset_n(pe) + local_node_on_pe);
                const NodeID local_node  = _p_graph->global_to_local_node(global_node);
                KASSERT(new_block != _p_graph->block(local_node)); // otherwise, we should not have gotten this message

                _p_graph->set_block<false>(local_node, new_block);
            });
        }
    );
}

void ColoredLPRefiner::handle_node(const NodeID u) {}
} // namespace kaminpar::dist
