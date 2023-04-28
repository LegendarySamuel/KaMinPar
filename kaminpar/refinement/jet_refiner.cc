#include "kaminpar/refinement/jet_refiner.h"

#include "kaminpar/datastructures/delta_partitioned_graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/metrics.h"
#include "kaminpar/refinement/gain_cache.h"
#include "kaminpar/refinement/greedy_balancer.h"

#include "common/logger.h"
#include "common/noinit_vector.h"
#include "common/timer.h"

namespace kaminpar::shm {
SET_DEBUG(true);
SET_STATISTICS(true);

namespace {
void perform_iteration(
    PartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    const double c,
    NoinitVector<std::uint8_t> &lock
) {
  DenseGainCache gain_cache(p_ctx.k, p_graph.n());
  gain_cache.initialize(p_graph);

  NoinitVector<BlockID> next_partition(p_graph.n());
  p_graph.pfor_nodes([&](const NodeID u) {
    const BlockID from = p_graph.block(u);

    EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();
    BlockID best_block = from;

    for (const BlockID to : p_graph.blocks()) {
      if (to == from) {
        continue;
      }

      const EdgeWeight gain = gain_cache.gain(u, from, to);
      if (gain > best_gain) {
        best_gain = gain;
        best_block = to;
      }
    }

    KASSERT(u < lock.size());
    if (!lock[u] && -best_gain < std::floor(c * gain_cache.conn(u, from))) {
      next_partition[u] = best_block;
    } else {
      next_partition[u] = from;
    }
  });

  p_graph.pfor_nodes([&](const NodeID u) { lock[u] = 0; });

  p_graph.pfor_nodes([&](const NodeID u) {
    const BlockID from = p_graph.block(u);
    const BlockID to = next_partition[u];
    if (from == to) {
      return;
    }

    const EdgeWeight gain_u = gain_cache.gain(u, from, to);
    EdgeWeight gain = 0;

    for (const auto &[e, v] : p_graph.neighbors(u)) {
      const EdgeWeight weight = p_graph.edge_weight(e);

      const bool v_before_u = [&, v = v] {
        const BlockID from_v = p_graph.block(v);
        const BlockID to_v = next_partition[v];
        if (from_v != to_v) {
          const EdgeWeight gain_v = gain_cache.gain(v, from_v, to_v);
          return gain_v > gain_u || (gain_v == gain_u && v < u);
        }
        return false;
      }();
      const BlockID block_v = v_before_u ? next_partition[v] : p_graph.block(v);

      if (to == block_v) {
        gain += weight;
      } else {
        gain -= weight;
      }
    }

    if (gain > 0) {
      p_graph.set_block(u, next_partition[u]);
      lock[u] = 1;
    }
  });
}
} // namespace

JetRefiner::JetRefiner(const Context &ctx) : _ctx(ctx) {}

bool JetRefiner::refine(
    PartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  SCOPED_TIMER("JET");

  const NodeID min_size = p_ctx.k * _ctx.coarsening.contraction_limit;
  const NodeID cur_size = p_graph.n();
  const NodeID max_size = p_ctx.n;
  const double min_c = _ctx.refinement.jet.min_c;
  const double max_c = _ctx.refinement.jet.max_c;
  const double c = [&] {
    if (_ctx.refinement.jet.interpolate_c) {
      return min_c +
             (max_c - min_c) * (cur_size - min_size) / (max_size - min_size);
    } else {
      if (cur_size <= 2 * min_size) {
        return min_c;
      } else {
        return max_c;
      }
    }
  }();
  DBG << "Set c=" << c;

  NoinitVector<std::uint8_t> lock(p_graph.n());
  p_graph.pfor_nodes([&](const NodeID u) { lock[u] = 0; });

  const EdgeWeight initial_cut = IFDBG(metrics::edge_cut(p_graph));
  const double initial_balance = IFDBG(metrics::imbalance(p_graph));
  const bool initial_feasible = IFDBG(metrics::is_feasible(p_graph, p_ctx));
  DBG << "Initial cut=" << initial_cut << ", imbalance=" << initial_balance
      << ", feasible=" << initial_feasible;

  GreedyBalancer balancer(_ctx);

  for (int i = 0; i < _ctx.refinement.jet.num_iterations; ++i) {
    perform_iteration(p_graph, p_ctx, c, lock);

    const EdgeWeight pre_rebalance_cut = IFDBG(metrics::edge_cut(p_graph));
    const double pre_rebalance_balance = IFDBG(metrics::imbalance(p_graph));
    const bool pre_rebalance_feasible =
        IFDBG(metrics::is_feasible(p_graph, p_ctx));

    DBG << "After iteration " << i
        << ", pre-rebalance: cut=" << pre_rebalance_cut
        << ", imbalance=" << pre_rebalance_balance
        << ", feasible=" << pre_rebalance_feasible;

    balancer.initialize(p_graph);
    balancer.refine(p_graph, p_ctx);

    const EdgeWeight post_rebalance_cut = IFDBG(metrics::edge_cut(p_graph));
    const double post_rebalance_balance = IFDBG(metrics::imbalance(p_graph));
    const bool post_rebalance_feasible =
        IFDBG(metrics::is_feasible(p_graph, p_ctx));

    DBG << "After iteration " << i
        << ", post-rebalance: cut=" << post_rebalance_cut
        << ", imbalance=" << post_rebalance_balance
        << ", feasible=" << post_rebalance_feasible;
  }

  return false;
}
} // namespace kaminpar::shm