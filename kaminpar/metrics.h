/*******************************************************************************
 * @file:   metrics.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Functions to compute partition quality metric.s
 ******************************************************************************/
#pragma once

#include <cmath>
#include <numeric>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/definitions.h"

namespace kaminpar::shm::metrics {
template <typename PartitionedGraphType>
EdgeWeight edge_cut(const PartitionedGraphType &p_graph, tag::Parallel) {
  tbb::enumerable_thread_specific<int64_t> cut_ets{0};
  tbb::parallel_for(tbb::blocked_range(static_cast<NodeID>(0), p_graph.n()), [&](const auto &r) {
    auto &cut = cut_ets.local();
    for (NodeID u = r.begin(); u < r.end(); ++u) {
      for (const auto &[e, v] : p_graph.neighbors(u)) {
        cut += (p_graph.block(u) != p_graph.block(v)) ? p_graph.edge_weight(e) : 0;
      }
    }
  });

  int64_t global_cut = cut_ets.combine(std::plus<>{});
  KASSERT(global_cut % 2 == 0u);
  global_cut /= 2;
  KASSERT(
      (0 <= global_cut && global_cut <= std::numeric_limits<EdgeWeight>::max()),
      "edge cut overflows: " << global_cut,
      assert::always
  );
  return static_cast<EdgeWeight>(global_cut);
}

template <typename PartitionedGraphType>
EdgeWeight edge_cut(const PartitionedGraphType &p_graph, tag::Sequential) {
  int64_t cut{0};

  for (const NodeID u : p_graph.nodes()) {
    for (const auto &[e, v] : p_graph.neighbors(u)) {
      cut += (p_graph.block(u) != p_graph.block(v)) ? p_graph.edge_weight(e) : 0;
    }
  }

  KASSERT(cut % 2 == 0u);
  cut /= 2;
  KASSERT(
      (0 <= cut && cut <= std::numeric_limits<EdgeWeight>::max()),
      "edge cut overflows: " << cut,
      assert::always
  );
  return static_cast<EdgeWeight>(cut);
}

template <typename PartitionedGraphType> EdgeWeight edge_cut(const PartitionedGraphType &p_graph) {
  return edge_cut(p_graph, tag::par);
}

template <typename PartitionedGraphType> double imbalance(const PartitionedGraphType &p_graph) {
  const NodeWeight total_weight = p_graph.total_node_weight();
  const double perfect_block_weight = std::ceil(static_cast<double>(total_weight) / p_graph.k());

  double max_imbalance = 0.0;
  for (const BlockID b : p_graph.blocks()) {
    max_imbalance = std::max(max_imbalance, p_graph.block_weight(b) / perfect_block_weight - 1.0);
  }

  return max_imbalance;
}

template <typename PartitionedGraphType>
NodeWeight total_overload(const PartitionedGraphType &p_graph, const PartitionContext &context) {
  NodeWeight total_overload = 0;
  for (const BlockID b : p_graph.blocks()) {
    total_overload +=
        std::max<BlockWeight>(0, p_graph.block_weight(b) - context.block_weights.max(b));
  }
  return total_overload;
}

template <typename PartitionedGraphType>
bool is_balanced(const PartitionedGraphType &p_graph, const PartitionContext &p_ctx) {
  return std::all_of(
      p_graph.blocks().begin(),
      p_graph.blocks().end(),
      [&p_graph, &p_ctx](const BlockID b) {
        return p_graph.block_weight(b) <= p_ctx.block_weights.max(b);
      }
  );
}

template <typename PartitionedGraphType>
bool is_feasible(const PartitionedGraphType &p_graph, const BlockID input_k, const double eps) {
  const double max_block_weight = std::ceil((1.0 + eps) * p_graph.total_node_weight() / input_k);
  return std::all_of(
      p_graph.blocks().begin(),
      p_graph.blocks().end(),
      [&p_graph, max_block_weight](const BlockID b) {
        return p_graph.block_weight(b) <=
               max_block_weight * p_graph.final_k(b) + p_graph.max_node_weight();
      }
  );
}

template <typename PartitionedGraphType>
bool is_feasible(const PartitionedGraphType &p_graph, const PartitionContext &p_ctx) {
  return is_balanced(p_graph, p_ctx);
}
} // namespace kaminpar::shm::metrics
