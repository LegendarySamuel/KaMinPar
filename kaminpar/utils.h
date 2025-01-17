/*******************************************************************************
 * @file:   utils.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 * @brief:  Unsorted utility functions (@todo)
 ******************************************************************************/
#pragma once

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/kaminpar.h"

#include "common/math.h"

namespace kaminpar::shm {
template <typename CoarseningContext_, typename NodeID_ = NodeID, typename NodeWeight_ = NodeWeight>
NodeWeight_ compute_max_cluster_weight(
    const NodeID_ n,
    const NodeWeight_ total_node_weight,
    const PartitionContext &input_p_ctx,
    const CoarseningContext_ &c_ctx
) {
  double max_cluster_weight = 0.0;

  switch (c_ctx.cluster_weight_limit) {
  case ClusterWeightLimit::EPSILON_BLOCK_WEIGHT:
    max_cluster_weight = (input_p_ctx.epsilon * total_node_weight) /
                         std::clamp<BlockID>(n / c_ctx.contraction_limit, 2, input_p_ctx.k);
    break;

  case ClusterWeightLimit::BLOCK_WEIGHT:
    max_cluster_weight = (1.0 + input_p_ctx.epsilon) * total_node_weight / input_p_ctx.k;
    break;

  case ClusterWeightLimit::ONE:
    max_cluster_weight = 1.0;
    break;

  case ClusterWeightLimit::ZERO:
    max_cluster_weight = 0.0;
    break;
  }

  return static_cast<NodeWeight_>(max_cluster_weight * c_ctx.cluster_weight_multiplier);
}
template <typename NodeID_ = NodeID, typename NodeWeight_ = NodeWeight>
NodeWeight_ compute_max_cluster_weight(
    const NodeID_ n,
    const NodeWeight_ total_node_weight,
    const PartitionContext &input_p_ctx,
    const CoarseningContext &c_ctx
) {
  double max_cluster_weight = 0.0;

  switch (c_ctx.cluster_weight_limit) {
  case ClusterWeightLimit::EPSILON_BLOCK_WEIGHT:
    max_cluster_weight = (input_p_ctx.epsilon * total_node_weight) /
                         std::clamp<BlockID>(n / c_ctx.contraction_limit, 2, input_p_ctx.k);
    break;

  case ClusterWeightLimit::BLOCK_WEIGHT:
    max_cluster_weight = (1.0 + input_p_ctx.epsilon) * total_node_weight / input_p_ctx.k;
    break;

  case ClusterWeightLimit::ONE:
    max_cluster_weight = 1.0;
    break;

  case ClusterWeightLimit::ZERO:
    max_cluster_weight = 0.0;
    break;
  }

  return static_cast<NodeWeight_>(max_cluster_weight * c_ctx.cluster_weight_multiplier);
}

template <typename CoarseningContext_, typename NodeWeight_ = NodeWeight, typename Graph_ = Graph>
NodeWeight_ compute_max_cluster_weight(
    const Graph_ &c_graph, const PartitionContext &input_p_ctx, const CoarseningContext_ &c_ctx
) {
  return compute_max_cluster_weight(c_graph.n(), c_graph.total_node_weight(), input_p_ctx, c_ctx);
}

inline double compute_2way_adaptive_epsilon(
    const PartitionContext &p_ctx,
    const NodeWeight subgraph_total_node_weight,
    const BlockID subgraph_final_k
) {
  KASSERT(subgraph_final_k > 1u);

  const double base = (1.0 + p_ctx.epsilon) * subgraph_final_k * p_ctx.total_node_weight / p_ctx.k /
                      subgraph_total_node_weight;
  const double exponent = 1.0 / math::ceil_log2(subgraph_final_k);
  const double epsilon_prime = std::pow(base, exponent) - 1.0;
  const double adaptive_epsilon = std::max(epsilon_prime, 0.0001);
  return adaptive_epsilon;
}

inline PartitionContext create_bipartition_context(
    const PartitionContext &k_p_ctx,
    const Graph &subgraph,
    const BlockID final_k1,
    const BlockID final_k2
) {
  PartitionContext two_p_ctx{};
  two_p_ctx.k = 2;
  two_p_ctx.setup(subgraph);
  two_p_ctx.epsilon =
      compute_2way_adaptive_epsilon(k_p_ctx, subgraph.total_node_weight(), final_k1 + final_k2);
  two_p_ctx.block_weights.setup(two_p_ctx, {final_k1, final_k2});
  return two_p_ctx;
}
} // namespace kaminpar::shm
