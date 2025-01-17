/*******************************************************************************
 * @file:   context.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Configuration struct for KaMinPar.
 ******************************************************************************/
#include "kaminpar/context.h"

#include <iomanip>
#include <unordered_map>

#include <kassert/kassert.hpp>

#include "kaminpar/datastructures/graph.h"

#include "common/asserting_cast.h"
#include "common/console_io.h"
#include "common/math.h"

namespace kaminpar::shm {
//
// PartitionContext
//

void PartitionContext::setup(const Graph &graph) {
  n = graph.n();
  m = graph.m();
  total_node_weight = graph.total_node_weight();
  total_edge_weight = graph.total_edge_weight();
  max_node_weight = graph.max_node_weight();
  setup_block_weights();
}

void PartitionContext::setup_block_weights() {
  block_weights.setup(*this);
}

//
// BlockWeightsContext
//

void BlockWeightsContext::setup(const PartitionContext &p_ctx) {
  KASSERT(p_ctx.k != kInvalidBlockID, "PartitionContext::k not initialized");
  KASSERT(p_ctx.k != 0u, "PartitionContext::k not initialized");
  KASSERT(
      p_ctx.total_node_weight != kInvalidNodeWeight,
      "PartitionContext::total_node_weight not initialized"
  );
  KASSERT(
      p_ctx.max_node_weight != kInvalidNodeWeight,
      "PartitionContext::max_node_weight not initialized"
  );

  const auto perfectly_balanced_block_weight =
      static_cast<NodeWeight>(std::ceil(1.0 * p_ctx.total_node_weight / p_ctx.k));
  const auto max_block_weight =
      static_cast<NodeWeight>((1.0 + p_ctx.epsilon) * perfectly_balanced_block_weight);

  _max_block_weights.resize(p_ctx.k);
  _perfectly_balanced_block_weights.resize(p_ctx.k);

  tbb::parallel_for<BlockID>(0, p_ctx.k, [&](const BlockID b) {
    _perfectly_balanced_block_weights[b] = perfectly_balanced_block_weight;

    // relax balance constraint by max_node_weight on coarse levels only
    if (p_ctx.max_node_weight == 1) {
      _max_block_weights[b] = max_block_weight;
    } else {
      _max_block_weights[b] = std::max<NodeWeight>(
          max_block_weight, perfectly_balanced_block_weight + p_ctx.max_node_weight
      );
    }
  });
}

void BlockWeightsContext::setup(
    const PartitionContext &p_ctx, const std::vector<BlockID> &final_ks
) {
  KASSERT(p_ctx.k != kInvalidBlockID, "PartitionContext::k not initialized");
  KASSERT(
      p_ctx.total_node_weight != kInvalidNodeWeight,
      "PartitionContext::total_node_weight not initialized"
  );
  KASSERT(
      p_ctx.max_node_weight != kInvalidNodeWeight,
      "PartitionContext::max_node_weight not initialized"
  );
  KASSERT(
      p_ctx.k == final_ks.size(),
      "bad number of blocks: got " << final_ks.size() << ", expected " << p_ctx.k
  );

  const BlockID final_k =
      std::accumulate(final_ks.begin(), final_ks.end(), static_cast<BlockID>(0));
  const double block_weight = 1.0 * p_ctx.total_node_weight / final_k;

  _max_block_weights.resize(p_ctx.k);
  _perfectly_balanced_block_weights.resize(p_ctx.k);

  tbb::parallel_for<BlockID>(0, final_ks.size(), [&](const BlockID b) {
    _perfectly_balanced_block_weights[b] = std::ceil(final_ks[b] * block_weight);

    const auto max_block_weight =
        static_cast<BlockWeight>((1.0 + p_ctx.epsilon) * _perfectly_balanced_block_weights[b]);

    // relax balance constraint by max_node_weight on coarse levels only
    if (p_ctx.max_node_weight == 1) {
      _max_block_weights[b] = max_block_weight;
    } else {
      _max_block_weights[b] = std::max<BlockWeight>(
          max_block_weight, _perfectly_balanced_block_weights[b] + p_ctx.max_node_weight
      );
    }
  });
}

[[nodiscard]] BlockWeight BlockWeightsContext::max(const BlockID b) const {
  KASSERT(b < _max_block_weights.size());
  return _max_block_weights[b];
}

[[nodiscard]] const std::vector<BlockWeight> &BlockWeightsContext::all_max() const {
  return _max_block_weights;
}

[[nodiscard]] BlockWeight BlockWeightsContext::perfectly_balanced(const BlockID b) const {
  KASSERT(b < _perfectly_balanced_block_weights.size());
  return _perfectly_balanced_block_weights[b];
}

[[nodiscard]] const std::vector<BlockWeight> &BlockWeightsContext::all_perfectly_balanced() const {
  return _perfectly_balanced_block_weights;
}

void Context::setup(const Graph &graph) {
  partition.setup(graph);
}
} // namespace kaminpar::shm
