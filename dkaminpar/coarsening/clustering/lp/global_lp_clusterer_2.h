/*******************************************************************************
 * Label propagation with clusters that can grow to multiple PEs.
 *
 * @file:   global_lp_clusterer.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering/clusterer.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class GlobalLPClusterer2 : public Clusterer<GlobalNodeID> {
public:
  explicit GlobalLPClusterer2(const Context &ctx);

  GlobalLPClusterer2(const GlobalLPClusterer2 &) = delete;
  GlobalLPClusterer2 &operator=(const GlobalLPClusterer2 &) = delete;

  GlobalLPClusterer2(GlobalLPClusterer2 &&) = default;
  GlobalLPClusterer2 &operator=(GlobalLPClusterer2 &&) = default;

  ~GlobalLPClusterer2() override;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  std::unique_ptr<class GlobalLPClusteringImpl2> _impl;
};
} // namespace kaminpar::dist
