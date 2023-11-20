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
class MQGlobalLPClusterer : public Clusterer<GlobalNodeID> {
public:
  explicit MQGlobalLPClusterer(const Context &ctx);

  MQGlobalLPClusterer(const MQGlobalLPClusterer &) = delete;
  MQGlobalLPClusterer &operator=(const MQGlobalLPClusterer &) = delete;

  MQGlobalLPClusterer(MQGlobalLPClusterer &&) = default;
  MQGlobalLPClusterer &operator=(MQGlobalLPClusterer &&) = default;

  ~MQGlobalLPClusterer() override;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  std::unique_ptr<class MQGlobalLPClusteringImpl> _impl;
};
} // namespace kaminpar::dist
