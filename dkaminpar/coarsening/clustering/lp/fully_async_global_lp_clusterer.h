/*******************************************************************************
 * @file:   fully_async_global_lp_clusterer.h
 * @author: Samuel Gil
 * @date:   20.09.2023
 * @brief   Label propagation clustering without restrictions, i.e., clusters
 * can span across multiple PEs. (Code from global_lp_clustering.h (Daniel Seemaier) adjusted for this class)
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering/clusterer.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class FullyAsyncGlobalLPClusterer : public Clusterer<GlobalNodeID> {
public:
  explicit FullyAsyncGlobalLPClusterer(const Context &ctx);

  FullyAsyncGlobalLPClusterer(const FullyAsyncGlobalLPClusterer &) = delete;
  FullyAsyncGlobalLPClusterer &operator=(const FullyAsyncGlobalLPClusterer &) = delete;

  FullyAsyncGlobalLPClusterer(FullyAsyncGlobalLPClusterer &&) = default;
  FullyAsyncGlobalLPClusterer &operator=(FullyAsyncGlobalLPClusterer &&) = default;

  ~FullyAsyncGlobalLPClusterer() override;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  std::unique_ptr<class FullyAsyncGlobalLPClusteringImpl> _impl;
friend FullyAsyncGlobalLPClusteringImpl;
};
} // namespace kaminpar::dist
