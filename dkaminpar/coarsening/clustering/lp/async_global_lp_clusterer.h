/*******************************************************************************
 * @file:   async_global_lp_clusterer.h
 * @author: Samuel Gil
 * @date:   27.06.2023
 * @brief   Label propagation clustering without restrictions, i.e., clusters
 * can span across multiple PEs. (Code from global_lp_clustering.h (Daniel Seemaier) adjusted for this class)
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering/clusterer.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class AsyncGlobalLPClusterer : public Clusterer<GlobalNodeID> {
public:
  explicit AsyncGlobalLPClusterer(const Context &ctx);

  AsyncGlobalLPClusterer(const AsyncGlobalLPClusterer &) = delete;
  AsyncGlobalLPClusterer &operator=(const AsyncGlobalLPClusterer &) = delete;

  AsyncGlobalLPClusterer(AsyncGlobalLPClusterer &&) = default;
  AsyncGlobalLPClusterer &operator=(AsyncGlobalLPClusterer &&) = default;

  ~AsyncGlobalLPClusterer() override;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  std::unique_ptr<class AsyncGlobalLPClusteringImpl> _impl;
friend AsyncGlobalLPClusteringImpl;
};
} // namespace kaminpar::dist
