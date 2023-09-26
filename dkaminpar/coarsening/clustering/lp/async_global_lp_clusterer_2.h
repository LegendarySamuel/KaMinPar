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
class AsyncGlobalLPClusterer2 : public Clusterer<GlobalNodeID> {
public:
  explicit AsyncGlobalLPClusterer2(const Context &ctx);

  AsyncGlobalLPClusterer2(const AsyncGlobalLPClusterer2 &) = delete;
  AsyncGlobalLPClusterer2 &operator=(const AsyncGlobalLPClusterer2 &) = delete;

  AsyncGlobalLPClusterer2(AsyncGlobalLPClusterer2 &&) = default;
  AsyncGlobalLPClusterer2 &operator=(AsyncGlobalLPClusterer2 &&) = default;

  ~AsyncGlobalLPClusterer2() override;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  std::unique_ptr<class AsyncGlobalLPClusteringImpl2> _impl;
friend AsyncGlobalLPClusteringImpl2;
};
} // namespace kaminpar::dist
