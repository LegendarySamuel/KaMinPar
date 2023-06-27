/*******************************************************************************
 * @file:   async_global_lp_clustering.h
 * @author: Samuel Gil
 * @date:   27.06.2023
 * @brief   Label propagation clustering without restrictions, i.e., clusters
 * can span across multiple PEs. (Code from global_lp_clustering.h (Daniel Seemaier) adjusted for this class)
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class AsyncGlobalLPClustering : public ClusteringAlgorithm<GlobalNodeID> {
public:
  explicit AsyncGlobalLPClustering(const Context &ctx);

  AsyncGlobalLPClustering(const AsyncGlobalLPClustering &) = delete;
  AsyncGlobalLPClustering &operator=(const AsyncGlobalLPClustering &) = delete;

  AsyncGlobalLPClustering(AsyncGlobalLPClustering &&) = default;
  AsyncGlobalLPClustering &operator=(AsyncGlobalLPClustering &&) = default;

  ~AsyncGlobalLPClustering() override;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  std::unique_ptr<class AsyncGlobalLPClusteringImpl> _impl;
};
} // namespace kaminpar::dist
