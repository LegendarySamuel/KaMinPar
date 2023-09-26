/*******************************************************************************
 * @file:   mq_async_global_lp_clusterer.h
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
class MQAsyncGlobalLPClusterer : public Clusterer<GlobalNodeID> {
public:
  explicit MQAsyncGlobalLPClusterer(const Context &ctx);

  MQAsyncGlobalLPClusterer(const MQAsyncGlobalLPClusterer &) = delete;
  MQAsyncGlobalLPClusterer &operator=(const MQAsyncGlobalLPClusterer &) = delete;

  MQAsyncGlobalLPClusterer(MQAsyncGlobalLPClusterer &&) = default;
  MQAsyncGlobalLPClusterer &operator=(MQAsyncGlobalLPClusterer &&) = default;

  ~MQAsyncGlobalLPClusterer() override;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  std::unique_ptr<class MQAsyncGlobalLPClusteringImpl> _impl;
friend MQAsyncGlobalLPClusteringImpl;
};
} // namespace kaminpar::dist
