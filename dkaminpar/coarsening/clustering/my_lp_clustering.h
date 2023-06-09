# include "dkaminpar/coarsening/clustering/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class MyLPClustering : public ClusteringAlgorithm<GlobalNodeID> {

public:
  using ClusterArray = NoinitVector<GlobalNodeID>;

  ~MyLPClustering() override;

  void initialize(const DistributedGraph &graph);

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight);
};
}