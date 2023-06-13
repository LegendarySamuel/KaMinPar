# include "dkaminpar/coarsening/clustering/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class MyLPClustering : public ClusteringAlgorithm<GlobalNodeID> {
  private:
    ClusterArray _clusters;

  public:
    using ClusterArray = NoinitVector<GlobalNodeID>;
    explicit MyLPClustering(const Context &ctx) {};

    inline ClusterArray &clusters() { return _clusters; }

    ~MyLPClustering() override;

    void initialize(const DistributedGraph &graph);

    ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight);
};
}