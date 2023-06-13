# include "dkaminpar/coarsening/clustering/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class MyLPClustering : public ClusteringAlgorithm<GlobalNodeID> {
  public:
    using ClusterArray = NoinitVector<GlobalNodeID>;
    explicit MyLPClustering(const Context &ctx) : _clusters(NoinitVector<GlobalNodeID>()) {};

    inline void resize( NodeID total_nodes ) {
      _clusters.resize(total_nodes);
    }

    inline ClusterArray &clusters() { return _clusters; }

    ~MyLPClustering() override;

    void initialize(const DistributedGraph &graph);

    ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight);

  private:
    ClusterArray _clusters;

};
}