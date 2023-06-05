# include "dkaminpar/coarsening/clustering/my_lp_clustering.h"
# include "dkaminpar/datastructures/distributed_graph.h"
# include "dkaminpar/context.h"
# include <cmath>
# include <ctime>
# include "oneapi/tbb.h"
# include <utility>
# include <unordered_map>

namespace kaminpar::dist {
    MyLPClustering::~MyLPClustering() = default;

    /////////////////////////////////////////////////////////////////////////////// helpers
    double number_of_clusters(double blocks = 4, double contractionLimit = 1024, const DistributedGraph &graph) {
        return std::min(blocks, graph.total_n()/contractionLimit);
    }

    double maximum_cluster_weight(double blocks = 4, double contractionLimit = 1024, const DistributedGraph &graph) {
        std::srand(std::time(NULL));
        int rand = std::rand() % 100000;
        int eps = rand / 100000000000.0;
        return eps * (graph.global_total_node_weight() / number_of_clusters(blocks, contractionLimit, graph));
    }

    GlobalNodeID calculate_new_cluster(NodeID node, const DistributedGraph &graph, MyLPClustering::ClusterArray clusters) {
        // find adjacent nodes
        // NodeID degree = graph.degree(node);
        typedef std::pair<GlobalNodeID, EdgeWeight> clusterWeight;
        std::vector<clusterWeight> sums(0);

        for (auto&& edgeID : graph.incident_edges(node)) {
            NodeID target = graph.edge_target(edgeID);
            EdgeWeight eweight = graph.edge_weight(edgeID);            

            if (graph.is_ghost_node(target)) {
                graph.local_to_global_node(target);
                graph.ghost_owner(target);
                // TODO get clusterID from ghost owner
            }

            std::unordered_map<GlobalNodeID, EdgeWeight> sums;
            if (sums.find(clusters[target]) != sums.end()) {    // cluster is represented in sums
                EdgeWeight temp = sums[clusters[target]] + eweight;
                sums.erase(clusters[target]);
                sums.insert({clusters[target], temp});     // different Syntax than that of the else block, just cause
            } else {    // cluster is not represented in sums
                sums.insert(std::make_pair(clusters[target], eweight));
            }
        }
        // calculate cluster with maximum intra cluster edge weight
        // check weight of new edges and sum them up if in the same cluster, then choose cluster with the highest gain in weight

        // make sure max cluster weight constraint is not violated
        // if best cluster is too heavy, choose next lighter one
    }

    ///////////////////////////////////////////////////////////////////////////////

    void MyLPClustering::initialize(const DistributedGraph &graph) {

        MyLPClustering::ClusterArray clusters(graph.n());
        typedef MyLPClustering::ClusterArray::iterator iter;
        
        GlobalNodeID node = graph.offset_n();
        for (iter it = clusters.begin(); it != clusters.end(); it++) {
            clusters.push_back(node);
            node++;
        }

        mpi::barrier(graph.communicator());
        
        /*int rank = mpi::get_comm_rank(graph.communicator());
        int nOP = mpi::get_comm_size(graph.communicator());
        int begin = rank*nOP;
        int end = rank*nOP + (graph.n()/nOP);
        */

        // communicate labels
    }

    // subgraphs are already given by initial partitioning
    /* have to calculate clusters:
     * 1.) initialize clusters with single nodes
     * 2.) for each node maximize intra cluster edge weight by moving node to adjacent cluster without exceeding max cluster weight
     * (have to check for all nodes at the edges of clusters, not only for interface nodes)
     * 3.) put all isolated nodes in one cluster
     */
    MyLPClustering::ClusterArray &MyLPClustering::cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) {
        // TODO
    }
}