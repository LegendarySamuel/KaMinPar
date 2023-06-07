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

    bool is_overweight(std::vector<std::pair<GlobalNodeID, NodeWeight>> clusterWeight, const GlobalNodeID c_id, 
                        const NodeID n_id, const DistributedGraph &graph, double max_weight) {
        for (auto&& [cluster, weight] : clusterWeight) {
            if (cluster == c_id) {
                if (weight + graph.node_weight(n_id) > max_weight) {
                    return true;
                } else {
                    break;
                }
            }
        }
        return false;
    }

    /* calculates the best cluster to put a node into; does not do anything related to global communication */
    GlobalNodeID calculate_new_cluster(NodeID node, const DistributedGraph &graph, MyLPClustering::ClusterArray clusters, 
                                        std::vector<std::pair<GlobalNodeID, NodeWeight>> clusterWeight, double max_weight) {
        /* find adjacent nodes
         * calculate cluster with maximum intra cluster edge weight
         * check weight of "new" edges and sum them up if in the same cluster, 
         * then choose cluster with the highest gain in weight
         * make sure max cluster weight constraint is not violated
         */
        using ClusterID = GlobalNodeID;
        typedef std::pair<ClusterID, EdgeWeight> clusterEdgeWeight;
        std::vector<clusterEdgeWeight> sums(0);

        for (auto&& edgeID : graph.incident_edges(node)) {
            NodeID target = graph.edge_target(edgeID);
            EdgeWeight eweight = graph.edge_weight(edgeID);            
            ClusterID clusterID = clusters[target];

            // skip this cluster if it would be overweight
            if (is_overweight(clusterWeight, clusterID, target, graph, max_weight)) {
                break;
            }

            std::unordered_map<ClusterID, EdgeWeight> sums;
            if (sums.find(clusterID) != sums.end()) {    // cluster is already represented in sums
                EdgeWeight temp = sums[clusterID] + eweight;
                sums.erase(clusterID);
                sums.insert(std::make_pair(clusterID, temp));
            } else {    // cluster is not yet represented in sums
                sums.insert(std::make_pair(clusterID, eweight));
            }
        }

        // check for maxEdgeWeigth cluster
        std::pair<ClusterID, EdgeWeight> max = std::make_pair(0, 0);
        for (auto&& pair : sums) {
            if (pair.second > max.second) {
                max = pair;
            }
        }

        // return new clusterID
        return max.first;
    }

    ///////////////////////////////////////////////////////////////////////////////

    void MyLPClustering::initialize(const DistributedGraph &graph) {

    }

    // subgraphs are already given by initial partitioning
    /* have to calculate clusters:
     * 1.) initialize clusters with single nodes
     * 2.) for each node maximize intra cluster edge weight by moving node to adjacent cluster without exceeding max cluster weight
     * (have to check for all nodes at the edges of clusters, not only for interface nodes)
     * if cluster is the same, do not communicate
     * 3.) put all isolated nodes in one cluster
     */
    MyLPClustering::ClusterArray &MyLPClustering::cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) {
        // TODO
        using ClusterID = GlobalNodeID;

        // clusterIDs of the vertices
        MyLPClustering::ClusterArray clusters(graph.total_n());

        // cluster weights of the clusters
        typedef std::pair<ClusterID, NodeWeight> clusterNodeWeight;
        std::vector<clusterNodeWeight> clusterWeight(graph.total_n());

        // TODO vectors for PEs (depending on mpi::size allocate space)
        
        // initialize containers and 
        // TODO set up the send buffers if necessary
        for (NodeID u : graph.all_nodes()) {
            GlobalNodeID g_id = graph.local_to_global_node(u);
            clusters[u] = g_id;
            clusterWeight[u] = std::make_pair(g_id, graph.node_weight(u));
        }

        mpi::barrier(graph.communicator());
        
        /*int rank = mpi::get_comm_rank(graph.communicator());
        int nOP = mpi::get_comm_size(graph.communicator());
        int begin = rank*nOP;
        int end = rank*nOP + (graph.n()/nOP);
        */

        // TODO communicate labels ()


        // TODO calculate new cluster assignments, do not communicate if node stays in the same cluster
    }
}