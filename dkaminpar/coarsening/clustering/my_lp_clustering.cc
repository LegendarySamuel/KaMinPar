# include "dkaminpar/coarsening/clustering/my_lp_clustering.h"
# include "dkaminpar/datastructures/distributed_graph.h"
# include "dkaminpar/context.h"
// # include "dkaminpar/mpi/grid_alltoall.h" vlt Alternative
# include <cmath>
# include <ctime>
# include "oneapi/tbb.h"
# include <utility>
# include <map>
# include <unordered_map>
# include "mpi.h"

namespace kaminpar::dist {
    using ClusterID = GlobalNodeID;
    using clusterNodeWeight = std::pair<ClusterID, NodeWeight>;
    using clusterEdgeWeight = std::pair<ClusterID, EdgeWeight>;
    using cluster_update = std::pair<NodeID, GlobalNodeID>;
    using update_vector = std::vector<cluster_update>;

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

    bool is_overweight(std::vector<std::pair<ClusterID, NodeWeight>> clusterWeight, const ClusterID c_id, 
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

    /** calculates the best cluster to put a node into; does not do anything related to global communication */
    GlobalNodeID calculate_new_cluster(NodeID node, const DistributedGraph &graph, MyLPClustering::ClusterArray clusters, 
                                        std::vector<clusterNodeWeight> clusterWeight, double max_weight) {
        /* find adjacent nodes
         * calculate cluster with maximum intra cluster edge weight
         * check weight of "new" edges and sum them up if in the same cluster, 
         * then choose cluster with the highest gain in weight
         * make sure max cluster weight constraint is not violated
         */
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

    // find out whether an item is contained within a vector, needs "==" operator
    template<typename T>
    bool contains(std::vector<T> vec, T item) {
        for (auto&& element : vec) {
            if (item == element) {
                return true;
            }
        }
        return false;
    }

    // used to find the PEs that have to be notified of a label update to u
    std::vector<PEID> ghost_neighbors(NodeID u, const DistributedGraph &graph) {
        std::vector<PEID> ghost_PEs = std::vector<PEID>();
        for (auto&& [e, target] : graph.neighbors(u)) {
            if (graph.is_ghost_node(target)) {
                int peid = graph.ghost_owner(target);
                if (!contains(ghost_PEs, peid)) {
                    ghost_PEs.push_back(peid);
                }
            }
        }
        return ghost_PEs;
    }

    /**
     *  fills the appropriate send_buffers for Node u
     * this method is called after an update has been made to the cluster assignment of Node u
     */
    void fill_send_buffers(NodeID u, MyLPClustering::ClusterArray clusters, std::map<PEID, update_vector> send_buffers, const DistributedGraph &graph) {
        for (PEID pe : ghost_neighbors(u, graph)) {
            send_buffers.at(pe).push_back(std::make_pair(u, clusters[u]));
        }
    }

    /**
     *  set up the necessary containers for an mpi alltoallv communication
     * setting up the send containers and fields
     */
    void set_up_alltoallv_send(const std::map<PEID, update_vector> send_buffers, update_vector &send_buffer,
                            int *send_counts, int *send_displ) {
        int displ = 0;
        for (auto&& [peid, send] : send_buffers) {
            int count = 0;
            for (cluster_update upd : send) {
                send_buffer.push_back(upd);
                count++;
                displ++;
            }
            send_counts[peid] = count;
            send_displ[peid] = displ - count;
        }
    }

    /**
     *  set up the necessary containers for an mpi alltoallv communication
     * setting up the receive containers and fields
     */
    void set_up_alltoallv_recv(int *recv_counts, int *recv_displ, const DistributedGraph &graph) {
        std::map<PEID, int> counts;
        for (NodeID&& g : graph.ghost_nodes()) {
            PEID id = graph.ghost_owner(g);
            if (counts.find(id) != counts.end()) {
                counts[id]++;
            } else {
                counts.insert(std::make_pair(id, 1));
            }
        }
        int displ = 0;
        for (auto&& [peid, count] : counts) {
            recv_counts[peid] = count;
            displ+=count;
            recv_displ[peid] = displ - count;
        }
    }

    /**
     * Cleans up the label communication containers after one iteration.
    */
    void clean_up_iteration(std::map<PEID, update_vector> &send_buffers, update_vector &send_buffer, int* send_counts, int* send_displ, 
                                update_vector &recv_buffer, int* recv_counts, int* recv_displ) {
        // send buffers to PEs
        send_buffers.clear();
        send_buffer.clear();
        send_counts = {NULL};
        send_displ = {0};

        // receive buffer
        recv_buffer.clear();
        recv_counts = {NULL};
        recv_displ = {0};
    }

    /**
     * This is one iteration of the clustering algorithm.
     * New cluster assignments are calculated and the containers required for the label communication are filled.
     */
    MyLPClustering::ClusterArray cluster_iteration() {
        // TODO one iteration
    }

    ///////////////////////////////////////////////////////////////////////////////

    void MyLPClustering::initialize(const DistributedGraph &graph) {

    }

    // subgraphs are already given by initial partitioning
    /**
     * have to calculate clusters:
     * 1.) initialize clusters with single nodes
     * 2.) for each node maximize intra cluster edge weight by moving node to adjacent cluster without exceeding max cluster weight
     * (have to check for all nodes at the edges of clusters, not only for interface nodes)
     * if cluster is the same, do not communicate
     * 3.) put all isolated nodes in one cluster
     */
    MyLPClustering::ClusterArray &MyLPClustering::cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) {
        // TODO

        // clusterIDs of the vertices
        MyLPClustering::ClusterArray clusters(graph.total_n());

        // cluster weights of the clusters
        std::vector<clusterNodeWeight> clusterWeight(graph.total_n());

        // MPI rank and size
        int myrank = mpi::get_comm_rank(graph.communicator());
        int size = mpi::get_comm_size(graph.communicator());

        // adjacent PEs (put in hashmap to ensure uniqueness of PEs)
        std::unordered_map<PEID, PEID> adj_PEs;

        // find all adjacent PEs
        for (NodeID u : graph.all_nodes()) {
            if (graph.is_ghost_node(u)) {
                int pe = graph.ghost_owner(u);
                adj_PEs.insert(std::make_pair(pe,pe));
            }
        }

        // send buffers to PEs
        std::map<PEID, update_vector> send_buffers;
        update_vector *send_buffer = new update_vector();
        int send_counts[size] = {NULL};
        int send_displ[size] = {0};

        // receive buffer
        update_vector *recv_buffer = new update_vector();
        int recv_counts[size] = {NULL};
        int recv_displ[size] = {0};

        // vectors for PEs
        for (auto&& [peid, _] : adj_PEs) {
            update_vector *temp = new update_vector();
            send_buffers.insert(std::make_pair(peid, *temp));
        }
        
        // initialize containers for local clusterIDs and cluster weights
        for (NodeID u : graph.all_nodes()) {
            GlobalNodeID g_id = graph.local_to_global_node(u);
            clusters[u] = g_id;
            clusterWeight[u] = std::make_pair(g_id, graph.node_weight(u));
        }

        // fill send buffers initally
        for (NodeID u : graph.nodes()) {
            fill_send_buffers(u, clusters, send_buffers, graph);
        }

        // communicate labels ()
        
        MPI_Datatype update_type = mpi::type::get<cluster_update>();

        set_up_alltoallv_send(send_buffers, *send_buffer, send_counts, send_displ);
        set_up_alltoallv_recv(recv_counts, recv_displ, graph);
        MPI_Alltoallv(send_buffer, send_counts, send_displ, update_type, recv_buffer, recv_counts, recv_displ, update_type, graph.communicator());

        mpi::barrier(graph.communicator());
        // TODO evaluate recv_buffer content
        
        clean_up_iteration(send_buffers, *send_buffer, send_counts, send_displ, *recv_buffer, recv_counts, recv_displ);

        // TODO calculate new cluster assignments, do not communicate if node stays in the same cluster
        int iterations = 10;
        for (int i = 0; i < iterations; i++) {
            // TODO add cluster_iteration() arguments
            // cluster_iteration();
            // communicate labels
            set_up_alltoallv_send(send_buffers, *send_buffer, send_counts, send_displ);
            set_up_alltoallv_recv(recv_counts, recv_displ, graph);
            MPI_Alltoallv(send_buffer, send_counts, send_displ, update_type, recv_buffer, recv_counts, recv_displ, update_type, graph.communicator());

            mpi::barrier(graph.communicator());
            // TODO evaluate recv_buffer content
            clean_up_iteration(send_buffers, *send_buffer, send_counts, send_displ, *recv_buffer, recv_counts, recv_displ);
        }
    }
}