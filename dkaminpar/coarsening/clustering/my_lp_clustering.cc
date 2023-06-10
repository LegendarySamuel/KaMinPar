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
    MyLPClustering::MyLPClustering(const Context &ctx) {}

    using ClusterID = GlobalNodeID;
    using cluster_update = std::pair<NodeID, ClusterID>;
    using update_vector = std::vector<cluster_update>;

    MyLPClustering::ClusterArray clusters;

    MyLPClustering::~MyLPClustering() = default;

    /////////////////////////////////////////////////////////////////////////////// helpers

    bool is_overweight(const std::unordered_map<ClusterID, NodeWeight> cluster_node_weight, const ClusterID c_id, 
                        const NodeID n_id, const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) {
        for (auto&& [cluster, weight] : cluster_node_weight) {
            if (cluster == c_id) {
                if (weight + graph.node_weight(n_id) > max_cluster_weight) {
                    return true;
                } else {
                    break;
                }
            }
        }
        return false;
    }

    /** calculates the best cluster to put a node into; does not do anything related to global communication */
    ClusterID calculate_new_cluster(NodeID node, const DistributedGraph &graph, const MyLPClustering::ClusterArray &clusters, 
                                        std::unordered_map<ClusterID, NodeWeight> &cluster_node_weight, GlobalNodeWeight max_cluster_weight) {
        /* find adjacent nodes
         * calculate cluster with maximum intra cluster edge weight
         * check weight of "new" edges and sum them up if in the same cluster, 
         * then choose cluster with the highest gain in weight
         * make sure max cluster weight constraint is not violated
         */

        std::unordered_map<ClusterID, EdgeWeight> temp_edge_weights;

        for (auto&& edgeID : graph.incident_edges(node)) {
            NodeID target = graph.edge_target(edgeID);
            EdgeWeight eweight = graph.edge_weight(edgeID);            
            ClusterID clusterID = clusters[target];

            // skip this cluster if it would be overweight
            if (is_overweight(cluster_node_weight, clusterID, target, graph, max_cluster_weight)) {
                break;
            }

            if (temp_edge_weights.find(clusterID) != temp_edge_weights.end()) {    // cluster is already represented in temp_edge_weights
                EdgeWeight temp = temp_edge_weights[clusterID] + eweight;
                temp_edge_weights.erase(clusterID);
                temp_edge_weights.insert(std::make_pair(clusterID, temp));
            } else {    // cluster is not yet represented in temp_edge_weights
                temp_edge_weights.insert(std::make_pair(clusterID, eweight));
            }
        }

        // check for maxEdgeWeigth cluster
        std::pair<ClusterID, EdgeWeight> max = std::make_pair(0, 0);
        for (auto&& pair : temp_edge_weights) {
            if (pair.second > max.second) {
                max = pair;
            }
        }

        // return new clusterID
        return max.first;
    }

    // find out whether an item is contained within a vector, needs "==" operator
    template<typename T>
    bool contains(const std::vector<T> vec, const T item) {
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
    void fill_send_buffers(NodeID u, const MyLPClustering::ClusterArray clusters, std::map<PEID, update_vector> &send_buffers, 
                            const DistributedGraph &graph) {
        for (PEID pe : ghost_neighbors(u, graph)) {
            // update a label, if it has been changed before without being sent
            bool contained = false;
            for (auto& update : send_buffers[pe]) {
                if (update.first == u) {
                    update.second = clusters[u];
                    contained = true;
                    break;
                }
            }
            // add update to send_buffer if the node has not been reassigned yet
            if (!contained) {
                send_buffers.at(pe).push_back(std::make_pair(u, clusters[u]));
            }
        }
    }

    /**
     *  set up the necessary containers for an mpi alltoallv communication
     * setting up the send containers and fields
     */
    void set_up_alltoallv_send(const std::map<PEID, update_vector> send_buffers, update_vector &send_buffer,
                            int *send_counts, int *send_displ) {
        int displ = 0;
        for (auto& [peid, send] : send_buffers) {
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
     * Evaluates and processes the contents of the recv_buffer.
    */
    void evaluate_recv_buffer(update_vector &recv_buffer, MyLPClustering::ClusterArray &clusters) {
        for (auto&& [nodeID, clusterID] : recv_buffer) {
            clusters[nodeID] = clusterID;
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
        send_counts = {0};
        send_displ = {0};

        // receive buffer
        recv_buffer.clear();
        recv_counts = {0};
        recv_displ = {0};
    }

    /**
     * Changes the cluster assignment of the current node.
     * Adjusts the edge and node weights of the clusters.
     * If a cluster is left empty, it is removed.
    */
    void adjust_clusters(const DistributedGraph &graph, NodeID node, ClusterID old_id, ClusterID new_id, 
                        MyLPClustering::ClusterArray &clusters,
                        std::unordered_map<ClusterID, NodeWeight> &cluster_node_weight, 
                        std::unordered_map<ClusterID, EdgeWeight> &cluster_edge_weight) {
        // temporary weights to calculate the weight differences
        EdgeWeight old_delta = 0;
        EdgeWeight new_delta = 0;

        // calculate weight differences
        for (auto&& [e_id, target] : graph.neighbors(node)) {
            if (clusters[target] == old_id) {
                old_delta+=graph.edge_weight(e_id);
            } else if (clusters[target] == new_id) {
                new_delta+=graph.edge_weight(e_id);
            }
        }

        // adjust weights
        NodeWeight node_weight = graph.node_weight(node);
        if (cluster_node_weight[old_id]-node_weight == 0) { // remove weights for empty cluster
            cluster_node_weight.erase(old_id);
            cluster_edge_weight.erase(old_id);
            cluster_node_weight[new_id]+=node_weight;
            cluster_edge_weight[new_id]+=new_delta;
        } else {
            cluster_node_weight[old_id]-=node_weight;
            cluster_edge_weight[old_id]-=old_delta; 
            cluster_node_weight[new_id]+=node_weight;
            cluster_edge_weight[new_id]+=new_delta;
        }


        // set new clusterID
        clusters[node] = new_id;
    }

    /**
     * This is one iteration of the clustering algorithm.
     * New cluster assignments are calculated and the containers for the new label assignments are filled 
     * (namely clusters and send_buffers and cluster_node_weight and cluster_edge_weight).
     */
    void cluster_iteration(const DistributedGraph &graph, MyLPClustering::ClusterArray &clusters, 
                                std::unordered_map<ClusterID, NodeWeight> &cluster_node_weight, 
                                std::unordered_map<ClusterID, EdgeWeight> &cluster_edge_weight, 
                                std::map<PEID, update_vector> &send_buffers, 
                                GlobalNodeWeight max_cluster_weight) {
        // calculate_new_cluster for all owned nodes
        for (auto&& node : graph.nodes()) {
            ClusterID cl_id = calculate_new_cluster(node, graph, clusters, cluster_node_weight, max_cluster_weight);
            if (cl_id != clusters[node]) {
                adjust_clusters(graph, node, clusters[node], cl_id, clusters, cluster_node_weight, cluster_edge_weight);
                fill_send_buffers(node, clusters, send_buffers, graph);
            }
        }
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
        // clusterIDs of the vertices
        clusters = *(new MyLPClustering::ClusterArray(graph.total_n()));

        // cluster weights of the clusters
        std::unordered_map<ClusterID, NodeWeight> cluster_node_weight;
        std::unordered_map<ClusterID, EdgeWeight> cluster_edge_weight;

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
        int send_counts[size] = {0};
        int send_displ[size] = {0};

        // receive buffer
        update_vector *recv_buffer = new update_vector();
        int recv_counts[size] = {0};
        int recv_displ[size] = {0};

        // vectors for PEs
        for (auto&& [peid, _] : adj_PEs) {
            update_vector *temp = new update_vector();
            send_buffers.insert(std::make_pair(peid, *temp));
        }
        
        // initialize containers for local clusterIDs and cluster weights
        for (NodeID u : graph.all_nodes()) {
            ClusterID g_id = graph.local_to_global_node(u);
            clusters[u] = g_id;
            cluster_node_weight.insert(std::make_pair(g_id, graph.node_weight(u)));
            cluster_edge_weight.insert(std::make_pair(g_id, 0));
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
        // evaluate recv_buffer content
        evaluate_recv_buffer(*recv_buffer, clusters);

        // clean up containers
        clean_up_iteration(send_buffers, *send_buffer, send_counts, send_displ, *recv_buffer, recv_counts, recv_displ);

        // global cluster iterations
        int global_iterations = 5;
        int local_iterations = 3;
        for (int i = 0; i < global_iterations; i++) {
            // local cluster iterations
            for (int y = 0; y < local_iterations; y++) {
                cluster_iteration(graph, clusters, cluster_node_weight, cluster_edge_weight, send_buffers, max_cluster_weight);
            }

            // communicate labels
            set_up_alltoallv_send(send_buffers, *send_buffer, send_counts, send_displ);
            set_up_alltoallv_recv(recv_counts, recv_displ, graph);
            MPI_Alltoallv(send_buffer, send_counts, send_displ, update_type, recv_buffer, recv_counts, recv_displ, update_type, graph.communicator());

            mpi::barrier(graph.communicator());
            // evaluate recv_buffer content
            evaluate_recv_buffer(*recv_buffer, clusters);

            // clean up containers
            clean_up_iteration(send_buffers, *send_buffer, send_counts, send_displ, *recv_buffer, recv_counts, recv_displ);
        }

        return clusters;
    }
}