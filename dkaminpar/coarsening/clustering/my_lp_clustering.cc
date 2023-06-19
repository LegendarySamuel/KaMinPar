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
# include <iostream>
# include <memory>
# include <tuple>

namespace kaminpar::dist {

    using ClusterID = GlobalNodeID;
    using cluster_update = std::pair<NodeID, ClusterID>;
    using update_vector = std::vector<cluster_update>;
    using ClusterArray = MyLPClustering::ClusterArray;
    using weights_vector = std::vector<std::tuple<ClusterID, NodeWeight, EdgeWeight>>;

    MyLPClustering::~MyLPClustering() = default;

    /////////////////////////////////////////////////////////////////////////////// helpers

    bool is_overweight(const std::unordered_map<ClusterID, NodeWeight> &cluster_node_weight, const ClusterID c_id, 
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
    ClusterID calculate_new_cluster(NodeID node, const DistributedGraph &graph, const ClusterArray &clusters, 
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
                continue;
            }

            if (temp_edge_weights.find(clusterID) != temp_edge_weights.end()) {    // cluster is already represented in temp_edge_weights
                EdgeWeight temp = temp_edge_weights[clusterID] + eweight;
                temp_edge_weights.insert_or_assign(clusterID, temp);
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
    bool contains(const std::vector<T> &vec, const T item) {
        for (auto&& element : vec) {
            if (item == element) {
                return true;
            }
        }
        return false;
    }

    // used to find the PEs that have to be notified of a label update to u
    std::vector<PEID> ghost_neighbors(NodeID u, const DistributedGraph &graph) {
        std::vector<PEID> ghost_PEs(0);
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
    void fill_send_buffers(NodeID u, std::map<PEID, update_vector> &send_buffers, const ClusterArray &clusters,
                             const DistributedGraph &graph) {
        for (PEID pe : ghost_neighbors(u, graph)) {
            // update a label, if it has been changed before without being sent
            bool contained = false;
            for (auto&& update : send_buffers[pe]) {
                if (update.first == u) {
                    update.second = clusters[u];
                    KASSERT(update.second == clusters[u]);
                    contained = true;
                    break;
                }
            }
            // add update to send_buffer if the node has not been reassigned yet
            if (!contained) {
                send_buffers[pe].push_back(std::make_pair(u, clusters[u]));
            }
        }
    }

    /**
     *  set up the necessary containers for an mpi alltoallv communication
     * setting up the send containers and fields
     */
    void set_up_alltoallv_send(const std::map<PEID, update_vector> &send_buffers, update_vector &send_buffer,
                            int *send_counts, int *send_displ) {
        int displ = 0;
        for (auto& [peid, send] : send_buffers) {
            int count = 0;
            for (cluster_update upd : send) {
                send_buffer.push_back(upd);
                count++;
                displ++;
            }
            KASSERT(send_buffers.at(peid).size() == count);
            send_counts[peid] = count;
            send_displ[peid] = displ - count;
        }
    }

    /**
     * Communicate with the other PEs to tell them how much you'll send 
     * and find out how much to receive.
    */
    void find_recv_counts(int *send_counts, int *recv_counts, const DistributedGraph &graph) {
        MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, graph.communicator());
        MPI_Barrier(graph.communicator());
    }

    /**
     *  set up the necessary containers for an mpi alltoallv communication
     * setting up the receive containers and fields
     */
    void set_up_alltoallv_recv(int *recv_counts, int *recv_displ, update_vector &recv_buffer, 
                                int *send_counts, const DistributedGraph &graph) {
        find_recv_counts(send_counts, recv_counts, graph);
        int total = 0;
        int size = 0;
        MPI_Comm_size(graph.communicator(), &size);
        for (int i = 0; i < size; i++) {
            recv_displ[i] = total;
            total+=recv_counts[i];
        }
        // make place for elements to be received
        recv_buffer.resize(total);
    }

    /**
     * Evaluates and processes the contents of the recv_buffer.
    */
    void evaluate_recv_buffer(update_vector &recv_buffer, int *recv_counts, int *recv_displ,
                                ClusterArray &clusters, int size, int myrank, const DistributedGraph &graph) {
        GlobalNodeID offset_n = 0;
        int index = 0;
        for (int s = 0; s < size; s++) {
            if (myrank == s) {
                continue;
            }
            offset_n = graph.offset_n(s);

            for (int c = 0; c < recv_counts[s]; c++) {
                index = recv_displ[s] + c;
                NodeID local = graph.global_to_local_node(recv_buffer[index].first + offset_n);
                KASSERT(graph.is_ghost_node(local));
                clusters[local] = recv_buffer[index].second;
            }
        }
    }

    /**
     * Cleans up the label communication containers after one iteration.
    */
    void clean_up_iteration(std::map<PEID, update_vector> &send_buffers, update_vector &send_buffer, int* send_counts, 
                                int* send_displ, update_vector &recv_buffer) {
        // send buffers to PEs
        for (auto&& [_, buf] : send_buffers) {
            buf.clear();
            KASSERT(buf.size() == 0);
        }
        send_buffer.clear();
        KASSERT(send_buffer.size() == 0);
        send_counts = {0};
        send_displ = {0};
        // receive buffer
        recv_buffer.clear();
        KASSERT(recv_buffer.size() == 0);
    }

    /**
     * Changes the cluster assignment of the current node.
     * Adjusts the edge and node weights of the clusters.
     * If a cluster is left empty, it is removed.
    */
    void adjust_clusters(const DistributedGraph &graph, NodeID node, ClusterID old_id, ClusterID new_id, 
                        ClusterArray &clusters,
                        std::unordered_map<ClusterID, NodeWeight> &cluster_node_weight, 
                        std::unordered_map<ClusterID, EdgeWeight> &cluster_edge_weight) {
        KASSERT(graph.is_owned_node(node));
        KASSERT(clusters[node] == old_id);
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
        } else {
            cluster_node_weight[old_id]-=node_weight;
            cluster_edge_weight[old_id]-=old_delta;
        }
        if (cluster_node_weight.find(new_id) == cluster_node_weight.end()) {
            cluster_node_weight.insert(std::make_pair(new_id, node_weight));
            cluster_edge_weight.insert(std::make_pair(new_id, new_delta));
        } else {
            NodeWeight temp_nw = cluster_node_weight[new_id];
            cluster_node_weight.erase(new_id);
            cluster_node_weight.insert(std::make_pair(new_id, temp_nw+node_weight));

            EdgeWeight temp_ew = cluster_edge_weight[new_id];
            cluster_edge_weight.erase(new_id);
            cluster_edge_weight.insert(std::make_pair(new_id, temp_ew+new_delta));
        }

        // set new clusterID
        clusters[node] = new_id;
        KASSERT(clusters[node] == new_id);
    }

    /**
     * This is one iteration of the clustering algorithm.
     * New cluster assignments are calculated and the containers for the new label assignments are filled 
     * (namely clusters and send_buffers and cluster_node_weight and cluster_edge_weight).
     */
    void cluster_iteration(const DistributedGraph &graph, ClusterArray &clusters, 
                                std::unordered_map<ClusterID, NodeWeight> &cluster_node_weight, 
                                std::unordered_map<ClusterID, EdgeWeight> &cluster_edge_weight, 
                                std::map<PEID, update_vector> &send_buffers, 
                                GlobalNodeWeight max_cluster_weight) {
        // calculate new cluster for all owned nodes
        for (auto&& node : graph.nodes()) {
            KASSERT(graph.is_owned_node(node));
            ClusterID cl_id = calculate_new_cluster(node, graph, clusters, cluster_node_weight, max_cluster_weight);

            if (cl_id != clusters[node]) {
                adjust_clusters(graph, node, clusters[node], cl_id, clusters, cluster_node_weight, cluster_edge_weight);
                fill_send_buffers(node, send_buffers, clusters, graph);
            }
        }
    }

    std::vector<NodeID> isolated_nodes(const DistributedGraph &graph) {
        std::vector<NodeID> isolated(0);
        for (auto&& node : graph.nodes()) {
            KASSERT(graph.is_owned_node(node));
            if (graph.degree(node) == 0) {
                isolated.push_back(node);
            }
        }
        return isolated;
    }

    /**
     * returns either the PEID of the PE with lowest PEID containing isolated nodes,
     * or -1 if this PE does not contain isolated nodes
    */
    PEID find_lowest_isolated_PEID(const DistributedGraph &graph) {
        // communicate your PEID, if you have isolated nodes; send -1 if you don't have isolated nodes
        PEID message;
        int size = mpi::get_comm_size(graph.communicator());
        std::vector<NodeID> isolated = isolated_nodes(graph);
        if(isolated.size() == 0) {
            message = -1;
        } else {
            message = mpi::get_comm_rank(graph.communicator());
        }
        PEID send[size] = {message};
        PEID recv[size] = {-1};
        MPI_Alltoall(send, size, MPI_INT,recv, size, MPI_INT, graph.communicator());
        mpi::barrier(graph.communicator());
        
        // stop if you sent -1
        if (message == -1) {
            return -1;
        }
        // find lowest PEID
        PEID lowest = message;
        for (int i = 0; i < message; i++) {
            if (recv[i] != -1 && recv[i] < message) {
                lowest = recv[i];
                break;
            }
        }
        KASSERT(0 <= lowest <= size);
        return lowest;
    }

    /**
     * used to cluster the remaining isolated nodes.
     * if a PEs PEID is higher than that of another one, the lower PEs clusterID is used for the isolated nodes.
    */
    void cluster_isolated_nodes(const DistributedGraph &graph, ClusterArray &clusters) {
        PEID lowest = find_lowest_isolated_PEID(graph);
        PEID rank = mpi::get_comm_rank(graph.communicator());
        MPI_Comm isolated_comm;
        uint8_t color = 0;  // color == 0 for all PEs containing isolated nodes
        if (lowest == -1) {
            color = 1;  // group all PEs that don't have isolated nodes
        }
        MPI_Comm_split(graph.communicator(), color, rank, &isolated_comm);

        if (lowest == -1) {
            return;
        }
        std::vector<NodeID> isolated = isolated_nodes(graph);
        ClusterID isolated_cluster;
        if (rank == lowest) {
            isolated_cluster = graph.local_to_global_node(isolated[0]);
        }

        MPI_Bcast(&isolated_cluster, 1, MPI_UINT64_T, 0, isolated_comm);
        MPI_Barrier(isolated_comm);

        for (NodeID node : isolated) {
            clusters[node] = isolated_cluster;
        }
    }

    std::map<PEID, std::vector<NodeID>> calculate_interface_nodes(const DistributedGraph &graph) {
        std::map<PEID, std::vector<NodeID>> interface_nodes;

        for (auto&& node : graph.nodes()) {
            for (auto&& [e, target] : graph.neighbors(node)) {
                if (graph.is_ghost_node(target)) {
                    int peid = graph.ghost_owner(target);
                    interface_nodes.insert(std::make_pair(peid, node));
                }
            }
        }
        return interface_nodes;
    }

    void set_up_weights_comm(weights_vector &send_weights_buffer, weights_vector &recv_weights_buffer, 
                                int *send_weights_counts, int *send_weights_displ,
                                int *recv_weights_counts, int *recv_weights_displ,
                                std::map<PEID, std::vector<NodeID>> &interface_nodes, 
                                const std::unordered_map<ClusterID, NodeWeight> &cluster_node_weight,
                                const std::unordered_map<ClusterID, EdgeWeight> &cluster_edge_weight, 
                                const ClusterArray &clusters, int size,
                                const DistributedGraph &graph) {
        // fill send buffer with the weights of owned clusters assigned to interface nodes
        int displ = 0;
        for (int pe = 0; pe < size; pe++) {
            int count = 0;
            if (interface_nodes.find(pe) == interface_nodes.end()) {
                continue;
            }
            for (NodeID node : interface_nodes.at(pe)) {
                bool contained = false;
                ClusterID cluster = clusters[node];
                for (auto&& [cl_id, a, b] : send_weights_buffer) {
                    if (!graph.is_owned_global_node(cluster)) {
                        break;
                    }
                    if (cl_id == cluster) {
                        contained == true;
                        break;
                    }
                }
                if (!contained) {
                    send_weights_buffer.push_back(std::make_tuple(cluster, cluster_node_weight.at(cluster), cluster_edge_weight.at(cluster)));
                    count++;
                }
            }
            send_weights_counts[pe] = count;
            send_weights_displ[pe] = displ;
            displ+=count;
        }
        
        // set up recv sizes
        MPI_Alltoall(send_weights_counts, 1, MPI_INT, recv_weights_counts, 1, MPI_INT, graph.communicator());
        MPI_Barrier(graph.communicator());

        int total = 0;
        for (int i = 0; i < size; i++) {
            recv_weights_displ[i] = total;
            total+=recv_weights_counts[i];
        }
        // make place for elements to be received
        recv_weights_buffer.resize(total);
    }

    void evaluate_weights(std::unordered_map<ClusterID, NodeWeight> &cluster_node_weight,
                            std::unordered_map<ClusterID, EdgeWeight> &cluster_edge_weight, 
                            const weights_vector &recv_weights_buffer, 
                            int *recv_weights_counts, int size) {
        for (int pe = 0; pe < size; pe++) {
            int index = 0;
            for (int i = 0; i < recv_weights_counts[pe]; i++) {
                std::tuple<ClusterID, NodeWeight, EdgeWeight> tuple = recv_weights_buffer.at(index);
                cluster_node_weight.insert_or_assign(std::get<0>(tuple), std::get<1>(tuple));
                cluster_edge_weight.insert_or_assign(std::get<0>(tuple), std::get<2>(tuple));
                index++;
            }
        }
    }

    void clean_up_weights_comm(weights_vector &send_weights_buffer, int* send_weights_counts, 
                                int* send_weights_displ, weights_vector &recv_weights_buffer) {
        // send buffer
        send_weights_buffer.clear();
        KASSERT(send_weights_buffer.size() == 0);
        send_weights_counts = {0};
        send_weights_displ = {0};
        // receive buffer
        recv_weights_buffer.clear();
        KASSERT(recv_weights_buffer.size() == 0);
    }

    void print_clusters(ClusterArray &clusters, const DistributedGraph &graph) {
        int n = graph.n();
        int g = graph.ghost_n();
        std::cout << "owned nodes (" << n << "): ";
        for (int i = 0; i < n; i++) {
            std::cout << graph.local_to_global_node(i) << ":" << clusters[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "ghost nodes (" << g << "): ";
        for (int i = n; i < g + n; i++) {
            std::cout << graph.local_to_global_node(i) << ":" << clusters[i] << " ";
            
        }
        std::cout << std::endl;
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
        init_clusters(graph.total_n());

        // cluster weights of the clusters
        std::unordered_map<ClusterID, NodeWeight> cluster_node_weight;
        std::unordered_map<ClusterID, EdgeWeight> cluster_edge_weight;

        // MPI rank and size
        int myrank = mpi::get_comm_rank(graph.communicator());
        int size = mpi::get_comm_size(graph.communicator());

        // interface nodes
        std::map<PEID, std::vector<NodeID>> interface_nodes = calculate_interface_nodes(graph);

        // buffers for cluster-weights communication
        weights_vector send_weights_buffer(0);
        weights_vector recv_weights_buffer(0);
        int send_weights_counts[size] = {0};
        int send_weights_displ[size] = {0};
        int recv_weights_counts[size] = {0};
        int recv_weights_displ[size] = {0};
        
        MPI_Datatype weights_update_type = mpi::type::get<weights_vector>();
        
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
        update_vector send_buffer(0);
        int send_counts[size] = {0};
        int send_displ[size] = {0};

        // receive buffer
        update_vector recv_buffer(0);
        int recv_counts[size] = {0};
        int recv_displ[size] = {0};

        // vectors for PEs
        int z = 0;
        for (auto&& [peid, _] : adj_PEs) {
            update_vector temp(0);
            send_buffers.insert(std::make_pair(peid, temp));
            z++;
        }
        KASSERT(send_buffers.size() == adj_PEs.size());
    
        // initialize containers for local clusterIDs and cluster weights
        for (NodeID u : graph.all_nodes()) {
            ClusterID g_id = graph.local_to_global_node(u);
            get_clusters()[u] = g_id;
            cluster_node_weight.insert(std::make_pair(g_id, graph.node_weight(u)));
            cluster_edge_weight.insert(std::make_pair(g_id, 0));
        }
        KASSERT(cluster_node_weight.size() == graph.total_n());
        KASSERT(cluster_edge_weight.size() == graph.total_n());

        // communicate labels ()
        MPI_Datatype update_type = mpi::type::get<cluster_update>();

        int global_iterations = 1;
        for (int i = 0; i < global_iterations; i++) {
            // local cluster iteration
            cluster_iteration(graph, get_clusters(), cluster_node_weight, cluster_edge_weight, send_buffers, max_cluster_weight);

            // communicate labels
            set_up_alltoallv_send(send_buffers, send_buffer, send_counts, send_displ);
            set_up_alltoallv_recv(recv_counts, recv_displ, recv_buffer, send_counts, graph);
     
            MPI_Alltoallv(&send_buffer[0], send_counts, send_displ, update_type, &recv_buffer[0], recv_counts, 
                            recv_displ, update_type, graph.communicator());

            mpi::barrier(graph.communicator());
            // evaluate recv_buffer content
            evaluate_recv_buffer(recv_buffer, recv_counts, recv_displ, get_clusters(), size, myrank, graph);

            // exchange weights for ghost nodes
            // need to send information about interface nodes' cluster weights
            // can naively update weights for ghost nodes, since the sent weights are guaranteed to be the newest data
            /*set_up_weights_comm(send_weights_buffer, recv_weights_buffer, send_weights_counts, send_weights_displ,
                                    recv_weights_counts, recv_weights_displ, interface_nodes, cluster_node_weight, cluster_edge_weight,
                                    get_clusters(), size, graph);
            
            MPI_Alltoallv(&send_weights_buffer[0], send_weights_counts, send_weights_displ, weights_update_type, &recv_weights_buffer[0], 
                            recv_weights_counts, recv_weights_displ, weights_update_type, graph.communicator());

            // evaluate
            evaluate_weights(cluster_node_weight, cluster_edge_weight, recv_weights_buffer, recv_weights_counts, size);

            clean_up_weights_comm(send_weights_buffer, send_weights_counts, send_weights_displ, recv_weights_buffer);*/

            // clean up containers
            clean_up_iteration(send_buffers, send_buffer, send_counts, send_displ, recv_buffer);
        }
        // cluster isolated nodes
        cluster_isolated_nodes(graph, get_clusters());

        //return clusterarray*/
        return get_clusters();
    }
}