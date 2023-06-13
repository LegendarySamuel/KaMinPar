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

namespace kaminpar::dist {

    using ClusterID = GlobalNodeID;
    using cluster_update = std::pair<NodeID, ClusterID>;
    using update_vector = std::vector<cluster_update>;
    using ClusterArray = MyLPClustering::ClusterArray;

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
    std::vector<PEID> *ghost_neighbors(NodeID u, const DistributedGraph &graph) {
        std::vector<PEID> *ghost_PEs = new std::vector<PEID>;
        for (auto&& [e, target] : graph.neighbors(u)) {
            if (graph.is_ghost_node(target)) {
                int peid = graph.ghost_owner(target);
                if (!contains(*ghost_PEs, peid)) {
                    (*ghost_PEs).push_back(peid);
                }
            }
        }
        return ghost_PEs;
    }

    /**
     *  fills the appropriate send_buffers for Node u
     * this method is called after an update has been made to the cluster assignment of Node u
     */
    void fill_send_buffers(NodeID u, const ClusterArray &clusters, std::map<PEID, update_vector> &send_buffers, 
                            const DistributedGraph &graph) {
        for (PEID pe : (*ghost_neighbors(u, graph))) {
            // update a label, if it has been changed before without being sent
            bool contained = false;
            for (cluster_update update : send_buffers[pe]) {
                if (update.first == u) {
                    update.second = clusters[u];
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
    void set_up_alltoallv_send(const std::map<PEID, update_vector> send_buffers, update_vector *send_buffer,
                            int *send_counts, int *send_displ) {
        int displ = 0;
        for (auto& [peid, send] : send_buffers) {
            int count = 0;
            for (cluster_update upd : send) {
                (*send_buffer).push_back(upd);
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
    void set_up_alltoallv_recv(int *recv_counts, int *recv_displ, update_vector *recv_buffer, const DistributedGraph &graph) {
        std::map<PEID, int> counts;
        for (NodeID&& g : graph.ghost_nodes()) {
            PEID id = graph.ghost_owner(g);
            if (counts.find(id) != counts.end()) {
                counts[id]++;
            } else {
                counts.insert(std::make_pair(id, 1));
            }
        }
        int total = 0;
        for (auto&& [peid, count] : counts) {
            total+=count;
            recv_counts[peid] = count;
            recv_displ[peid] = total - count;
        }
        // make place for elements to be received
        (*recv_buffer).resize(total);
    }

    /**
     * Evaluates and processes the contents of the recv_buffer.
    */
    void evaluate_recv_buffer(update_vector *recv_buffer, int *recv_counts, 
                                ClusterArray &clusters, int size, 
                                int myrank, const DistributedGraph &graph) {
        GlobalNodeID offset_n = 0;
        int index = 0;
        for (int s = 0; s < size; s++) {
            if (myrank == s) {
                break;
            }
            offset_n = graph.offset_n(s);
            for (int c = 0; c < recv_counts[s]; c++) {
                NodeID local = graph.global_to_local_node((*recv_buffer)[index].first + offset_n);
                clusters[local] = (*recv_buffer)[index].second;
                index++;
            }
        }
    }

    /**
     * Cleans up the label communication containers after one iteration.
    */
    void clean_up_iteration(std::map<PEID, update_vector> &send_buffers, update_vector *send_buffer, int* send_counts, int* send_displ, 
                                update_vector *recv_buffer) {
        // send buffers to PEs
std::cout << "here-1" << std::endl;
        for (auto&& [_, buf] : send_buffers) {
std::cout << "here-0.5" << std::endl;
            buf.clear();
            buf.resize(0);
        }
std::cout << "here0" << std::endl;
        (*send_buffer).clear();
        send_counts = {0};
        send_displ = {0};
std::cout << "here1" << std::endl;
        // receive buffer
        (*recv_buffer).clear();
std::cout << "here2" << std::endl;
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
        // temporary weights to calculate the weight differences
        EdgeWeight old_delta = 0;
        EdgeWeight new_delta = 0;
std::cout << "loop1" << std::endl;
        // calculate weight differences
        for (auto&& [e_id, target] : graph.neighbors(node)) {
            if (clusters[target] == old_id) {
                old_delta+=graph.edge_weight(e_id);
            } else if (clusters[target] == new_id) {
                new_delta+=graph.edge_weight(e_id);
            }
        }
std::cout << "sec2" << std::endl;
        // adjust weights
        NodeWeight node_weight = graph.node_weight(node);
        if (cluster_node_weight[old_id]-node_weight == 0) { // remove weights for empty cluster
            cluster_node_weight.erase(old_id);
            cluster_edge_weight.erase(old_id);
std::cout << "3" << std::endl;
        } else {
            cluster_node_weight[old_id]-=node_weight;
            cluster_edge_weight[old_id]-=old_delta;
std::cout << "8" << std::endl;
        }
        if (cluster_node_weight.find(new_id) == cluster_node_weight.end()) {
std::cout << "if" << std::endl;
            cluster_node_weight.insert(std::make_pair(new_id, node_weight));
            cluster_edge_weight.insert(std::make_pair(new_id, new_delta));
        } else {
std::cout << "else" << std::endl;
            NodeWeight temp_nw = cluster_node_weight[new_id];
            cluster_node_weight.erase(new_id);
            cluster_node_weight.insert(std::make_pair(new_id, temp_nw+node_weight));

            EdgeWeight temp_ew = cluster_edge_weight[new_id];
            cluster_edge_weight.erase(new_id);
            cluster_edge_weight.insert(std::make_pair(new_id, temp_ew+new_delta));
        }
std::cout << "sec3" << std::endl;

        // set new clusterID
        clusters[node] = new_id;
    }

    /**
     * This is one iteration of the clustering algorithm.
     * New cluster assignments are calculated and the containers for the new label assignments are filled 
     * (namely clusters and send_buffers and cluster_node_weight and cluster_edge_weight).
     */
    void cluster_iteration(const DistributedGraph &graph, ClusterArray &clusters, 
                                std::unordered_map<ClusterID, NodeWeight> &cluster_node_weight, 
                                std::unordered_map<ClusterID, EdgeWeight> &cluster_edge_weight, 
                                std::map<PEID, update_vector> send_buffers, 
                                GlobalNodeWeight max_cluster_weight) {
        // calculate_new_cluster for all owned nodes
        for (auto&& node : graph.nodes()) {
std::cout << "start iteration for" << std::endl;
            ClusterID cl_id = calculate_new_cluster(node, graph, clusters, cluster_node_weight, max_cluster_weight);
std::cout << "calculated new cluster" << std::endl;
            if (cl_id != clusters[node]) {
std::cout << "adjust clusters" << std::endl;
                adjust_clusters(graph, node, clusters[node], cl_id, clusters, cluster_node_weight, cluster_edge_weight);
std::cout << "fill send buffers" << std::endl;
                fill_send_buffers(node, clusters, send_buffers, graph);
            }
        }
    }

    std::vector<NodeID>* isolated_nodes(const DistributedGraph &graph) {
        std::vector<NodeID> *isolated = new std::vector<NodeID>;
        for (auto&& node : graph.nodes()) {
            if (graph.adjacent_nodes(node).begin() == graph.adjacent_nodes(node).end()) {
                (*isolated).push_back(node);
            }
        }
        return isolated;
    }

    PEID find_lowest_isolated_PEID(const DistributedGraph &graph) {
        // communicate your PEID, if you have isolated nodes; send -1 if you don't have isolated nodes
        PEID message;
        int size = mpi::get_comm_size(graph.communicator());
        std::vector<NodeID> *isolated = isolated_nodes(graph);
        if((*isolated).size() == 0) {
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
        return lowest;
    }

    /**
     * used to cluster the remaining isolated nodes.
     * if a PEs PEID is higher than that of another one, the lower PEs clusterID is used for the isolated nodes.
    */
    void cluster_isolated_nodes(const DistributedGraph &graph, ClusterArray &clusters) {
        // TODO
        PEID lowest = find_lowest_isolated_PEID(graph);
        PEID rank = mpi::get_comm_rank(graph.communicator());
        MPI_Comm isolated_comm;
        MPI_Comm_split(graph.communicator(), lowest, rank, &isolated_comm);

        if (lowest == -1) {
            return;
        }
        std::vector<NodeID> *isolated = isolated_nodes(graph);
        ClusterID isolated_cluster;
        if (rank == lowest) {
            isolated_cluster = graph.local_to_global_node((*isolated)[0]);
        }
        MPI_Bcast(&isolated_cluster, 1, MPI_INT, lowest, isolated_comm);

        for (NodeID node : (*isolated)) {
            clusters[node] = isolated_cluster;
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
        ClusterArray clusters(graph.total_n());
        MyLPClustering::clusters() = clusters;

        // cluster weights of the clusters
        std::unordered_map<ClusterID, NodeWeight> cluster_node_weight;
        std::unordered_map<ClusterID, EdgeWeight> cluster_edge_weight;

        // MPI rank and size
        int myrank = mpi::get_comm_rank(graph.communicator());
        int size = mpi::get_comm_size(graph.communicator());
std::cout << "total_n: " << myrank << ", " << graph.total_n() << std::endl;
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
        std::vector<update_vector *> send_buf_ptrs;
        update_vector *send_buffer = new update_vector();
        int send_counts[size] = {0};
        int send_displ[size] = {0};

        // receive buffer
        update_vector *recv_buffer = new update_vector();
        int recv_counts[size] = {0};
        int recv_displ[size] = {0};

        // vectors for PEs
        int z = 0;
        for (auto&& [peid, _] : adj_PEs) {
            update_vector *temp = new update_vector();
            send_buf_ptrs.push_back(temp);
            send_buffers.insert(std::make_pair(peid, *send_buf_ptrs[z]));
            z++;
        }
    
        // initialize containers for local clusterIDs and cluster weights
        for (NodeID u : graph.all_nodes()) {
            ClusterID g_id = graph.local_to_global_node(u);
            clusters[u] = g_id;
            cluster_node_weight.insert(std::make_pair(g_id, graph.node_weight(u)));
            cluster_edge_weight.insert(std::make_pair(g_id, 0));
        }
std::cout << "okay" << std::endl;
        // fill send buffers initally
        for (NodeID u : graph.nodes()) {
            fill_send_buffers(u, clusters, send_buffers, graph);
        }
std::cout << "initialization complete" << std::endl;
std::cout << "start communication" << std::endl;
        // communicate labels ()
        MPI_Datatype update_type = mpi::type::get<cluster_update>();
std::cout << "okay1" << std::endl;
        set_up_alltoallv_send(send_buffers, send_buffer, send_counts, send_displ);
std::cout << "okay2" << std::endl;
        set_up_alltoallv_recv(recv_counts, recv_displ, recv_buffer, graph);
std::cout << "okay3: " << recv_counts[(myrank+1)%2] << " " << (*recv_buffer).size() << std::endl;
        int x = MPI_Alltoallv(send_buffer, send_counts, send_displ, update_type, recv_buffer, recv_counts, recv_displ, update_type, graph.communicator());
std::cout << "okay4: " << myrank << ", " <<  x << std::endl;
        mpi::barrier(graph.communicator());
        // evaluate recv_buffer content
std::cout << "okay5" << std::endl;
        evaluate_recv_buffer(recv_buffer, recv_counts, clusters, size, myrank, graph);
std::cout << "okay6" << std::endl;
        // clean up containers
        clean_up_iteration(send_buffers, send_buffer, send_counts, send_displ, recv_buffer);
std::cout << "start global iterations" << std::endl;
        // global cluster iterations
        int global_iterations = 3;
        int local_iterations = 3;
        for (int i = 0; i < global_iterations; i++) {
std::cout << "start global iteration: " << i << std::endl;
            // local cluster iterations
            for (int y = 0; y < local_iterations; y++) {
std::cout << "start local iteration: (" << i << ", " << y << ")" << std::endl;
                cluster_iteration(graph, clusters, cluster_node_weight, cluster_edge_weight, send_buffers, max_cluster_weight);
            }

            // communicate labels
            set_up_alltoallv_send(send_buffers, send_buffer, send_counts, send_displ);
            MPI_Alltoallv(send_buffer, send_counts, send_displ, update_type, recv_buffer, recv_counts, recv_displ, update_type, graph.communicator());

            mpi::barrier(graph.communicator());
            // evaluate recv_buffer content
            evaluate_recv_buffer(recv_buffer, recv_counts, clusters, size, myrank, graph);

            // clean up containers
            clean_up_iteration(send_buffers, send_buffer, send_counts, send_displ, recv_buffer);
        }
std::cout << "start clustering isolated nodes" << std::endl;
        // cluster_isolated_nodes
        cluster_isolated_nodes(graph, clusters);
std::cout << "start return clusterarray" << std::endl;
        //return *(clusters_ptr.get());
        return MyLPClustering::clusters();
    }
}