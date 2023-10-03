/*******************************************************************************
 * @file:   mq_async_global_lp_clusterer.cc
 * @author: Samuel Gil
 * @date:   20.09.2023
 * @brief:  Label propagation with clusters that can grow to multiple PEs. (Code from global_lp_clustering.cc adjusted for this class with additions (Daniel Seemaier))
 ******************************************************************************/
#include "dkaminpar/coarsening/clustering/lp/mq_async_global_lp_clusterer.h"

#include <google/dense_hash_map>

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/growt.h"
#include "dkaminpar/graphutils/communication.h"

#include "kaminpar/label_propagation.h"

#include "common/datastructures/fast_reset_array.h"
#include "common/math.h"

#include <external/message-queue/include/message-queue/buffered_queue_v2.h>

namespace kaminpar::dist {
namespace {
// Wrapper to make google::dense_hash_map<> compatible with
// kaminpar::RatingMap<>.
struct UnorderedRatingMap {
  UnorderedRatingMap() {
    map.set_empty_key(kInvalidGlobalNodeID);
  }

  EdgeWeight &operator[](const GlobalNodeID key) {
    return map[key];
  }

  [[nodiscard]] auto &entries() {
    return map;
  }

  void clear() {
    map.clear();
  }

  std::size_t capacity() const {
    return std::numeric_limits<std::size_t>::max();
  }

  void resize(const std::size_t /* capacity */) {}

  google::dense_hash_map<GlobalNodeID, EdgeWeight> map{};
};

struct MQAsyncGlobalLPClusteringConfig : public LabelPropagationConfig {
  using Graph = DistributedGraph;
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, GlobalNodeID, UnorderedRatingMap>;
  using ClusterID = GlobalNodeID;
  using ClusterWeight = GlobalNodeWeight;

  static constexpr bool kTrackClusterCount = false;         // NOLINT
  static constexpr bool kUseTwoHopClustering = false;       // NOLINT
  static constexpr bool kUseActiveSetStrategy = false;      // NOLINT
  static constexpr bool kUseLocalActiveSetStrategy = false; // NOLINT
};
} // namespace

class MQAsyncGlobalLPClusteringImpl final
    : public ChunkRandomdLabelPropagation<MQAsyncGlobalLPClusteringImpl, MQAsyncGlobalLPClusteringConfig>,
      public NonatomicOwnedClusterVector<NodeID, GlobalNodeID> {
  SET_DEBUG(false);

  using Base = ChunkRandomdLabelPropagation<MQAsyncGlobalLPClusteringImpl, MQAsyncGlobalLPClusteringConfig>;
  using ClusterBase = NonatomicOwnedClusterVector<NodeID, GlobalNodeID>;
  using WeightDeltaMap = growt::GlobalNodeIDMap<GlobalNodeWeight>;
  
  using MessageQueue = message_queue::BufferedMessageQueueV2<std::pair<NodeID, ClusterID>>;
  using WeightsMessageQueue = message_queue::BufferedMessageQueueV2<std::pair<GlobalNodeID, GlobalNodeWeight>>;

  struct Statistics {};

public:
  explicit MQAsyncGlobalLPClusteringImpl(const Context &ctx)
      : ClusterBase{ctx.partition.graph->total_n},
        _ctx(ctx),
        _c_ctx(ctx.coarsening),
        _changed_label(ctx.partition.graph->n),
        _cluster_weights(ctx.partition.graph->total_n - ctx.partition.graph->n),
        _local_cluster_weights(ctx.partition.graph->n),
        _passive_high_degree_threshold(_c_ctx.global_lp.passive_high_degree_threshold) {
    set_max_num_iterations(_c_ctx.global_lp.num_iterations);
    set_max_degree(_c_ctx.global_lp.active_high_degree_threshold);
    set_max_num_neighbors(_c_ctx.global_lp.max_num_neighbors);
  }

  void initialize(const DistributedGraph &graph) {
    _graph = &graph;

    mpi::barrier(graph.communicator());

    SCOPED_TIMER("Initialize label propagation clustering");

    START_TIMER("High-degree computation");
    if (_passive_high_degree_threshold > 0) {
      graph.init_high_degree_info(_passive_high_degree_threshold);
    }
    STOP_TIMER();

    mpi::barrier(graph.communicator());

    START_TIMER("Allocation");
    allocate(graph);
    STOP_TIMER();

    mpi::barrier(graph.communicator());

    START_TIMER("Datastructures");
    // Clear hash map
    _cluster_weights_handles_ets.clear();
    _cluster_weights = ClusterWeightsMap{0};
    std::fill(_local_cluster_weights.begin(), _local_cluster_weights.end(), 0);

    // Initialize data structures
    Base::initialize(&graph, graph.total_n());
    initialize_ghost_node_clusters();
    STOP_TIMER();
  }

  // TODO async
  auto &
  compute_clustering(const DistributedGraph &graph, const GlobalNodeWeight max_cluster_weight) {
    _max_cluster_weight = max_cluster_weight;

    mpi::barrier(graph.communicator());

    KASSERT(_graph == &graph, "must call initialize() before cluster()", assert::always);

    MPI_Comm_dup(graph.communicator(), &_w_comm);
    // message queue (owner_lnode, new_gcluster)
    // optional: make merger, splitter, cleaner
    MessageQueue queue = message_queue::BufferedMessageQueueV2<std::pair<NodeID, ClusterID>>(graph.communicator(), 100);
    queue.global_threshold(_ctx.msg_q_context.global_threshold);
    queue.local_threshold(_ctx.msg_q_context.local_threshold);

    // weights queue (cluster, weight_delta)
    // TODO maybe use a third queue for owning-pe-send messages
    WeightsMessageQueue w_queue = message_queue::BufferedMessageQueueV2<std::pair<GlobalNodeID, GlobalNodeWeight>>(_w_comm, 100);
    w_queue.global_threshold(_ctx.msg_q_context.weights_global_threshold);
    w_queue.local_threshold(_ctx.msg_q_context.weights_local_threshold);

    SCOPED_TIMER("Compute label propagation clustering");

    mpi::barrier(graph.communicator());

    for (int iteration = 0; iteration < _max_num_iterations; ++iteration) {

      NodeID local_num_moved_nodes = 0;

      // asynchronic iteration body
      int counter = 0;
      for (NodeID u = 0; u < graph.n(); ++u) {
        local_num_moved_nodes += process_node(u, queue);

        // if should handle messages now: handle messages
        if (counter < _ctx.msg_q_context.message_handle_threshold) {
          ++counter;
          continue;
        } else {
          counter = 0;

          // weight handling here 
          handle_cluster_weights(w_queue, u);

          handle_messages(queue);
std::cout << "handle_messages done" << std::endl;
        }
      }

std::cout << "loop done" << std::endl;
      mpi::barrier(_graph->communicator());

std::cout << "passed barrier" << std::endl;
      const GlobalNodeID global_num_moved_nodes = 
        mpi::allreduce(local_num_moved_nodes, MPI_SUM, _graph->communicator());

std::cout << "allreduce done" << std::endl;
      if (_c_ctx.global_lp.merge_singleton_clusters) {
        cluster_isolated_nodes(0, graph.n());
      }

std::cout << "cluster_isolated_nodes done" << std::endl;
      // if noting changed during the iteration, end clustering
      if (global_num_moved_nodes == 0) {
        break;
      }
std::cout << "iteration done" << std::endl;
      // terminate and reactivate queue
      handle_cluster_weights(w_queue, graph.n() - 1);
      terminate_queue(queue);
      handle_cluster_weights(w_queue, graph.n() - 1);
      terminate_weights_queue(w_queue, graph.n() - 1);
      w_queue.reactivate();
      queue.reactivate();

std::cout << "iteration termination done" << std::endl;
      _graph->pfor_nodes(0, graph.n(), [&](const NodeID lnode) {
        _changed_label[lnode] = kInvalidGlobalNodeID;
      });
    }
std::cout << "iteration loop done" << std::endl;
    
    // finish handling labels before returning
    handle_cluster_weights(w_queue, graph.n() - 1);
    terminate_queue(queue);
    handle_cluster_weights(w_queue, graph.n() - 1);
    terminate_weights_queue(w_queue, graph.n() - 1);

    // free unused communicator
    MPI_Comm_free(&_w_comm);

std::cout << "return clusters" << std::endl;
    return clusters();
  }

  void set_max_num_iterations(const int max_num_iterations) {
    _max_num_iterations =
        max_num_iterations == 0 ? std::numeric_limits<int>::max() : max_num_iterations;
  }

  //--------------------------------------------------------------------------------
  //
  // Called from base class
  //
  // VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
  void reset_node_state(const NodeID u) {
    Base::reset_node_state(u);
    _changed_label[u] = kInvalidGlobalNodeID;
  }

  /*
   * Cluster weights
   * Note: offset cluster IDs by 1 since growt cannot use 0 as key.
   */

  void init_cluster_weight(const ClusterID lcluster, const ClusterWeight weight) {
    if (_graph->is_owned_node(lcluster)) {
      __atomic_store_n(&_local_cluster_weights[lcluster], weight, __ATOMIC_RELAXED);
    } else {
      KASSERT(lcluster < _graph->total_n());
      const auto gcluster = _graph->local_to_global_node(static_cast<NodeID>(lcluster));
      auto &handle = _cluster_weights_handles_ets.local();
      [[maybe_unused]] const auto [it, success] = handle.insert(gcluster + 1, weight);
      KASSERT(success, "Cluster already initialized: " << gcluster + 1);
    }
  }

  ClusterWeight cluster_weight(const ClusterID gcluster) {
    if (_graph->is_owned_global_node(gcluster)) {
      const NodeID lcluster = _graph->global_to_local_node(gcluster);
      return __atomic_load_n(&_local_cluster_weights[lcluster], __ATOMIC_RELAXED);
    } else {
      auto &handle = _cluster_weights_handles_ets.local();
      auto it = handle.find(gcluster + 1);
      KASSERT(it != handle.end(), "read weight of uninitialized cluster: " << gcluster);
      return (*it).second;
    }
  }

  bool move_cluster_weight(
      const ClusterID old_gcluster,
      const ClusterID new_gcluster,
      const ClusterWeight weight_delta,
      const ClusterWeight max_weight,
      const bool check_weight_constraint = true
  ) {
    // Reject move if it violates local weight constraint
    if (check_weight_constraint && cluster_weight(new_gcluster) + weight_delta > max_weight) {
      return false;
    }

    auto &handle = _cluster_weights_handles_ets.local();

    if (_graph->is_owned_global_node(old_gcluster)) {
      const NodeID old_lcluster = _graph->global_to_local_node(old_gcluster);
      __atomic_fetch_sub(&_local_cluster_weights[old_lcluster], weight_delta, __ATOMIC_RELAXED);
    } else {
      // Otherwise, move node to new cluster
      [[maybe_unused]] const auto [it, found] = handle.update(
          old_gcluster + 1, [](auto &lhs, const auto rhs) { return lhs -= rhs; }, weight_delta
      );
      KASSERT(
          it != handle.end() && found, "moved weight from uninitialized cluster: " << old_gcluster
      );
    }

    if (_graph->is_owned_global_node(new_gcluster)) {
      const NodeID new_lcluster = _graph->global_to_local_node(new_gcluster);
      __atomic_fetch_add(&_local_cluster_weights[new_lcluster], weight_delta, __ATOMIC_RELAXED);
    } else {
      [[maybe_unused]] const auto [it, found] = handle.update(
          new_gcluster + 1, [](auto &lhs, const auto rhs) { return lhs += rhs; }, weight_delta
      );
      KASSERT(
          it != handle.end() && found, "moved weight to uninitialized cluster: " << new_gcluster
      );
    }

    return true;
  }

  void change_cluster_weight(
      const ClusterID gcluster, const ClusterWeight delta, [[maybe_unused]] const bool must_exist
  ) {
    if (_graph->is_owned_global_node(gcluster)) {
      const NodeID lcluster = _graph->global_to_local_node(gcluster);
      __atomic_fetch_add(&_local_cluster_weights[lcluster], delta, __ATOMIC_RELAXED);
    } else {
      auto &handle = _cluster_weights_handles_ets.local();

      [[maybe_unused]] const auto [it, not_found] = handle.insert_or_update(
          gcluster + 1, delta, [](auto &lhs, const auto rhs) { return lhs += rhs; }, delta
      );
      KASSERT(
          it != handle.end() && (!must_exist || !not_found),
          "changed weight of uninitialized cluster: " << gcluster
      );
    }
  }

  [[nodiscard]] NodeWeight initial_cluster_weight(const GlobalNodeID u) {
    KASSERT(u < _graph->total_n());
    return _graph->node_weight(static_cast<NodeID>(u));
  }

  [[nodiscard]] ClusterWeight max_cluster_weight(const GlobalNodeID /* cluster */) {
    return _max_cluster_weight;
  }

  /*
   * Clusters
   */

  void move_node(const NodeID lu, const ClusterID gcluster) {
    KASSERT(lu < _changed_label.size());
    _changed_label[lu] = this->cluster(lu);
    NonatomicOwnedClusterVector::move_node(lu, gcluster);

    // Detect if a node was moved back to its original cluster
    if (_c_ctx.global_lp.prevent_cyclic_moves && gcluster == initial_cluster(lu)) {
      // If the node ID is the smallest among its non-local neighbors, lock the
      // node to its original cluster
      bool interface_node = false;
      bool smallest = true;

      for (const NodeID lv : _graph->adjacent_nodes(lu)) {
        if (_graph->is_owned_node(lv)) {
          continue;
        }

        interface_node = true;
        const GlobalNodeID gu = _graph->local_to_global_node(lu);
        const GlobalNodeID gv = _graph->local_to_global_node(lv);
        if (gv < gu) {
          smallest = false;
          break;
        }
      }

      if (interface_node && smallest) {
        _locked[lu] = 1;
      }
    }
  }

  [[nodiscard]] ClusterID initial_cluster(const NodeID u) {
    return _graph->local_to_global_node(u);
  }

  /*
   * Moving nodes
   */

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight <=
                max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  [[nodiscard]] inline bool activate_neighbor(const NodeID u) {
    return _graph->is_owned_node(u);
  }

  [[nodiscard]] inline bool accept_neighbor(NodeID /* u */, const NodeID v) {
    return _passive_high_degree_threshold == 0 || !_graph->is_high_degree_node(v);
  }

  [[nodiscard]] inline bool skip_node(const NodeID lnode) {
    return _c_ctx.global_lp.prevent_cyclic_moves && _locked[lnode];
  }

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  //
  // Called from base class
  //
  //--------------------------------------------------------------------------------

private:
  void allocate(const DistributedGraph &graph) {
    ensure_cluster_size(graph.total_n());

    const NodeID allocated_num_active_nodes = _changed_label.size();

    if (allocated_num_active_nodes < graph.n()) {
      _changed_label.resize(graph.n());
      _local_cluster_weights.resize(graph.n());
    }

    Base::allocate(graph.total_n(), graph.n(), graph.total_n());

    if (_c_ctx.global_lp.prevent_cyclic_moves) {
      _locked.resize(graph.n());
    }
  }

  void initialize_ghost_node_clusters() {
    tbb::parallel_for(_graph->n(), _graph->total_n(), [&](const NodeID local_u) {
      const GlobalNodeID label = _graph->local_to_global_node(local_u);
      init_cluster(local_u, label);
    });
  }

  void control_cluster_weights(const NodeID from, const NodeID to) {
    START_TIMER("Synchronize cluster weights");

    if (!should_sync_cluster_weights()) {
      return;
    }

    const PEID size = mpi::get_comm_size(_graph->communicator());

    START_TIMER("Allocation");
    _weight_delta_handles_ets.clear();
    _weight_deltas = WeightDeltaMap(0);
    std::vector<parallel::Atomic<std::size_t>> num_messages(size);
    STOP_TIMER();

    START_TIMER("Fill hash table");
    _graph->pfor_nodes(from, to, [&](const NodeID u) {
      if (_changed_label[u] != kInvalidGlobalNodeID) {
        auto &handle = _weight_delta_handles_ets.local();
        const GlobalNodeID old_label = _changed_label[u];
        const GlobalNodeID new_label = cluster(u);
        const NodeWeight weight = _graph->node_weight(u);

        if (!_graph->is_owned_global_node(old_label)) {
          auto [old_it, old_inserted] = handle.insert_or_update(
              old_label + 1, -weight, [&](auto &lhs, auto &rhs) { return lhs -= rhs; }, weight
          );
          if (old_inserted) {
            const PEID owner = _graph->find_owner_of_global_node(old_label);
            ++num_messages[owner];
          }
        }

        if (!_graph->is_owned_global_node(new_label)) {
          auto [new_it, new_inserted] = handle.insert_or_update(
              new_label + 1, weight, [&](auto &lhs, auto &rhs) { return lhs += rhs; }, weight
          );
          if (new_inserted) {
            const PEID owner = _graph->find_owner_of_global_node(new_label);
            ++num_messages[owner];
          }
        }
      }
    });
    STOP_TIMER();

    mpi::barrier(_graph->communicator());

    struct Message {
      GlobalNodeID cluster;
      GlobalNodeWeight delta;
    };

    START_TIMER("Allocation");
    std::vector<NoinitVector<Message>> out_msgs(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { out_msgs[pe].resize(num_messages[pe]); });
    STOP_TIMER();

    mpi::barrier(_graph->communicator());

    START_TIMER("Create messages");
    growt::pfor_handles(
        _weight_delta_handles_ets,
        [&](const GlobalNodeID gcluster_p1, const GlobalNodeWeight weight) {
          const GlobalNodeID gcluster = gcluster_p1 - 1;
          const PEID owner = _graph->find_owner_of_global_node(gcluster);
          const std::size_t index = num_messages[owner].fetch_sub(1) - 1;
          out_msgs[owner][index] = {.cluster = gcluster, .delta = weight};
        }
    );
    STOP_TIMER();

    mpi::barrier(_graph->communicator());

    START_TIMER("Exchange messages");
    auto in_msgs = mpi::sparse_alltoall_get<Message>(out_msgs, _graph->communicator());
    STOP_TIMER();

    mpi::barrier(_graph->communicator());

    START_TIMER("Integrate messages");
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
      tbb::parallel_for<std::size_t>(0, in_msgs[pe].size(), [&](const std::size_t i) {
        const auto [cluster, delta] = in_msgs[pe][i];
        change_cluster_weight(cluster, delta, false);
      });
    });

    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
      tbb::parallel_for<std::size_t>(0, in_msgs[pe].size(), [&](const std::size_t i) {
        const auto [cluster, delta] = in_msgs[pe][i];
        in_msgs[pe][i].delta = cluster_weight(cluster);
      });
    });
    STOP_TIMER();

    mpi::barrier(_graph->communicator());

    START_TIMER("Exchange messages");
    auto in_resps = mpi::sparse_alltoall_get<Message>(in_msgs, _graph->communicator());
    STOP_TIMER();

    mpi::barrier(_graph->communicator());

    START_TIMER("Integrate messages");
    parallel::Atomic<std::uint8_t> violation = 0;
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
      tbb::parallel_for<std::size_t>(0, in_resps[pe].size(), [&](const std::size_t i) {
        const auto [cluster, delta] = in_resps[pe][i];
        GlobalNodeWeight new_weight = delta;
        const GlobalNodeWeight old_weight = cluster_weight(cluster);

        if (delta > _max_cluster_weight) {
          const GlobalNodeWeight increase_by_others = new_weight - old_weight;

          auto &handle = _weight_delta_handles_ets.local();
          auto it = handle.find(cluster + 1);
          KASSERT(it != handle.end());
          const GlobalNodeWeight increase_by_me = (*it).second;

          violation = 1;
          if (_c_ctx.global_lp.enforce_legacy_weight) {
            new_weight = _max_cluster_weight + (1.0 * increase_by_me / increase_by_others) *
                                                   (new_weight - _max_cluster_weight);
          } else {
            new_weight =
                _max_cluster_weight + (1.0 * increase_by_me / (increase_by_others + increase_by_me)
                                      ) * (new_weight - _max_cluster_weight);
          }
        }
        change_cluster_weight(cluster, -old_weight + new_weight, true);
      });
    });
    STOP_TIMER();

    mpi::barrier(_graph->communicator());

    STOP_TIMER();

    // If we detected a max cluster weight violation, remove node weight
    // proportional to our chunk of the cluster weight
    if (!should_enforce_cluster_weights() || !violation) {
      return;
    }

    START_TIMER("Enforce cluster weights");
    _graph->pfor_nodes(from, to, [&](const NodeID u) {
      const GlobalNodeID old_label = _changed_label[u];
      if (old_label == kInvalidGlobalNodeID) {
        return;
      }

      const GlobalNodeID new_label = cluster(u);
      const GlobalNodeWeight new_label_weight = cluster_weight(new_label);
      if (new_label_weight > _max_cluster_weight) {
        move_node(u, old_label);
        move_cluster_weight(new_label, old_label, _graph->node_weight(u), 0, false);
      }
    });
    STOP_TIMER();
  }

  /**
   * label propagation for one node
  */
  NodeID process_node(const NodeID u, MessageQueue &queue) {
    START_TIMER("Node iteration");
    // find cluster to move node to
    const NodeID local_num_moved_nodes = perform_iteration_for_node(u);
    STOP_TIMER();

    if (local_num_moved_nodes == 0) {
      return 0;
    }
    // put messages in send buffer
    int pes;
    MPI_Comm_size((*_graph).communicator(), &pes);
    int added_for_pe[pes] {0};
    if (_changed_label[u] != kInvalidGlobalNodeID) {
      for (const auto [e, v] : (*_graph).neighbors(u)) {
        if (!(*_graph).is_ghost_node(v)) {
          continue;
        }
        const PEID pe = (*_graph).ghost_owner(v);

        if (added_for_pe[pe] == 1) {
          continue;
        }
        std::pair<NodeID, GlobalNodeID> message = std::make_pair(u, cluster(u));
        queue.post_message(message, pe);
        added_for_pe[pe] = 1;
      }
    }

    return local_num_moved_nodes;
  }

  /**
   * provides messge handler for label message queue
  */
  std::function<void(std::vector<std::pair<NodeID, ClusterID>>, PEID, int)> get_message_handler() {
    return [&](std::vector<std::pair<NodeID, ClusterID>> &&buffer, PEID owner, int tag) {

      // handle received messages
      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, buffer.size()), [&](const auto &r) {
        auto &weight_delta_handle = _weight_delta_handles_ets.local();

        // iterate for each interface node, that has received an update
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
          const auto [owner_lnode, new_gcluster] = buffer[i];

          const GlobalNodeID gnode = _graph->offset_n(owner) + owner_lnode;
          KASSERT(!_graph->is_owned_global_node(gnode));

          const NodeID lnode = _graph->global_to_local_node(gnode);
              
          const NodeWeight weight = _graph->node_weight(lnode);

          const GlobalNodeID old_gcluster = cluster(lnode);

          // If we synchronize the weights of clusters with local
          // changes, we already have the right weight including ghost
          // vertices --> only update weight if we did not get an update

          if (!should_sync_cluster_weights() ||
              weight_delta_handle.find(old_gcluster + 1) == weight_delta_handle.end()) {
            change_cluster_weight(old_gcluster, -weight, true);
          }

          // move node to the newly assigned cluster
          NonatomicOwnedClusterVector::move_node(lnode, new_gcluster);
          if (!should_sync_cluster_weights() ||
              weight_delta_handle.find(new_gcluster + 1) == weight_delta_handle.end()) {
            change_cluster_weight(new_gcluster, weight, false);
          }
        }
      });
    };
  }

  // handle label messages
  bool handle_messages(MessageQueue &queue) {
    return queue.poll(get_message_handler());
  }

  // terminate label queue
  bool terminate_queue(MessageQueue &queue) {
    return queue.terminate(get_message_handler());
  }

  /**
   * provides messge handler for weights message queue
   * @var u the NodeID of the node currently being processed
  */
  std::function<void(std::vector<std::pair<GlobalNodeID, GlobalNodeWeight>>, PEID, int)> get_weights_message_handler(const NodeID u, 
          parallel::Atomic<std::uint8_t> &violation, std::vector<NoinitVector<GlobalNodeID>> &owned_clusters) {
    //TODO handling received messages

    /***************************** message handling ******************************/
    /**
     * |-- if cluster is owned
						|-- send back total cluster weight
					|-- if cluster is not owned
						|-- if cluster is too heavy
							|-- adjust cluster
    */
   
   // only aggregate and not send total weights if owned cluster (need weights from all PEs)
    return [&](std::vector<std::pair<GlobalNodeID, GlobalNodeWeight>> &&buffer, PEID sender, int tag) {
      
      const PEID rank = mpi::get_comm_rank(_w_comm);
      
      tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
        const auto [cluster, delta] = buffer[i];
        if (_graph->find_owner_of_global_node(cluster) == rank) {
          // case: cluster is owned
          change_cluster_weight(cluster, delta, false);
          bool contained = false;
          tbb::parallel_for<int>(0, owned_clusters[sender].size(), [&](const int index) {
            if (owned_clusters[sender][index] == cluster) {
              contained = true;
              return;
            }
          });
          if (!contained) {
            owned_clusters[sender].push_back(cluster);
          }
        } else {
          // case: cluster is not owned
std::cout << "not owned case" << std::endl;
          GlobalNodeWeight new_weight = delta;
          const GlobalNodeWeight old_weight = cluster_weight(cluster);

          if (delta > _max_cluster_weight) {
            const GlobalNodeWeight increase_by_others = new_weight - old_weight;

            auto &handle = _weight_delta_handles_ets.local();
            auto it = handle.find(cluster + 1);
            KASSERT(it != handle.end());
            const GlobalNodeWeight increase_by_me = (*it).second;

std::cout << "violation = 1" << std::endl;
            violation = 1;
            if (_c_ctx.global_lp.enforce_legacy_weight) {
              new_weight = _max_cluster_weight + (1.0 * increase_by_me / increase_by_others) *
                                                   (new_weight - _max_cluster_weight);
            } else {
              new_weight =
                  _max_cluster_weight + (1.0 * increase_by_me / (increase_by_others + increase_by_me)
                                      ) * (new_weight - _max_cluster_weight);
            }
          }
          change_cluster_weight(cluster, -old_weight + new_weight, true);
        }
      });
    };
  }

  // TODO handle weights messages
  bool handle_weights_messages(WeightsMessageQueue &w_queue, const NodeID u, parallel::Atomic<std::uint8_t> &violation, std::vector<NoinitVector<GlobalNodeID>> &owned_clusters) {
    return w_queue.poll(get_weights_message_handler(u, violation, owned_clusters));
  }

  // TODO terminate weights queue; need to make changes (violation, owned_clusters not really used)
  bool terminate_weights_queue(WeightsMessageQueue &w_queue, const NodeID u) {
    parallel::Atomic<std::uint8_t> violation = 0;
    int size = mpi::get_comm_size(_w_comm);
    std::vector<NoinitVector<GlobalNodeID>> owned_clusters(size);
    return w_queue.terminate(get_weights_message_handler(u, violation, owned_clusters));
  }
  
  /**
   * handle cluster weights in order to keep the weight restraint
   *
  */ 
  void handle_cluster_weights(WeightsMessageQueue &w_queue, const NodeID u) {
    SCOPED_TIMER("Synchronize cluster weights");

    if (!should_sync_cluster_weights()) {
      return;
    }

    const PEID size = mpi::get_comm_size(_w_comm);

    // no need to fix clustering yet
    if (u < (_max_cluster_weight/size)) {
      return;
    }
    // posting messages for interval [_last_handled_node_weight, u]
    /********************************* setting up and creating messages **************************************/
    _weight_delta_handles_ets.clear();
    _weight_deltas = WeightDeltaMap(0);
    std::vector<parallel::Atomic<std::size_t>> num_messages(size);

    // aggregating weight changes for clusters
    _graph->pfor_nodes(_last_handled_node_weight, u, [&](const NodeID u) {
      if (_changed_label[u] != kInvalidGlobalNodeID) {
        auto &handle = _weight_delta_handles_ets.local();
        const GlobalNodeID old_label = _changed_label[u];
        const GlobalNodeID new_label = cluster(u);
        const NodeWeight weight = _graph->node_weight(u);

        if (!_graph->is_owned_global_node(old_label)) {
          auto [old_it, old_inserted] = handle.insert_or_update(
              old_label + 1, -weight, [&](auto &lhs, auto &rhs) { return lhs -= rhs; }, weight
          );
          if (old_inserted) {
            const PEID owner = _graph->find_owner_of_global_node(old_label);
            ++num_messages[owner];
          }
        }

        if (!_graph->is_owned_global_node(new_label)) {
          auto [new_it, new_inserted] = handle.insert_or_update(
              new_label + 1, weight, [&](auto &lhs, auto &rhs) { return lhs += rhs; }, weight
          );
          if (new_inserted) {
            const PEID owner = _graph->find_owner_of_global_node(new_label);
            ++num_messages[owner];
          }
        }
      }
    });

    // cluster, weight_delta
    typedef std::pair<GlobalNodeID, GlobalNodeWeight> WeightsMessage;

    std::vector<NoinitVector<WeightsMessage>> out_msgs(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { out_msgs[pe].resize(num_messages[pe]); });

    growt::pfor_handles(
        _weight_delta_handles_ets,
        [&](const GlobalNodeID gcluster_p1, const GlobalNodeWeight weight) {
          const GlobalNodeID gcluster = gcluster_p1 - 1;
          const PEID owner = _graph->find_owner_of_global_node(gcluster);
          const std::size_t index = num_messages[owner].fetch_sub(1) - 1;
          out_msgs[owner][index] = std::make_pair(gcluster, weight);
        }
    );

    // post messages
    for (int target = 0; target < size; target++) {
      for (auto && msg: out_msgs[target]) {
        w_queue.post_message(msg, target);
      }
    }

    w_queue.flush_all_buffers();

    parallel::Atomic<std::uint8_t> violation = 0;
    std::vector<NoinitVector<GlobalNodeID>> owned_clusters(size);

    // handle messages
    bool not_empty = false;
    do {
      not_empty = handle_weights_messages(w_queue, u, violation, owned_clusters);
    } while (not_empty);

    // make message vectors to improve communication speed
    std::vector<std::vector<WeightsMessage>> out_msg_vectors(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
      for (GlobalNodeID c : owned_clusters[pe]) {
        out_msg_vectors[pe].push_back(std::make_pair(c, cluster_weight(c)));
      }
    });
    // send the owned cluster's total weights, if there has been a change
    for (int pe = 0; pe < size; ++pe) {
      w_queue.post_message(std::move(out_msg_vectors[pe]), pe, 0);
    }
    
    w_queue.flush_all_buffers();

    // handle messages
    do {
std::cout << "got new messages" << std::endl;
      not_empty = handle_weights_messages(w_queue, u, violation, owned_clusters);
    } while (not_empty);

    // TODO check nodes in reverse from _last_handled_node_weight, if the clusters are not conformative to the weight restraint
    // If we detected a max cluster weight violation, remove node weight
    // proportional to our chunk of the cluster weight
    if (!should_enforce_cluster_weights() || !violation) {
      return;
    }
std::cout << "start handling violations" << std::endl;

    // TODO optimize the calculation, by setting the proper interval or posthandling
    //_graph->pfor_nodes(_last_handled_node_weight, u, [&](const NodeID u) {
    _graph->pfor_nodes(0, u, [&](const NodeID u) {
      const GlobalNodeID old_label = _changed_label[u];
      if (old_label == kInvalidGlobalNodeID) {
        return;
      }

      const GlobalNodeID new_label = cluster(u);
      const GlobalNodeWeight new_label_weight = cluster_weight(new_label);
      if (new_label_weight > _max_cluster_weight) {
        move_node(u, old_label);
        move_cluster_weight(new_label, old_label, _graph->node_weight(u), 0, false);
      }
    });

    // set _last_handled_node_weight
    _last_handled_node_weight = u;

std::cout << "cluster_weights handled" << std::endl;
  }

  /*!
   * Build clusters of isolated nodes: store the first isolated node and add
   * subsequent isolated nodes to its cluster until the maximum cluster weight
   * is violated; then, move on to the next isolated node etc.
   * @param from The first node to consider.
   * @param to One-after the last node to consider.
   */
  void cluster_isolated_nodes(const NodeID from, const NodeID to) {
    SCOPED_TIMER("Cluster isolated nodes");

    tbb::enumerable_thread_specific<GlobalNodeID> isolated_node_ets(kInvalidNodeID);
    tbb::parallel_for(tbb::blocked_range<NodeID>(from, to), [&](tbb::blocked_range<NodeID> r) {
      NodeID current = isolated_node_ets.local();
      ClusterID current_cluster =
          current == kInvalidNodeID ? kInvalidGlobalNodeID : cluster(current);
      ClusterWeight current_weight =
          current == kInvalidNodeID ? kInvalidNodeWeight : cluster_weight(current_cluster);

      for (NodeID u = r.begin(); u != r.end(); ++u) {
        if (_graph->degree(u) == 0) {
          const auto u_cluster = cluster(u);
          const auto u_weight = cluster_weight(u_cluster);

          if (current != kInvalidNodeID &&
              current_weight + u_weight <= max_cluster_weight(u_cluster)) {
            change_cluster_weight(current_cluster, u_weight, true);
            NonatomicOwnedClusterVector::move_node(u, current_cluster);
            current_weight += u_weight;
          } else {
            current = u;
            current_cluster = u_cluster;
            current_weight = u_weight;
          }
        }
      }

      isolated_node_ets.local() = current;
    });

    //mpi::barrier(_graph->communicator());
  }

  [[nodiscard]] bool should_sync_cluster_weights() const {
    return _ctx.coarsening.global_lp.sync_cluster_weights &&
           (!_ctx.coarsening.global_lp.cheap_toplevel ||
            _graph->global_n() != _ctx.partition.graph->global_n);
  }

  [[nodiscard]] bool should_enforce_cluster_weights() const {
    return _ctx.coarsening.global_lp.enforce_cluster_weights &&
           (!_ctx.coarsening.global_lp.cheap_toplevel ||
            _graph->global_n() != _ctx.partition.graph->global_n);
  }

  using Base::_graph;
  const Context &_ctx;
  const CoarseningContext &_c_ctx;

  NodeWeight _max_cluster_weight = std::numeric_limits<NodeWeight>::max();
  int _max_num_iterations = std::numeric_limits<int>::max();

  // If a node was moved during the current iteration: its label before the move
  StaticArray<GlobalNodeID> _changed_label;

  // Used to lock nodes to prevent cyclic node moves
  StaticArray<std::uint8_t> _locked;

  // Weights of non-local clusters (i.e., cluster ID is owned by another PE)
  using ClusterWeightsMap = typename growt::GlobalNodeIDMap<GlobalNodeWeight>;
  ClusterWeightsMap _cluster_weights{0};
  tbb::enumerable_thread_specific<typename ClusterWeightsMap::handle_type>
      _cluster_weights_handles_ets{[&] {
        return _cluster_weights.get_handle();
      }};

  // Weights of local clusters (i.e., cluster ID is owned by this PE)
  StaticArray<GlobalNodeWeight> _local_cluster_weights;

  // Skip neighbors if their degree is larger than this threshold, never skip
  // neighbors if set to 0
  EdgeID _passive_high_degree_threshold = 0;

  WeightDeltaMap _weight_deltas{0};
  tbb::enumerable_thread_specific<WeightDeltaMap::handle_type> _weight_delta_handles_ets{[this] {
    return _weight_deltas.get_handle();
  }};

  // the last node of which the weight has been handled
  NodeID _last_handled_node_weight = 0;

  // MPI communicator for weights related communication
  MPI_Comm _w_comm;
};

//
// Public interface
//

MQAsyncGlobalLPClusterer::MQAsyncGlobalLPClusterer(const Context &ctx)
    : _impl{std::make_unique<MQAsyncGlobalLPClusteringImpl>(ctx)} {}

MQAsyncGlobalLPClusterer::~MQAsyncGlobalLPClusterer() = default;

void MQAsyncGlobalLPClusterer::initialize(const DistributedGraph &graph) {
  _impl->initialize(graph);
}

MQAsyncGlobalLPClusterer::ClusterArray &MQAsyncGlobalLPClusterer::cluster(
    const DistributedGraph &graph, const GlobalNodeWeight max_cluster_weight
) {
  return _impl->compute_clustering(graph, max_cluster_weight);
}
} // namespace kaminpar::dist
