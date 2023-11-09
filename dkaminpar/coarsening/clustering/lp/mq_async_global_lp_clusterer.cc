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

#undef V
#include <message-queue/buffered_queue.hpp>
#include <message-queue/indirection.hpp>
#include <range/v3/all.hpp>

#include <sparsehash/dense_hash_set>
#include <sparsehash/dense_hash_map>

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

struct LabelMessage {
  uint32_t owner_lnode; // uint32_t
  uint64_t new_gcluster; // uint64_t
};

struct WeightsMessage {
  uint64_t flag:2;
  uint64_t clusterID:62;  // uint64_t
  int64_t delta; // int64_t
};

/**
 * Label Merger
*/
struct LabelMerger {
  template <message_queue::MPIBuffer BufferContainer>
  void operator()(BufferContainer& buffer,
          PEID buffer_destination,
          PEID my_rank,
          message_queue::Envelope auto envelope) const {
    if (!buffer.empty()) {
          buffer.emplace_back(-1);  // sentinel
      }
      buffer.emplace_back(static_cast<uint64_t>(envelope.sender) << 32 | static_cast<uint64_t>(envelope.receiver));
      for (auto elem : envelope.message) {
          buffer.emplace_back(static_cast<uint64_t>(elem.owner_lnode));
          buffer.emplace_back(elem.new_gcluster);
      }
  }
  template <typename MessageContainer, typename BufferContainer>
  size_t estimate_new_buffer_size(BufferContainer const& buffer,
                  PEID buffer_destination,
                  PEID my_rank,
                  message_queue::MessageEnvelope<MessageContainer> const& envelope) const {
    return buffer.size() + envelope.message.size() * 2 + 2;
  };
};
static_assert(message_queue::aggregation::Merger<LabelMerger, LabelMessage, std::vector<uint64_t>>);
static_assert(message_queue::aggregation::EstimatingMerger<LabelMerger, LabelMessage, std::vector<uint64_t>>);

/**
 * Label Splitter
*/
struct LabelSplitter {
    decltype(auto) operator()(message_queue::MPIBuffer<uint64_t> auto const& buffer, PEID buffer_origin, PEID my_rank) const {
      return buffer | std::ranges::views::split(-1)
                    | std::ranges::views::transform([](auto&& chunk) {
                  auto send_recv = chunk[0];
                  auto message = chunk | ranges::views::drop(1)
                                       | ranges::views::chunk(2)
                                       | std::ranges::views::transform([&](auto const& chunk) {
                                            return LabelMessage{ 
                                              .owner_lnode = static_cast<uint32_t>(chunk[0]), .new_gcluster = chunk[1]};
                                          });

                  return message_queue::MessageEnvelope{
                      .message = std::move(message), .sender = static_cast<int>(send_recv >> 32), .receiver = static_cast<int>(send_recv & ((1ULL << 32) - 1)), .tag = 0};
              });
    }
};
static_assert(message_queue::aggregation::Splitter<LabelSplitter, LabelMessage, std::vector<uint64_t>>);

/**
 * Weights Merger
*/
struct WeightsMerger {
  template <message_queue::MPIBuffer BufferContainer>
  void operator()(BufferContainer& buffer,
          PEID buffer_destination,
          PEID my_rank,
          message_queue::Envelope auto envelope) const {
    if (!buffer.empty()) {
          buffer.emplace_back(-1);  // sentinel
      }
      buffer.emplace_back(static_cast<uint64_t>(envelope.sender) << 32 | static_cast<uint64_t>(envelope.receiver));
      for (auto elem : envelope.message) {
          buffer.emplace_back(static_cast<uint64_t>(elem.flag) << 62 | static_cast<uint64_t>(elem.clusterID));
          buffer.emplace_back(static_cast<uint64_t>(elem.delta));
      }
  }
  template <typename MessageContainer, typename BufferContainer>
  size_t estimate_new_buffer_size(BufferContainer const& buffer,
                  PEID buffer_destination,
                  PEID my_rank,
                  message_queue::MessageEnvelope<MessageContainer> const& envelope) const {
    return buffer.size() + envelope.message.size() * 2 + 2;
  };
};
static_assert(message_queue::aggregation::Merger<WeightsMerger, WeightsMessage, std::vector<uint64_t>>);
static_assert(message_queue::aggregation::EstimatingMerger<WeightsMerger, WeightsMessage, std::vector<uint64_t>>);

/**
 * Weights Splitter
*/
struct WeightsSplitter {
  auto operator()(std::vector<uint64_t> const& buffer, PEID buffer_origin, PEID my_rank) const {
    std::vector<message_queue::MessageEnvelope<std::vector<WeightsMessage>>> split_range;
    auto it = buffer.begin();
    size_t counter = 0;
    uint64_t buffered_element;
    int sender;
    int receiver;
    std::vector<WeightsMessage> message;
    while (it != buffer.end()) {
      uint64_t element = *it;
      if (element == -1) {
        split_range.push_back(message_queue::MessageEnvelope{
          .message = std::move(message), .sender = sender, .receiver = receiver, .tag = 0
        });
        message = std::vector<WeightsMessage>();
        ++it;
        counter = 0;
        continue;
      }
      if (counter == 0) {
        sender = static_cast<int>(element >> 32);
        receiver = static_cast<int>(element & ((1ULL << 32) - 1));
      } else if (counter % 2 == 1) {
        buffered_element = element;
      } else /*counter%2 == 0*/ {
        message.push_back(WeightsMessage{
          .flag = buffered_element >> 62, .clusterID = buffered_element & ((1ULL << 62) - 1), .delta = static_cast<int64_t>(element)
        });
      }
      ++it;
      ++counter;
    }
    split_range.push_back(message_queue::MessageEnvelope{
      .message = std::move(message), .sender = sender, .receiver = receiver, .tag = 0
    });
    return std::move(split_range);
  }
};
static_assert(message_queue::aggregation::Splitter<WeightsSplitter, WeightsMessage, std::vector<uint64_t>>);

class MQAsyncGlobalLPClusteringImpl final
    : public MQLabelPropagation<MQAsyncGlobalLPClusteringImpl, MQAsyncGlobalLPClusteringConfig>,
      public NonatomicOwnedClusterVector<NodeID, GlobalNodeID> {
  SET_DEBUG(false);

  using Base = MQLabelPropagation<MQAsyncGlobalLPClusteringImpl, MQAsyncGlobalLPClusteringConfig>;
  using ClusterBase = NonatomicOwnedClusterVector<NodeID, GlobalNodeID>;
  using WeightDeltaMap = growt::GlobalNodeIDMap<GlobalNodeWeight>;

  using MessageQueue = message_queue::BufferedMessageQueue<LabelMessage, uint64_t, std::vector<uint64_t>, LabelMerger, LabelSplitter>;

  using WeightsMessageQueue = message_queue::BufferedMessageQueue<WeightsMessage, uint64_t, std::vector<uint64_t>, WeightsMerger, WeightsSplitter>;

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

    // TODO
    if (_ctx.msg_q_context.lock_then_retry) {
      _global_locked_clusters.set_empty_key(std::numeric_limits<ClusterID>::max());
      _global_locked_clusters.set_deleted_key(std::numeric_limits<ClusterID>::max() - 1);
      
      std::fill(_local_locked_clusters.begin(), _local_locked_clusters.end(), 0);
      
      _unowned_clusters_local_weight.set_empty_key(std::numeric_limits<ClusterID>::max());
      _unowned_clusters_local_weight.set_deleted_key(std::numeric_limits<ClusterID>::max() - 1);

      _weight_deltas.set_empty_key(std::numeric_limits<ClusterID>::max());
      _weight_deltas.set_deleted_key(std::numeric_limits<ClusterID>::max() - 1);
    }

    // Initialize data structures
    Base::initialize(&graph, graph.total_n());
    initialize_ghost_node_clusters();
    STOP_TIMER();
  }

  /**
   * Message Queue for sending Labels (owner_lnode, new_gcluster)
  */
  void make_label_message_queue(const DistributedGraph &graph) {

    // message queue
    _queue = message_queue::make_buffered_queue<LabelMessage, uint64_t>(graph.communicator(), LabelMerger{}, LabelSplitter{});
    
    _queue.global_threshold(_ctx.msg_q_context.global_threshold);
    _queue.local_threshold(_ctx.msg_q_context.local_threshold);

    _queue = message_queue::IndirectionAdapter<message_queue::GridIndirectionScheme, decltype(_queue)>{std::move(_queue)};
  }

  /**
   *  Weights Message Queue sending WeightsMessage (cluster, weight_delta)
  */
  void make_weights_message_queue() {
    
    _w_queue = message_queue::make_buffered_queue<WeightsMessage, uint64_t>(_w_comm, WeightsMerger{}, WeightsSplitter{});

    _w_queue.global_threshold(_ctx.msg_q_context.weights_global_threshold);
    _w_queue.local_threshold(_ctx.msg_q_context.weights_local_threshold);

    _queue = message_queue::IndirectionAdapter<message_queue::GridIndirectionScheme, decltype(_queue)>{std::move(_queue)};
  }

  /**
   * adding indirection functionality to the message queues
  */
  void add_indirection() {
    // add indirection to label message queue
    _queue = message_queue::IndirectionAdapter<message_queue::GridIndirectionScheme, decltype(_queue)>{std::move(_queue)};

    // add indirection to weights message queue
    _w_queue = message_queue::IndirectionAdapter<message_queue::GridIndirectionScheme, decltype(_w_queue)>{std::move(_w_queue)};
  }

  // TODO async
  auto &
  compute_clustering(const DistributedGraph &graph, const GlobalNodeWeight max_cluster_weight) {
    _max_cluster_weight = max_cluster_weight;

    mpi::barrier(graph.communicator());

    KASSERT(_graph == &graph, "must call initialize() before cluster()", assert::always);

    MPI_Comm_dup(graph.communicator(), &_w_comm);

    // label queue
    make_label_message_queue(graph);

    // weights queue
    make_weights_message_queue();

    if (_ctx.msg_q_context.indirection) {
      add_indirection();
    }

    SCOPED_TIMER("Compute label propagation clustering");

    mpi::barrier(graph.communicator());

    for (int iteration = 0; iteration < _max_num_iterations; ++iteration) {

      NodeID local_num_moved_nodes = 0;

      // asynchronic iteration body
      std::size_t label_msg_counter = 0;
      std::size_t weights_msg_counter = 0;
      for (NodeID u = 0; u < graph.n(); ++u) {
        local_num_moved_nodes += process_node(u);

        // TODO
        // seprarate weights and message handling times
        if (weights_msg_counter < _ctx.msg_q_context.weights_handle_threshold) {
          ++weights_msg_counter;
        } else {
          weights_msg_counter = 0;
          
          // weight handling here 
          handle_cluster_weights(u);
              
          // handle received messages
          handle_weights_messages(*(_graph));
        }
        
        // if should handle messages now: handle messages
        if (label_msg_counter < _ctx.msg_q_context.message_handle_threshold) {
          ++label_msg_counter;
          continue;
        } else {
          label_msg_counter = 0;

          // handle received label messages
          handle_messages();
        }
      }

      mpi::barrier(_graph->communicator());

      const GlobalNodeID global_num_moved_nodes = 
        mpi::allreduce(local_num_moved_nodes, MPI_SUM, _graph->communicator());

      if (_c_ctx.global_lp.merge_singleton_clusters) {
        cluster_isolated_nodes(0, graph.n());
      }

      // if nothing changed during the iteration, end clustering
      if (global_num_moved_nodes == 0) {
        break;
      }
      // terminate and reactivate queue
      handle_cluster_weights(graph.n() - 1);
      terminate_queue();
      terminate_weights_queue(graph);
      _w_queue.reactivate();
      _queue.reactivate();

      // cleanup: handle overweight clusters
      if (should_enforce_cluster_weights() && _ctx.msg_q_context.lock_then_retry && _violation) {
        fix_overweight_clusters(graph);
        _violation = false;
      }

      _graph->pfor_nodes(0, graph.n(), [&](const NodeID lnode) {
        _changed_label[lnode] = kInvalidGlobalNodeID;
      });

      // clear the hashmap for the next iteration
      _unowned_clusters_local_weight.clear_no_resize();
    }
    
    // terminate queues
    terminate_queue();
    terminate_weights_queue(graph);

    // free unused communicator
    MPI_Comm_free(&_w_comm);

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
            state.current_cluster == state.initial_cluster) && !is_cluster_locked(*_graph, state.current_cluster);
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

    // TODO
    if (_ctx.msg_q_context.lock_then_retry) {
      // owned locked clusters
      _local_locked_clusters.resize(graph.n());
    }
  }

  void initialize_ghost_node_clusters() {
    tbb::parallel_for(_graph->n(), _graph->total_n(), [&](const NodeID local_u) {
      const GlobalNodeID label = _graph->local_to_global_node(local_u);
      init_cluster(local_u, label);
    });
  }

  /**
   * label propagation for one node
  */
  NodeID process_node(const NodeID u) {
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
        LabelMessage message = { .owner_lnode = u, .new_gcluster = (cluster(u)) };
        _queue.post_message(message, pe);
        added_for_pe[pe] = 1;
      }
    }

    return local_num_moved_nodes;
  }

  /**
   * provides messge handler for label message queue
  */
  auto get_message_handler() {
    return [&](message_queue::Envelope<LabelMessage> auto && envelope) {
      
      // handle received messages
      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, envelope.message.size()), [&](const auto &r) {
        // iterate for each interface node, that has received an update
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
          const auto [owner_lnode, new_gcluster] = envelope.message[i];

          const GlobalNodeID gnode = _graph->offset_n(envelope.sender) + owner_lnode;
          KASSERT(!_graph->is_owned_global_node(gnode));

          const NodeID lnode = _graph->global_to_local_node(gnode);
              
          const NodeWeight weight = _graph->node_weight(lnode);

          const GlobalNodeID old_gcluster = cluster(lnode);

          // we do not adjust the clusters' weights with the additional weight of ghost nodes
          // we get the actual weights when we try to move local nodes to the clusters

          if (!should_sync_cluster_weights()) {
            change_cluster_weight(old_gcluster, -weight, true);
          }

          NonatomicOwnedClusterVector::move_node(lnode, new_gcluster);

          if (!should_sync_cluster_weights()) {
            change_cluster_weight(new_gcluster, weight, false);
          }
        }
      });
    };
  }

  // handle label messages
  bool handle_messages() {
    return _queue.poll(get_message_handler());
  }

  // terminate label queue
  bool terminate_queue() {
    return _queue.terminate(get_message_handler());
  }

  /**
   * provides messge handler for weights message queue lock then retry strategy
   * @var u the NodeID of the node currently being processed
  */
  auto get_weights_message_handler(const DistributedGraph &graph) {

    /***************************** message handling ******************************/
    return [&](message_queue::Envelope<WeightsMessage> auto &&envelope) {

      /**
       * |-- handle message (if available) (flag 2)
					|-- modify local cluster weight
					|-- if cluster is too heavy
						|-- lock cluster
						|-- send lock to sender
      */

      const PEID rank = mpi::get_comm_rank(_w_comm);

      for (size_t i = 0; i < envelope.message.size(); ++i) {
        const auto [flag, cluster, delta] = envelope.message[i];

        if (flag == 0 || flag == 3) {
          // error
          throw std::invalid_argument("Flag is invalid.");
        } else if (flag == 1) {
          // received a message containing a weight change
          // in case of indirection: may need to send it to the receiver  // TODO later
          
          if (envelope.receiver == rank) {
            // case: cluster is owned

            // apply weight change
            change_cluster_weight(cluster, delta, false);
            // if cluster is now too heavy, send cluster-lock
            if (cluster_weight(cluster) >= _max_cluster_weight) {
              _w_queue.post_message({ .flag = 2, .clusterID = cluster, .delta = 0 }, envelope.sender);
            }
          } else {
            // case: cluster is not owned -> need to redirect
            _w_queue.post_message(std::move(envelope.message), envelope.receiver, envelope.sender, envelope.receiver, 0);
            break;
          }
        } else if (flag == 2) {
          // received a message indicating that a cluster got locked
          if (!_violation) {
            _violation = true;
          }
          lock_cluster(graph, cluster);
        }
      }
    };
  }

  // handle weights messages
  bool handle_weights_messages(const DistributedGraph &graph) {
    return _w_queue.poll(get_weights_message_handler(graph));
  }

  // terminate weights queue; need to make changes
  bool terminate_weights_queue(const DistributedGraph &graph) {
    return _w_queue.terminate(get_weights_message_handler(graph));
  }

  // TODO maybe just handle weight for every node separately (not necessary)
  /**
   * weight handling by lock then retry strategy
   * this strategy probably does not get very good results if the update rate is too low
   * (since there may be more additions to full clusters before they get locked)
  */
  void handle_cluster_weights(const NodeID u) {
    SCOPED_TIMER("Synchronize cluster weights");

    if (!should_sync_cluster_weights() || !_ctx.msg_q_context.lock_then_retry) {
      return;
    }

    const PEID size = mpi::get_comm_size(_w_comm);

    // posting messages for interval [_w_last_handled_node, u]
    /********************************* setting up and creating messages **************************************/

    // aggregating weight changes for clusters
    _graph->pfor_nodes(_w_last_handled_node, u, [&](const NodeID u) {
      if (_changed_label[u] != kInvalidGlobalNodeID) {
        const GlobalNodeID old_label = _changed_label[u];
        const GlobalNodeID new_label = cluster(u);
        const NodeWeight weight = _graph->node_weight(u);

        if (!_graph->is_owned_global_node(old_label)) {
          _weight_deltas[old_label] -= weight;

          auto it = _unowned_clusters_local_weight.find(old_label);
          if (it != _unowned_clusters_local_weight.end()) {
            it->second -= weight;
          } else {
            _unowned_clusters_local_weight.insert(std::make_pair(old_label, -weight));
          }
        }

        if (!_graph->is_owned_global_node(new_label)) {
          _weight_deltas[new_label] += weight;

          auto it = _unowned_clusters_local_weight.find(new_label);
          if (it != _unowned_clusters_local_weight.end()) {
            it->second += weight;
          } else {
            _unowned_clusters_local_weight.insert(std::make_pair(new_label, weight));
          }
        }
      }
    });

    // post messages
    for (const auto& [gcluster, weight] : _weight_deltas) {
        const PEID owner = _graph->find_owner_of_global_node(gcluster);
          
        // TODO could add indirection here
        _w_queue.post_message({ .flag = 1, .clusterID = gcluster, .delta = weight }, owner);
    }

    // clearing temporary weight delta map
    _weight_deltas.clear_no_resize();
    
    // set _w_last_handled_node
    _w_last_handled_node = u;
  }

  /**
     * alternatively if I just move back the nodes to their original clusters
     * I can also do so initially when I get the lock message
     * PROBLEM: I don't know which nodes to move back
     * SOLUTION: iterate backwards from last handled node (_w_last_handled_node)
     * might lead to other problems, keep this method in mind for later
     * possible problems:
     *  - when moving the nodes to their original clusters, 
     *    those might be or become overweight, but they won't be handled
     *  - possibly huge overhead because of multiple searches over the same range
     *    (may have to revert multiple moves)
    */
  /** // TODO
   * used to fix overweight clusters
  */
  void fix_overweight_clusters(const DistributedGraph &graph) {

    SCOPED_TIMER("Handle Overweight Clusters");

    mpi::barrier(_graph->communicator());

    /**
     * ask owning pe for each locked cluster's weight
     * owning pe counts the number of pes that have asked for each cluster
     * and sends back the number of pes and the part of the weight exceeding the limit
     * PROBLEM: the pes might not have made as many moves as they have to revert
     * SOLUTION: pes send the local weight of the cluster aswell, 
     *           so that the owning PE can send how much should be reverted for each pe
     * PROBLEM: local weight of unowned cluster is not tracked
     * SOLUTION: create new hashmap to track the local weight of those clusters
    */

    const PEID size = mpi::get_comm_size(_graph->communicator());

    struct Message {
      GlobalNodeID cluster;
      GlobalNodeWeight delta; // local cluster weight
    };

    START_TIMER("Allocation");
    std::vector<std::vector<Message>> out_msgs(size);
    STOP_TIMER();

    START_TIMER("Create Messages");
    for (const auto& element : _global_locked_clusters) {
      PEID owner = _graph->find_owner_of_global_node(element);
      if (_unowned_clusters_local_weight.find(element) != _unowned_clusters_local_weight.end()) {
        out_msgs[owner].push_back({ .cluster = element, .delta = _unowned_clusters_local_weight.find(element)->second });
      }
    }
    STOP_TIMER();

    START_TIMER("Allocation");
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { out_msgs[pe].resize(out_msgs[pe].size()); });
    STOP_TIMER();

    mpi::barrier(_graph->communicator());

    START_TIMER("Exchange messages");
    auto in_msgs = mpi::sparse_alltoall_get<Message>(out_msgs, _graph->communicator());
    STOP_TIMER();

    mpi::barrier(_graph->communicator());

    // map that tracks the new local weight of owned clusters after the changes
    // on remote PEs have been applied
    google::dense_hash_map<ClusterID, GlobalNodeWeight> new_cluster_weight;
    new_cluster_weight.set_empty_key(std::numeric_limits<ClusterID>::max());
    new_cluster_weight.set_deleted_key(std::numeric_limits<ClusterID>::max() - 1);

    START_TIMER("Integrate messages");
    /**
     * owning PE
     * calculate the part of the cluster weight, that has to be reverted per PE and cluster
     * use owned local cluster weight as reference
     * delta: weight to be reverted
    */
    for (PEID pe = 0; pe < size; ++pe) {
      for (size_t i = 0; i < in_msgs[pe].size(); ++i) {
        const auto [cluster, delta] = in_msgs[pe][i];
        GlobalNodeWeight pe_remote_weight = delta;
        const GlobalNodeWeight total_weight = cluster_weight(cluster);

        auto it = new_cluster_weight.find(cluster);
        if (it == new_cluster_weight.end()) {
          new_cluster_weight.insert(std::make_pair(cluster, 0));
        }

        if (total_weight > _max_cluster_weight) {
          GlobalNodeWeight increase_by_pe = pe_remote_weight;
          GlobalNodeWeight increase_by_others = total_weight - _max_cluster_weight - increase_by_pe;
          
          if (_c_ctx.global_lp.enforce_legacy_weight) {
            GlobalNodeWeight weight_to_remove = (1.0 * increase_by_pe / increase_by_others) *
                                                   (total_weight - _max_cluster_weight);
            in_msgs[pe][i].delta = weight_to_remove;
            new_cluster_weight.find(cluster)->second -= weight_to_remove;

          } else {
            GlobalNodeWeight weight_to_remove = (1.0 * increase_by_pe / (increase_by_others + increase_by_pe)
                                      ) * (total_weight - _max_cluster_weight);
            in_msgs[pe][i].delta = weight_to_remove;
            new_cluster_weight.find(cluster)->second -= weight_to_remove;
          }
        } else {
          in_msgs[pe][i].delta = 0;
        }

      }
    }
    // modifying owned cluster weight with the changes on remote PEs
    for (const auto [cluster, weight_to_remove] : new_cluster_weight) {
      change_cluster_weight(cluster, weight_to_remove, true);
      if (cluster_weight(cluster) < _max_cluster_weight && is_cluster_locked(graph, cluster)) {
        unlock_cluster(graph, cluster);
      }
    }
    STOP_TIMER();

    mpi::barrier(_graph->communicator());

    START_TIMER("Exchange messages");
    auto in_resps = mpi::sparse_alltoall_get<Message>(in_msgs, _graph->communicator());
    STOP_TIMER();

    for (PEID pe = 0; pe < size; ++pe) {
      for (size_t i = 0; i < in_resps[pe].size(); ++i) {
        const auto [cluster, delta] = in_resps[pe][i];
        change_cluster_weight(cluster, delta, true);
        if (cluster_weight(cluster) > _max_cluster_weight && !is_cluster_locked(graph, cluster)) {
          lock_cluster(graph, cluster);
        }
      }
    }

    /**
     * just revert the nodes to their previous clusters
     * need to keep track of weight, may need to lock new clusters
    */
    START_TIMER("Enforce cluster weights");
    _graph->pfor_nodes(0, graph.n(), [&](const NodeID u) {
      const GlobalNodeID old_label = _changed_label[u];
      if (old_label == kInvalidGlobalNodeID) {
        return;
      }

      const GlobalNodeID new_label = cluster(u);
      const GlobalNodeWeight new_label_weight = cluster_weight(new_label);
      if (new_label_weight > _max_cluster_weight) {
        move_node(u, old_label);
        move_cluster_weight(new_label, old_label, _graph->node_weight(u), 0, false);
        if (cluster_weight(old_label) > _max_cluster_weight && !is_cluster_locked(graph, old_label)) {
          lock_cluster(graph, old_label);
        }
      }
    });
    STOP_TIMER();
  }

  /**
   * check if cluster is locked
  */
  bool is_cluster_locked(const DistributedGraph &graph, ClusterID cluster) {
    if (!_ctx.msg_q_context.lock_then_retry) {
      return false;
    }
    if (graph.is_owned_global_node(cluster)) {
      NodeID l_cluster = graph.global_to_local_node(cluster);
      return _local_locked_clusters[l_cluster];
    }
    auto it = _global_locked_clusters.find(cluster);
    if (it != _global_locked_clusters.end()) {
      return true;
    }
    return false;
  }

  /**
   * lock a cluster
  */
  void lock_cluster(const DistributedGraph &graph, ClusterID cluster) {
    if (!_ctx.msg_q_context.lock_then_retry) {
      return;
    }
    if (graph.is_owned_global_node(cluster)) {
      NodeID l_cluster = graph.global_to_local_node(cluster);
      _local_locked_clusters[l_cluster] = 1;
    } else {
      // modify or create entry for unowned cluster
      _global_locked_clusters.insert(cluster);
    }
  }

  /**
   * unlock a cluster
  */
  void unlock_cluster(const DistributedGraph &graph, ClusterID cluster) {
    if (!_ctx.msg_q_context.lock_then_retry) {
      return;
    }
    if (graph.is_owned_global_node(cluster)) {
      NodeID l_cluster = graph.global_to_local_node(cluster);
      _local_locked_clusters[l_cluster] = 0;
    } else {
      // modify or create entry for unowned cluster
      _global_locked_clusters.erase(cluster);
    }
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

  // used to track how much weight was newly added to the unowned cluster
  google::dense_hash_map<ClusterID, GlobalNodeWeight> _weight_deltas;

  // the last node of which the weight has been handled
  NodeID _w_last_handled_node = 0;

  // MPI communicator for weights related communication
  MPI_Comm _w_comm;

  // used to lock local clusters to prevent too many moves to full clusters
  // using offset_n() to convert ClusterIDs to NodeIDs(for the indices)
  StaticArray<std::uint8_t> _local_locked_clusters;

  // used to lock global clusters to prevent too many moves to full clusters
  google::dense_hash_set<ClusterID> _global_locked_clusters;

  // weights message queue
  WeightsMessageQueue _w_queue;

  // label message queue
  MessageQueue _queue;

  // used to track the total local weight of unowned clusters (key, value) = (clusterID, localDelta),
  // that was added during the current iteration
  google::dense_hash_map<ClusterID, GlobalNodeWeight> _unowned_clusters_local_weight;

  // whether there was a violation to the weight constraint during the iteration
  bool _violation = false;
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
