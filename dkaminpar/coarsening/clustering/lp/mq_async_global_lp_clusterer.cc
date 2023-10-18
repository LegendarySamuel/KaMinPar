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
#include <range/v3/all.hpp>

#include <sparsehash/dense_hash_set>

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
      buffer.emplace_back(static_cast<uint64_t>(envelope.sender));
      buffer.emplace_back(static_cast<uint64_t>(envelope.receiver));
      buffer.emplace_back(static_cast<uint64_t>(envelope.tag));
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
    return buffer.size() + envelope.message.size() * 2 + 4;
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
                  auto sender = chunk[0];
                  auto receiver = chunk[1];
                  auto tag = chunk[2];
                  auto message = chunk | ranges::views::drop(3)
                                       | ranges::views::chunk(2)
                                       | std::ranges::views::transform([&](auto const& chunk) {
                                            return LabelMessage(static_cast<uint32_t>(chunk[0]), chunk[1]);
                                          });

                  return message_queue::MessageEnvelope{
                      .message = std::move(message), .sender = static_cast<int>(sender), .receiver = static_cast<int>(receiver), .tag = static_cast<int>(tag)};
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
      buffer.emplace_back(static_cast<uint64_t>(envelope.sender));
      buffer.emplace_back(static_cast<uint64_t>(envelope.receiver));
      buffer.emplace_back(static_cast<uint64_t>(envelope.tag));
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
    return buffer.size() + envelope.message.size() * 2 + 4;
  };
};
static_assert(message_queue::aggregation::Merger<WeightsMerger, WeightsMessage, std::vector<uint64_t>>);
static_assert(message_queue::aggregation::EstimatingMerger<WeightsMerger, WeightsMessage, std::vector<uint64_t>>);

/**
 * Weights Splitter
*/
struct WeightsSplitter {
  auto operator()(message_queue::MPIBuffer<uint64_t> auto const& buffer, PEID buffer_origin, PEID my_rank) const {
    return buffer | std::ranges::views::split(-1) | std::ranges::views::transform([](auto&& chunk) {
                auto sender = chunk[0];
                auto receiver = chunk[1];
                auto tag = chunk[2];
                auto message = chunk | ranges::views::drop(3)
                                     | ranges::views::chunk(2)
                                     | std::ranges::views::transform([&](auto const& chunk) {
                                          return WeightsMessage(chunk[0] >> 62, chunk[0] & ((1ULL << 62) - 1), static_cast<int64_t>(chunk[1]));
                                        });
               return message_queue::MessageEnvelope{
                    .message = std::move(message), .sender = static_cast<int>(sender), .receiver = static_cast<int>(receiver), .tag = static_cast<int>(tag)};
            });
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
  }

  /**
   *  Weights Message Queue sending WeightsMessage (cluster, weight_delta)
  */
  void make_weights_message_queue() {
    
    _w_queue = message_queue::make_buffered_queue<WeightsMessage, uint64_t>(_w_comm, WeightsMerger{}, WeightsSplitter{});

    _w_queue.global_threshold(_ctx.msg_q_context.weights_global_threshold);
    _w_queue.local_threshold(_ctx.msg_q_context.weights_local_threshold);
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
        }

        // if should handle messages now: handle messages
        if (label_msg_counter < _ctx.msg_q_context.message_handle_threshold) {
          ++label_msg_counter;
          continue;
        } else {
          label_msg_counter = 0;

          handle_messages();
        }
      }

      mpi::barrier(_graph->communicator());

      const GlobalNodeID global_num_moved_nodes = 
        mpi::allreduce(local_num_moved_nodes, MPI_SUM, _graph->communicator());

      if (_c_ctx.global_lp.merge_singleton_clusters) {
        cluster_isolated_nodes(0, graph.n());
      }

      // if noting changed during the iteration, end clustering
      if (global_num_moved_nodes == 0) {
        break;
      }
      // terminate and reactivate queue
      handle_cluster_weights(graph.n() - 1);
      terminate_queue();
      terminate_weights_queue(graph.n() - 1, graph);
      _w_queue.reactivate();
      _queue.reactivate();

      _graph->pfor_nodes(0, graph.n(), [&](const NodeID lnode) {
        _changed_label[lnode] = kInvalidGlobalNodeID;
      });
    }
    
    // finish handling labels before returning
    handle_cluster_weights(graph.n() - 1);
    terminate_queue();
    terminate_weights_queue(graph.n() - 1, graph);

    // free unused communicator
    MPI_Comm_free(&_w_comm);

    // TODO handle overweight clusters
    if (fix_overweight_clusters(graph)) {
      return compute_clustering(graph, max_cluster_weight);
    }

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
            state.current_cluster == state.initial_cluster) && is_cluster_locked(*_graph, state.current_cluster);
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
    return [&](message_queue::Envelope<LabelMessage> auto const& envelope) {
      
      // handle received messages
      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, envelope.message.size()), [&](const auto &r) {
        auto &weight_delta_handle = _weight_delta_handles_ets.local();

        // iterate for each interface node, that has received an update
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
          const auto [owner_lnode, new_gcluster] = envelope.message[i];

          const GlobalNodeID gnode = _graph->offset_n(envelope.sender) + owner_lnode;
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
  auto get_weights_message_handler(const NodeID u, const DistributedGraph &graph) {

    /***************************** message handling ******************************/
    return [&](message_queue::Envelope<WeightsMessage> auto const &&envelope) {

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
          // in case of indirection: may need to send it to the receiver
          
          if (envelope.receiver == rank) {
            // case: cluster is owned

            // apply weight change
            change_cluster_weight(cluster, delta, false);
            // if cluster is now too heave, send cluster-lock
            if (cluster_weight(cluster) >= _max_cluster_weight) {
              _w_queue.post_message({ .flag = 2, .clusterID = cluster, .delta = 0 }, envelope.sender);
            }
          } else {
            // case: cluster is not owned -> need to redirect
            _w_queue.post_message(envelope.message, envelope.receiver, envelope.sender, envelope.receiver, 0);
            break;
          }
        } else if (flag == 2) {
          // received a message indicating that a cluster got locked
          lock_cluster(graph, cluster);
        }
      }
    };
  }

  // handle weights messages
  bool handle_weights_messages(const NodeID u, const DistributedGraph &graph) {
    return _w_queue.poll(get_weights_message_handler(u, graph));
  }

  // terminate weights queue; need to make changes
  bool terminate_weights_queue(const NodeID u, const DistributedGraph &graph) {
    return _w_queue.terminate(get_weights_message_handler(u, graph));
  }
  
  /** @deprecated
   * handle cluster weights in order to keep the weight restraint
   *
  */ 
  /*void handle_cluster_weights(const NodeID u, const DistributedGraph &graph) {
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

    std::vector<NoinitVector<WeightsMessage>> out_msgs(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { out_msgs[pe].resize(num_messages[pe]); });

    growt::pfor_handles(
        _weight_delta_handles_ets,
        [&](const GlobalNodeID gcluster_p1, const GlobalNodeWeight weight) {
          const GlobalNodeID gcluster = gcluster_p1 - 1;
          const PEID owner = _graph->find_owner_of_global_node(gcluster);
          const std::size_t index = num_messages[owner].fetch_sub(1) - 1;
          out_msgs[owner][index] = { .clusterID = gcluster, .delta = weight };
        }
    );

    // post messages
    for (int target = 0; target < size; target++) {
      for (auto && msg: out_msgs[target]) {
        _w_queue.post_message(msg, target);
      }
    }

    _w_queue.flush_all_buffers();

    parallel::Atomic<std::uint8_t> violation = 0;
    std::vector<NoinitVector<GlobalNodeID>> owned_clusters(size);

    // handle messages
    bool not_empty = false;
    do {
      not_empty = handle_weights_messages(u, graph);
    } while (not_empty);

    // make message vectors to improve communication speed
    std::vector<std::vector<WeightsMessage>> out_msg_vectors(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
      for (GlobalNodeID c : owned_clusters[pe]) {
        out_msg_vectors[pe].push_back({ .clusterID = c, .delta = cluster_weight(c) });
      }
    });
    // send the owned cluster's total weights, if there has been a change
    for (int pe = 0; pe < size; ++pe) {
      _w_queue.post_message(std::move(out_msg_vectors[pe]), pe, 0);
    }
    
    _w_queue.flush_all_buffers();

    // handle messages
    do {
      not_empty = handle_weights_messages(u, graph);
    } while (not_empty);

    // TODO check nodes in reverse from _last_handled_node_weight, if the clusters are not conformative to the weight restraint
    // If we detected a max cluster weight violation, remove node weight
    // proportional to our chunk of the cluster weight
    if (!should_enforce_cluster_weights() || !violation) {
      return;
    }

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
  }*/

  /**
   * weight handling by lock then retry strategy
   * this strategy probably does not get very good results if the update rate is too low
   * (since there may be more additions to full clusters before they get locked)
  */
  void handle_cluster_weights(const NodeID u) {
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
    // clearing temporary weight delta map
    _weight_delta_handles_ets.clear();
    _weight_deltas = WeightDeltaMap(0);

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
        }

        if (!_graph->is_owned_global_node(new_label)) {
          auto [new_it, new_inserted] = handle.insert_or_update(
              new_label + 1, weight, [&](auto &lhs, auto &rhs) { return lhs += rhs; }, weight
          );
        }
      }
    });

    // post messages
    growt::pfor_handles(
        _weight_delta_handles_ets,
        [&](const GlobalNodeID gcluster_p1, const GlobalNodeWeight weight) {
          const GlobalNodeID gcluster = gcluster_p1 - 1;
          const PEID owner = _graph->find_owner_of_global_node(gcluster);
          // TODO could add indirection here
          _w_queue.post_message({ .flag = 1, .clusterID = gcluster, .delta = weight }, owner);
        }
    );
    
    // set _last_handled_node_weight
    _last_handled_node_weight = u;
  }

  /** // TODO
   * used to fix overweight clusters
   * @return whether there were overweight clusters
  */
  bool fix_overweight_clusters(const DistributedGraph &graph) {

    mpi::barrier(_graph->communicator());

    struct Message {

    };
    /**
     * ask owning pe for each locked cluster's weight
     * owning pe counts the number of pes that have asked for each cluster
     * and sends back the number of pes and the part of the weight exceeding the limit
     * PROBLEM: the pes might not have made as many moves as they have to revert
     * SOLUTION: pes send the local weight of the cluster aswell (this is not tracked though)
    */

    /**
     * need to send to certain target PE that might not be a neighbor
    */

  }

  /**
   * check if cluster is locked
  */
  bool is_cluster_locked(const DistributedGraph &graph, ClusterID cluster) {
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

  WeightDeltaMap _weight_deltas{0};
  tbb::enumerable_thread_specific<WeightDeltaMap::handle_type> _weight_delta_handles_ets{[this] {
    return _weight_deltas.get_handle();
  }};

  // the last node of which the weight has been handled
  NodeID _last_handled_node_weight = 0;

  // MPI communicator for weights related communication
  MPI_Comm _w_comm;

  // used to lock local clusters to prevent too many moves to full clusters
  // using offset_n() to convert ClusterIDs to NodeIDs(for the indices)
  StaticArray<std::uint8_t> _local_locked_clusters;

  // used to lock global clusters to prevent too many moves to full clusters
  google::dense_hash_set<ClusterID> _global_locked_clusters;

  WeightsMessageQueue _w_queue;

  MessageQueue _queue;
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