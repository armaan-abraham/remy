#ifndef RATBRAIN_HH
#define RATBRAIN_HH

#include <vector>
#include <array>
#include <tuple>
#include <algorithm>
#include <torch/torch.h>

#include "memory.hh"
#include "memoryrange.hh"

/* Observation space: same active axes as the WhiskerTree default for Rat */
constexpr std::array<Axis, 4> ACTIVE_AXES = {
  RemyBuffers::MemoryRange::SEND_EWMA,
  RemyBuffers::MemoryRange::REC_EWMA,
  RemyBuffers::MemoryRange::RTT_RATIO,
  RemyBuffers::MemoryRange::SLOW_REC_EWMA
};
constexpr int INPUT_DIM = ACTIVE_AXES.size();

/* Training hyperparameters */
constexpr size_t REPLAY_BUFFER_SIZE = 5e6;
constexpr size_t BATCH_SIZE = 131072;
constexpr double LEARNING_RATE = 3e-4;
constexpr double PPO_EPSILON = 0.2;
// Update-to-data ratio: number of training iterations after each experience
// collection
constexpr size_t UTD_RATIO = 10; 
constexpr double VALUE_LOSS_COEFF = 1.0;
constexpr double ENTROPY_COEFF = 0.005;
constexpr double MAX_GRAD_NORM = 500.0;
constexpr int HIDDEN_SIZE = 256;
constexpr int NUM_HIDDEN_LAYERS = 2;

/* Action space ranges (matching Whisker optimization settings in whisker.hh) */
constexpr int    WINDOW_INCREMENT_MIN  = 0;
constexpr int    WINDOW_INCREMENT_MAX  = 256;
constexpr int    WINDOW_INCREMENT_STEP = 1;

constexpr double WINDOW_MULTIPLE_MIN  = 0.0;
constexpr double WINDOW_MULTIPLE_MAX  = 1.0;
constexpr double WINDOW_MULTIPLE_STEP = 0.01;

constexpr double INTERSEND_MIN  = 0.25;
constexpr double INTERSEND_MAX  = 3.0;
constexpr double INTERSEND_STEP = 0.05;

/* Derived action space dimensions */
constexpr int NUM_WINDOW_INCREMENT = static_cast<int>((WINDOW_INCREMENT_MAX - WINDOW_INCREMENT_MIN) / WINDOW_INCREMENT_STEP) + 1;
constexpr int NUM_WINDOW_MULTIPLE  = static_cast<int>((WINDOW_MULTIPLE_MAX - WINDOW_MULTIPLE_MIN) / WINDOW_MULTIPLE_STEP) + 1;
constexpr int NUM_INTERSEND        = static_cast<int>((INTERSEND_MAX - INTERSEND_MIN) / INTERSEND_STEP) + 1;

struct ObsAction {
  std::array<double, INPUT_DIM> observation;
  int action_wi_idx;
  int action_wm_idx;
  int action_is_idx;
  float old_log_prob;
};

struct ActionResult {
  int the_window;
  double intersend_time;
  ObsAction obs_action;
};

struct PolicyValueNetImpl : torch::nn::Module {
  torch::nn::Linear input_proj{nullptr};
  std::vector<torch::nn::Linear> hidden_layers;
  std::vector<torch::nn::LayerNorm> layer_norms;
  torch::nn::LayerNorm final_norm{nullptr};
  torch::nn::Linear policy_wi{nullptr}, policy_wm{nullptr}, policy_is{nullptr};
  torch::nn::Linear value_head{nullptr};

  PolicyValueNetImpl();

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  forward( torch::Tensor x );
};

TORCH_MODULE(PolicyValueNet);

class RatBrain {
private:
  torch::Device _device;
  PolicyValueNet _network;
  std::shared_ptr<torch::optim::Adam> _optimizer;

  /* Replay buffer stored as flat tensors for vectorized batch indexing */
  torch::Tensor _buf_obs;          /* [REPLAY_BUFFER_SIZE, INPUT_DIM] float */
  torch::Tensor _buf_utility;      /* [REPLAY_BUFFER_SIZE] float */
  torch::Tensor _buf_old_log_prob; /* [REPLAY_BUFFER_SIZE] float */
  torch::Tensor _buf_action_wi;    /* [REPLAY_BUFFER_SIZE] long */
  torch::Tensor _buf_action_wm;    /* [REPLAY_BUFFER_SIZE] long */
  torch::Tensor _buf_action_is;    /* [REPLAY_BUFFER_SIZE] long */

  size_t _write_pos;
  size_t _buffer_count;

public:
  RatBrain();

  ActionResult get_window_and_intersend( const Memory & memory, int current_window );
  void remember_episode( double utility, const std::vector<ObsAction> & observations );
  void learn();
  void save( const std::string & filename ) const;
};

#endif
