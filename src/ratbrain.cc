#include "ratbrain.hh"
#include <iostream>

using namespace std;

/* ---- PolicyValueNet implementation ---- */

PolicyValueNetImpl::PolicyValueNetImpl()
{
  fc1 = register_module( "fc1", torch::nn::Linear( Memory::datasize, HIDDEN_SIZE ) );
  fc2 = register_module( "fc2", torch::nn::Linear( HIDDEN_SIZE, HIDDEN_SIZE ) );
  policy_wi = register_module( "policy_wi", torch::nn::Linear( HIDDEN_SIZE, NUM_WINDOW_INCREMENT ) );
  policy_wm = register_module( "policy_wm", torch::nn::Linear( HIDDEN_SIZE, NUM_WINDOW_MULTIPLE ) );
  policy_is = register_module( "policy_is", torch::nn::Linear( HIDDEN_SIZE, NUM_INTERSEND ) );
  value_head = register_module( "value_head", torch::nn::Linear( HIDDEN_SIZE, 1 ) );
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PolicyValueNetImpl::forward( torch::Tensor x )
{
  x = torch::relu( fc1->forward( x ) );
  x = torch::relu( fc2->forward( x ) );
  return make_tuple(
    policy_wi->forward( x ),
    policy_wm->forward( x ),
    policy_is->forward( x ),
    value_head->forward( x )
  );
}

/* ---- RatBrain implementation ---- */

RatBrain::RatBrain()
  : _device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU ),
    _network(),
    _optimizer( nullptr ),
    _buf_obs( torch::zeros( {REPLAY_BUFFER_SIZE, static_cast<long>(Memory::datasize)} ) ),
    _buf_utility( torch::zeros( {REPLAY_BUFFER_SIZE} ) ),
    _buf_old_log_prob( torch::zeros( {REPLAY_BUFFER_SIZE} ) ),
    _buf_action_wi( torch::zeros( {REPLAY_BUFFER_SIZE}, torch::kLong ) ),
    _buf_action_wm( torch::zeros( {REPLAY_BUFFER_SIZE}, torch::kLong ) ),
    _buf_action_is( torch::zeros( {REPLAY_BUFFER_SIZE}, torch::kLong ) ),
    _write_pos( 0 ),
    _buffer_count( 0 )
{
  _network->to( _device );
  _optimizer = make_shared<torch::optim::Adam>( _network->parameters(), LEARNING_RATE );
  cerr << "RatBrain using device: " << _device << endl;
}

ActionResult RatBrain::get_window_and_intersend( const Memory & memory, int current_window )
{
  torch::NoGradGuard no_grad;

  /* Convert memory to tensor */
  float obs[Memory::datasize];
  for ( unsigned int i = 0; i < Memory::datasize; i++ ) {
    obs[i] = static_cast<float>( memory.field( i ) );
  }
  auto obs_tensor = torch::from_blob( obs, {1, static_cast<long>(Memory::datasize)} ).to( _device );

  /* Forward pass */
  auto output = _network->forward( obs_tensor );
  auto logits_wi = get<0>( output );
  auto logits_wm = get<1>( output );
  auto logits_is = get<2>( output );

  /* Sample from categorical distributions */
  auto probs_wi = torch::softmax( logits_wi, 1 );
  auto probs_wm = torch::softmax( logits_wm, 1 );
  auto probs_is = torch::softmax( logits_is, 1 );

  int action_wi = torch::multinomial( probs_wi, 1 ).item<int64_t>();
  int action_wm = torch::multinomial( probs_wm, 1 ).item<int64_t>();
  int action_is = torch::multinomial( probs_is, 1 ).item<int64_t>();

  /* Compute log probabilities */
  auto log_probs_wi = torch::log_softmax( logits_wi, 1 );
  auto log_probs_wm = torch::log_softmax( logits_wm, 1 );
  auto log_probs_is = torch::log_softmax( logits_is, 1 );

  float log_prob = log_probs_wi[0][action_wi].item<float>()
                 + log_probs_wm[0][action_wm].item<float>()
                 + log_probs_is[0][action_is].item<float>();

  /* Convert indices to actual parameter values */
  int window_increment = WINDOW_INCREMENT_MIN + action_wi * WINDOW_INCREMENT_STEP;
  double window_multiple = WINDOW_MULTIPLE_MIN + action_wm * WINDOW_MULTIPLE_STEP;
  double intersend = INTERSEND_MIN + action_is * INTERSEND_STEP;

  /* Compute new window (same formula as Whisker::window) */
  int new_window = min( max( 0, static_cast<int>( current_window * window_multiple + window_increment ) ), 1000000 );

  /* Build result */
  ActionResult result;
  result.the_window = new_window;
  result.intersend_time = intersend;

  for ( unsigned int i = 0; i < Memory::datasize; i++ ) {
    result.obs_action.observation[i] = memory.field( i );
  }
  result.obs_action.action_wi_idx = action_wi;
  result.obs_action.action_wm_idx = action_wm;
  result.obs_action.action_is_idx = action_is;
  result.obs_action.old_log_prob = log_prob;

  return result;
}

void RatBrain::remember_episode( double utility, const vector<ObsAction> & observations )
{
  for ( const auto & obs : observations ) {
    for ( unsigned int j = 0; j < Memory::datasize; j++ ) {
      _buf_obs[_write_pos][j] = static_cast<float>( obs.observation[j] );
    }
    _buf_utility[_write_pos] = static_cast<float>( utility );
    _buf_old_log_prob[_write_pos] = obs.old_log_prob;
    _buf_action_wi[_write_pos] = static_cast<int64_t>( obs.action_wi_idx );
    _buf_action_wm[_write_pos] = static_cast<int64_t>( obs.action_wm_idx );
    _buf_action_is[_write_pos] = static_cast<int64_t>( obs.action_is_idx );

    _write_pos = ( _write_pos + 1 ) % REPLAY_BUFFER_SIZE;
    if ( _buffer_count < REPLAY_BUFFER_SIZE ) _buffer_count++;
  }
}

void RatBrain::learn()
{
  if ( _buffer_count < BATCH_SIZE ) return;

  for ( size_t epoch = 0; epoch < PPO_EPOCHS; epoch++ ) {
    /* Sample random batch indices and use vectorized index_select */
    auto indices = torch::randint( 0, static_cast<long>(_buffer_count),
                                   {static_cast<long>(BATCH_SIZE)}, torch::kLong );

    auto obs_batch = _buf_obs.index_select( 0, indices ).to( _device );
    auto utility_batch = _buf_utility.index_select( 0, indices ).to( _device );
    auto old_log_prob_batch = _buf_old_log_prob.index_select( 0, indices ).to( _device );
    auto action_wi_batch = _buf_action_wi.index_select( 0, indices ).to( _device );
    auto action_wm_batch = _buf_action_wm.index_select( 0, indices ).to( _device );
    auto action_is_batch = _buf_action_is.index_select( 0, indices ).to( _device );

    /* Forward pass */
    auto output = _network->forward( obs_batch );
    auto logits_wi = get<0>( output );
    auto logits_wm = get<1>( output );
    auto logits_is = get<2>( output );
    auto values = get<3>( output ).squeeze( 1 );

    /* Compute new log probabilities for taken actions */
    auto log_probs_wi = torch::log_softmax( logits_wi, 1 );
    auto log_probs_wm = torch::log_softmax( logits_wm, 1 );
    auto log_probs_is = torch::log_softmax( logits_is, 1 );

    auto new_log_prob = log_probs_wi.gather( 1, action_wi_batch.unsqueeze( 1 ) ).squeeze( 1 )
                      + log_probs_wm.gather( 1, action_wm_batch.unsqueeze( 1 ) ).squeeze( 1 )
                      + log_probs_is.gather( 1, action_is_batch.unsqueeze( 1 ) ).squeeze( 1 );

    /* Advantage: A(s) = G - V(s), with gamma=1 so G = episode utility */
    auto advantage = utility_batch - values.detach();

    /* PPO clipped surrogate loss */
    auto ratio = torch::exp( new_log_prob - old_log_prob_batch );
    auto surr1 = ratio * advantage;
    auto surr2 = torch::clamp( ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON ) * advantage;
    auto policy_loss = -torch::min( surr1, surr2 ).mean();

    /* Value loss */
    auto value_loss = torch::mse_loss( values, utility_batch );

    /* Entropy bonus */
    auto entropy_wi = -( torch::softmax( logits_wi, 1 ) * log_probs_wi ).sum( 1 );
    auto entropy_wm = -( torch::softmax( logits_wm, 1 ) * log_probs_wm ).sum( 1 );
    auto entropy_is = -( torch::softmax( logits_is, 1 ) * log_probs_is ).sum( 1 );
    auto entropy = ( entropy_wi + entropy_wm + entropy_is ).mean();

    /* Total loss */
    auto loss = policy_loss + VALUE_LOSS_COEFF * value_loss - ENTROPY_COEFF * entropy;

    _optimizer->zero_grad();
    loss.backward();
    _optimizer->step();
  }
}
