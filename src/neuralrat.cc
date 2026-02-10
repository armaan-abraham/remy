#include "neuralrat.hh"

WhiskerTree & NeuralRat::get_dummy_whiskers()
{
  static WhiskerTree dummy;
  return dummy;
}

NeuralRat::NeuralRat( RatBrain & brain )
  : Rat( get_dummy_whiskers() ),
    _brain( brain ),
    _local_network( brain.network()->clone_network() ),
    _episode_observations()
{
}

NeuralRat::NeuralRat( const NeuralRat & other )
  : Rat( other ),
    _brain( other._brain ),
    _local_network( other._local_network->clone_network() ),
    _episode_observations()  /* each copy starts with fresh observations */
{
}

void NeuralRat::update_window_and_intersend()
{
  ActionResult result = infer_action( _local_network, _memory, _the_window );
  _the_window = result.the_window;
  _intersend_time = result.intersend_time;
  _episode_observations.push_back( result.obs_action );
}

size_t NeuralRat::episode_done( double utility, unsigned int num_senders )
{
  size_t episode_size = _episode_observations.size();
  _brain.remember_episode( utility, _episode_observations, num_senders );
  _episode_observations.clear();
  return episode_size;
}
