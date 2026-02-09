#ifndef NEURALRAT_HH
#define NEURALRAT_HH

#include <vector>

#include "rat.hh"
#include "ratbrain.hh"

class NeuralRat : private Rat {
private:
  RatBrain & _brain;
  PolicyValueNet _local_network;
  std::vector<ObsAction> _episode_observations;

  static WhiskerTree & get_dummy_whiskers();

  void update_window_and_intersend() override;

public:
  NeuralRat( RatBrain & brain );
  NeuralRat( const NeuralRat & other );

  using Rat::packets_received;
  using Rat::reset;
  using Rat::send;
  using Rat::next_event_time;
  using Rat::packets_sent;
  using Rat::state_DNA;

  size_t episode_done( double utility );
};

#endif
