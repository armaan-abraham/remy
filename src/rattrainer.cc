#include <cstdio>
#include <vector>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "ratbrain.hh"
#include "neuralrat.hh"
#include "configrange.hh"
#include "evaluator.hh"

/* Include template implementation files for NeuralRat network instantiation */
#include "network.cc"
#include "rat-templates.cc"

using namespace std;

/* ---- Experience collection and main loop ---- */

double collect_experience( RatBrain & brain,
                           const unsigned int prng_seed,
                           const vector<NetConfig> & configs,
                           const unsigned int tick_count )
{
  PRNG run_prng( prng_seed );
  double total_score = 0;

  for ( auto & config : configs ) {
    /* Run simulation with NeuralRat senders */
    Network< SenderGang<NeuralRat, TimeSwitchedSender<NeuralRat>>,
             SenderGang<NeuralRat, TimeSwitchedSender<NeuralRat>> >
      network( NeuralRat( brain ), run_prng, config );

    network.run_simulation( tick_count );

    double sim_utility = network.senders().utility();
    total_score += sim_utility;

    /* For each sender, call episode_done with the simulation utility */
    auto & gang = network.mutable_senders().mutable_gang1();
    for ( unsigned int i = 0; i < gang.count_senders(); i++ ) {
      gang.mutable_sender( i ).mutable_inner_sender().episode_done( sim_utility );
    }
  }

  return total_score;
}

int main( int argc, char *argv[] )
{
  RemyBuffers::ConfigRange input_config;
  string config_filename;

  for ( int i = 1; i < argc; i++ ) {
    string arg( argv[i] );
    if ( arg.substr( 0, 3 ) == "cf=" ) {
      config_filename = string( arg.substr( 3 ) );
      int cfd = open( config_filename.c_str(), O_RDONLY );
      if ( cfd < 0 ) {
        perror( "open config file error" );
        exit( 1 );
      }
      if ( !input_config.ParseFromFileDescriptor( cfd ) ) {
        fprintf( stderr, "Could not parse input config from file %s.\n", config_filename.c_str() );
        exit( 1 );
      }
      if ( close( cfd ) < 0 ) {
        perror( "close" );
        exit( 1 );
      }
    }
  }

  if ( config_filename.empty() ) {
    fprintf( stderr, "An input configuration protobuf must be provided via the cf= option.\n" );
    fprintf( stderr, "You can generate one using './configuration'.\n" );
    exit( 1 );
  }

  ConfigRange config_range( input_config );
  vector<NetConfig> configs = get_config_outer_product( config_range );
  unsigned int prng_seed = global_PRNG()();
  unsigned int tick_count = config_range.simulation_ticks;

  printf( "#######################\n" );
  printf( "Neural Rat Trainer\n" );
  printf( "Evaluator simulations will run for %d ticks\n", tick_count );
  printf( "#######################\n" );

  RatBrain brain;

  unsigned int run = 0;

  while ( 1 ) {
    double score = collect_experience( brain, prng_seed, configs, tick_count );
    printf( "run = %u, score = %f\n", run, score );

    brain.learn();

    fflush( NULL );
    run++;
  }

  return 0;
}
