#include <cstdio>
#include <vector>
#include <string>
#include <future>
#include <mutex>
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

/* Default number of times to replicate the config list per collect_experience call,
   so more total experience is collected per iteration. */
const unsigned int DEFAULT_NUM_CONFIG_EVALS = 8;

/* ---- Experience collection and main loop ---- */

double collect_experience( RatBrain & brain,
                           const unsigned int prng_seed,
                           const vector<NetConfig> & configs,
                           const unsigned int tick_count )
{
  mutex brain_mutex;

  /* Generate deterministic per-config PRNG seeds (each thread needs its own) */
  PRNG seed_prng( prng_seed );
  vector<unsigned int> seeds;
  for ( size_t i = 0; i < configs.size(); i++ ) {
    seeds.push_back( seed_prng() );
  }

  /* Launch a parallel async task for each config (mirrors breeder.cc pattern) */
  size_t total_events = 0;
  vector<future<double>> futures;

  for ( size_t i = 0; i < configs.size(); i++ ) {
    futures.push_back(
      async( launch::async,
        [&brain, &configs, &brain_mutex, &total_events, tick_count]
        ( unsigned int seed, size_t idx ) -> double {
          PRNG run_prng( seed );

          /* Run simulation with NeuralRat senders */
          Network< SenderGang<NeuralRat, TimeSwitchedSender<NeuralRat>>,
                   SenderGang<NeuralRat, TimeSwitchedSender<NeuralRat>> >
            network( NeuralRat( brain ), run_prng, configs[idx] );

          network.run_simulation( tick_count );

          double sim_utility = network.senders().utility();

          /* Safely record experience into the shared replay buffer.
             Inference (get_window_and_intersend) is read-only under NoGradGuard
             and safe to call concurrently, but remember_episode writes to the
             shared buffer and needs mutex protection. */
          {
            lock_guard<mutex> lock( brain_mutex );
            auto & gang = network.mutable_senders().mutable_gang1();
            for ( unsigned int j = 0; j < gang.count_senders(); j++ ) {
              total_events += gang.mutable_sender( j ).mutable_inner_sender().episode_done( sim_utility );
            }
          }

          return sim_utility;
        },
        seeds[i], i )
    );
  }

  /* Collect results from all futures (blocks until each completes) */
  double total_score = 0;
  for ( auto & f : futures ) {
    total_score += f.get();
  }

  printf( "total_events = %zu\n", total_events );

  return total_score;
}

int main( int argc, char *argv[] )
{
  RemyBuffers::ConfigRange input_config;
  string config_filename;
  string output_filename;
  unsigned int save_every = 0;
  unsigned int num_config_evals = DEFAULT_NUM_CONFIG_EVALS;

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
    } else if ( arg.substr( 0, 3 ) == "of=" ) {
      output_filename = string( arg.substr( 3 ) );
    } else if ( arg.substr( 0, 11 ) == "save_every=" ) {
      save_every = stoul( arg.substr( 11 ) );
    } else if ( arg.substr( 0, 17 ) == "num_config_evals=" ) {
      num_config_evals = stoul( arg.substr( 17 ) );
    }
  }

  if ( config_filename.empty() ) {
    fprintf( stderr, "An input configuration protobuf must be provided via the cf= option.\n" );
    fprintf( stderr, "You can generate one using './configuration'.\n" );
    exit( 1 );
  }

  ConfigRange config_range( input_config );
  vector<NetConfig> base_configs = get_config_outer_product( config_range );
  unsigned int tick_count = config_range.simulation_ticks;

  vector<NetConfig> configs;
  configs.reserve( base_configs.size() * num_config_evals );
  for ( unsigned int r = 0; r < num_config_evals; r++ ) {
    configs.insert( configs.end(), base_configs.begin(), base_configs.end() );
  }

  printf( "#######################\n" );
  printf( "Neural Rat Trainer\n" );
  printf( "  config file:       %s\n", config_filename.c_str() );
  printf( "  output prefix:     %s\n", output_filename.empty() ? "(none)" : output_filename.c_str() );
  printf( "  save_every:        %u\n", save_every );
  printf( "  num_config_evals:  %u\n", num_config_evals );
  printf( "  base configs:      %zu\n", base_configs.size() );
  printf( "  total configs:     %zu\n", configs.size() );
  printf( "  tick_count:        %u\n", tick_count );
  printf( "#######################\n" );

  RatBrain brain;

  unsigned int run = 0;

  while ( 1 ) {
    unsigned int prng_seed = global_PRNG()();
    double score = collect_experience( brain, prng_seed, configs, tick_count );
    printf( "run = %u, score = %f\n", run, score );

    brain.learn();

    if ( save_every > 0 && !output_filename.empty() && ( run % save_every == 0 ) ) {
      char of[ 256 ];
      snprintf( of, 256, "%s.%u", output_filename.c_str(), run );
      brain.save( string( of ) );
    }

    fflush( NULL );
    run++;
  }

  return 0;
}
