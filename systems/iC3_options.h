#pragma once

namespace c3 {

namespace systems {

struct iC3Options {

  int num_iters;
  bool add_position_constraints;
  bool add_input_constraints;
  int num_segments;
  int N_penalize_input_change;

  bool early_termination;

  bool print_costs;


  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(num_iters));
    a->Visit(DRAKE_NVP(N_penalize_input_change));
    a->Visit(DRAKE_NVP(add_position_constraints));
    a->Visit(DRAKE_NVP(add_input_constraints));
    a->Visit(DRAKE_NVP(early_termination));
    a->Visit(DRAKE_NVP(num_segments));
    a->Visit(DRAKE_NVP(print_costs));

  }
};


} // Namespace systems
} // Namespace c3