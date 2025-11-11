#pragma once

namespace c3 {

namespace systems {

struct iC3Options {

  int num_iters;
  bool add_position_constraints;
  int num_segments;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(num_iters));
    a->Visit(DRAKE_NVP(add_position_constraints));
    a->Visit(DRAKE_NVP(num_segments));

  }
};


} // Namespace systems
} // Namespace c3