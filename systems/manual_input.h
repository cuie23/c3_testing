#include "drake/systems/framework/leaf_system.h"
#include "drake/multibody/plant/multibody_plant.h"

#include <Eigen/Dense>
#include <cmath>
#include <iostream> 
#include "core/c3.h"

using drake::multibody::MultibodyPlant;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace c3 {
namespace systems {

// Outputs a manually generated set of inputs
class ManualInput : public drake::systems::LeafSystem<double> {
 public:
  explicit ManualInput(const MultibodyPlant<double>& plant, int N, double dt);

  // Getter for the output port.
  const drake::systems::OutputPort<double>& get_manual_output() const {
    return this->get_output_port(output_port_index_);
  }


 private:
  drake::systems::OutputPortIndex output_port_index_;

  const MultibodyPlant<double>& plant_;
  double dt_; 
  int N_;

  void ComputeManualInput(const drake::systems::Context<double>& context,
                  drake::systems::BasicVector<double>* output) const;


};

} // namespace systems
} // namespace c3