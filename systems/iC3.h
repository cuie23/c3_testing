#include <string>
#include <vector>

#include <drake/common/yaml/yaml_io.h>

#include "common/find_resource.h"
#include "core/c3.h"
#include "core/c3_miqp.h"
#include "core/c3_qp.h"
#include "core/lcs.h"
#include "systems/c3_controller_options.h"
#include "systems/framework/c3_output.h"
#include "systems/framework/timestamped_vector.h"

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"

namespace c3 {
namespace systems {

class iC3 : public drake::systems::LeafSystem<double> {

public:
  iC3::iC3(
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::multibody::MultibodyPlant<double>& plant_ad,
    const C3::CostMatrices& costs, C3ControllerOptions controller_options);

  std::vector<VectorXd> iC3::ComputeTrajectory(const Context<double>& context,
      const Context<double>& context_ad,
      const std::vector<drake::SortedPair<drake::geometry::GeometryId>>& contact_geoms);

private:
  
void iC3::UpdateQuaternionCosts(
    std::vector<VectorXd> x_hat, const Eigen::VectorXd& x_des);

  const drake::multibody::MultibodyPlant<double>& plant_;
  const drake::multibody::MultibodyPlant<double>& plant_ad_;

  // C3 options and solver configuration.
  C3ControllerOptions controller_options_;

  // Convenience variables for dimensions.
  int n_q_;       ///< Number of generalized positions.
  int n_v_;       ///< Number of generalized velocities.
  int n_x_;       ///< Total state dimension.
  int n_lambda_;  ///< Number of Lagrange multipliers.
  int n_u_;       ///< Number of control inputs.
  double dt_;     ///< Time step.

  // C3 solver instance.
  mutable std::unique_ptr<C3> c3_;

  // Cost matrices for optimization.
  mutable std::vector<Eigen::MatrixXd> Q_;  ///< State cost matrices.
  mutable std::vector<Eigen::MatrixXd> R_;  ///< Input cost matrices.
  mutable std::vector<Eigen::MatrixXd> G_;  ///< State-input cross-term matrices.
  mutable std::vector<Eigen::MatrixXd> U_;  ///< Constraint matrices.

  int N_;  ///< Horizon length.

} 
} // namespace systems
} // namespace c3