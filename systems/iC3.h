#include <string>
#include <vector>

#include <drake/common/yaml/yaml_io.h>

#include "common/find_resource.h"
#include "core/c3.h"
#include "core/c3_miqp.h"
#include "core/c3_qp.h"
#include "core/lcs.h"
#include "systems/c3_controller_options.h"
#include "systems/iC3_options.h"
#include "systems/framework/c3_output.h"
#include "systems/framework/timestamped_vector.h"

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"

using std::vector;
using std::pair;

namespace c3 {
namespace systems {

class iC3 : public drake::systems::LeafSystem<double> {

public:
 explicit iC3(
    drake::multibody::MultibodyPlant<double>& plant,
    drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    C3::CostMatrices& costs, 
    C3ControllerOptions controller_options, iC3Options ic3_options);

  vector<vector<MatrixXd>> ComputeTrajectory(
    drake::systems::Context<double>& context,
    drake::systems::Context<drake::AutoDiffXd>& context_ad, 
    const vector<drake::SortedPair<drake::geometry::GeometryId>>& contact_geoms);

private:
  
  // Given an initial x and u trajectory, return x rollout out using lcs
  pair<LCS, MatrixXd> DoLCSRollout(VectorXd x0, MatrixXd u_hat, LCSFactory factory);
  pair<LCS, MatrixXd> DoRollout(VectorXd x0, MatrixXd u_hat, LCSFactory factory);

  LCS MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory);

  // removes num_timesteps_to_remove timesteps from the front of the LCS
  LCS ShortenLCSFront(LCS lcs, int num_timesteps_to_remove);

  C3::CostMatrices ShortenCostsFront(int num_timesteps_to_remove);

  // x_hat (N by n_x), kth row is x at time k
  void UpdateQuaternionCosts(
    MatrixXd x_hat, const Eigen::VectorXd& x_des, vector<VectorXd> c3_quat_norms);

  const drake::multibody::MultibodyPlant<double>& plant_;
  const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad_;

  // C3 options and solver configuration.
  C3ControllerOptions controller_options_;
  iC3Options ic3_options_;

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

};

} // namespace systems
} // namespace c3