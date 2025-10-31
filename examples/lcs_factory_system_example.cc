// Includes for core controllers, simulators, and test problems.
#include <algorithm>
#include <cmath>
#include <atomic>
#include <csignal>
#include <chrono>
#include <thread>
#include <iostream>

#include <drake/geometry/drake_visualizer.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/meshcat_visualizer_params.h>
#include <drake/multibody/meshcat/contact_visualizer.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/tree/multibody_element.h>
#include <drake/multibody/tree/rigid_body.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/primitives/constant_value_source.h>
#include <drake/systems/primitives/constant_vector_source.h>
#include <drake/systems/primitives/demultiplexer.h>
#include <drake/systems/primitives/multiplexer.h>
#include <drake/systems/primitives/zero_order_hold.h>
#include <drake/systems/lcm/lcm_publisher_system.h>
#include <drake/lcm/drake_lcm.h>
#include <gflags/gflags.h>

#include "core/test/c3_cartpole_problem.hpp"
#include "examples/common_systems.hpp"
#include "systems/c3_controller.h"
#include "systems/c3_controller_options.h"
#include "systems/iC3.h"
#include "systems/common/system_utils.hpp"
#include "systems/lcs_factory_system.h"
#include "systems/lcs_simulator.h"
#include "lcm/lcm_trajectory.h"

#include "c3/lcmt_timestamped_saved_traj.hpp"
#include "c3/lcmt_saved_traj.hpp"
#include "c3/lcmt_trajectory_block.hpp"

#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"

#include <drake/systems/primitives/saturation.h>
#include <drake/systems/framework/leaf_system.h>



DEFINE_string(experiment_type, "cube_pivoting",
              "The type of experiment to test the LCSFactorySystem with. "
              "Options: 'cartpole_softwalls [Frictionless Spring System]', "
              "'cube_pivoting [Stewart and Trinkle System]'");
DEFINE_string(lcm_url, "udpm://239.255.76.67:7667?ttl=0",
              "LCM URL with IP, port, and TTL settings");
DEFINE_string(diagram_path, "",
              "Path to store the diagram (.ps) for the system. If empty, will "
              "be ignored");

using c3::systems::C3Controller;
using c3::systems::C3ControllerOptions;
using c3::systems::LCSFactorySystem;
using c3::systems::LCSSimulator;
using c3::LcmTrajectory;

using drake::SortedPair;
using drake::math::RigidTransformd;
using drake::math::RotationMatrixd;
using drake::geometry::GeometryId;
using drake::geometry::SceneGraph;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::DiagramBuilder;
using drake::systems::rendering::MultibodyPositionToGeometryPose;
using drake::systems::lcm::LcmPublisherSystem;
using drake::systems::TriggerType;
using drake::systems::TriggerTypeSet;

class TimedGravityCompGate final : public drake::systems::LeafSystem<double> {
 public:
  // input_size = # of actuators (n_u)
  TimedGravityCompGate(const MultibodyPlant<double>& plant, double t_delay, int input_size)
    : plant_(plant), t_delay_(t_delay), input_size_(input_size) {
    this->DeclareVectorInputPort("x_state", plant.num_multibody_states()+1);
    this->DeclareVectorInputPort("u_ctrl", input_size_);
    this->DeclareVectorOutputPort("u_out",
        drake::systems::BasicVector<double>(input_size_),
        &TimedGravityCompGate::CalcOutput);
  }

 private:
  void CalcOutput(const drake::systems::Context<double>& context,
                  drake::systems::BasicVector<double>* output) const {
    const auto& x_in = this->get_input_port(0).Eval(context);     // [t; q; v]


		Eigen::VectorXd x = x_in.head(x_in.size() - 1);  // skip timestamp
    const auto& u_ctrl = this->get_input_port(1).Eval(context);
    double t = context.get_time();

    // Compute gravity torques/forces τ_g
    // Need a temporary plant context
		std::unique_ptr<drake::systems::Context<double>> plant_context = plant_.CreateDefaultContext();		
		auto& context_ref = *plant_context;  

    plant_.SetPositionsAndVelocities(&context_ref, x);

    Eigen::VectorXd tau_g = plant_.CalcGravityGeneralizedForces(context_ref);
    Eigen::VectorXd u_gravity = Eigen::VectorXd::Zero(plant_.num_actuators());

		// u_gravity[0] = -1 * (tau_g[0]);
		// u_gravity[1] = -1 * (tau_g[1]);
		u_gravity[2] = -1 * (tau_g[2] + tau_g[10]); // Hard-coded cube + plate
		u_gravity[3] = (tau_g[10] * x[10]); // Hard-coded cube + plate
		u_gravity[4] = (tau_g[10] * x[9]); // Hard-coded cube + plate

		// u_gravity[4] = -1 * (tau_g[4]); // Hard-coded cube + plate

    if (t < t_delay_) {
      // For first 5 s: only gravity compensation
      output->SetFromVector(u_gravity);
			//std::cout << "tau_g: " << tau_g.transpose() << std::endl;
			//std::cout << "u_gravity: " << u_gravity.transpose() << std::endl;

    } else {
      // After 5 s: controller + gravity compensation
      output->SetFromVector(u_ctrl);
			//std::cout << u_ctrl.transpose() << std::endl;
    }
		//std::cout << u_ctrl.transpose() << std::endl;

  }

  const MultibodyPlant<double>& plant_;
  double t_delay_;
  int input_size_;
};

class TrajToLcmSystem : public drake::systems::LeafSystem<double> {
 public:
  TrajToLcmSystem(std::vector<MatrixXd> traj_set)
      : traj_set_(traj_set) {
    this->DeclareAbstractOutputPort(
        "traj_message",
        &TrajToLcmSystem::CalcMessage);
  }

 private:
  void CalcMessage(
      const drake::systems::Context<double>& context,
      lcmt_timestamped_saved_traj* msg) const {

      LcmTrajectory lcm_traj;

      for (int i = 0; i < traj_set_.size(); i++) {
        c3::LcmTrajectory::Trajectory traj;
        std::string name = "iteration_" + std::to_string(i);

        MatrixXd traj_i = traj_set_.at(i);

        traj.traj_name = name;
        traj.datatypes = std::vector<std::string>(traj_i.rows(), "double");
        traj.datapoints = traj_i;

        VectorXd timestamps(traj_i.rows());
        for (int t = 0; t < traj_i.rows(); t++) {
          timestamps(t) = t;
        }
        traj.time_vector = timestamps.cast<double>();
        if (i == 0) {
          lcm_traj = LcmTrajectory({traj}, {name}, name, name, false);
        } else {
          lcm_traj.AddTrajectory(traj.traj_name, traj);
        }
      }
      msg->saved_traj = lcm_traj.GenerateLcmObject();
      msg->utime = context.get_time() * 1e6;   

      std::cout << "calc message running" << std::endl;
  }
  

  std::vector<MatrixXd> traj_set_;
};


class SoftWallReactionForce final : public drake::systems::LeafSystem<double> {
  // Converted to C++ from cartpole_softwall.py by Hien
  // This class provides the spatial reaction forces given the state of the
  // cartpole
 public:
  explicit SoftWallReactionForce(
      const MultibodyPlant<double>* cartpole_softwalls,
      double wall_stiffness = 100, double left_wall_xpos = -0.35,
      double right_wall_xpos = 0.35, double pole_length = 0.6)
      : cartpole_softwalls_(cartpole_softwalls),
        wall_stiffness_(wall_stiffness),
        left_wall_xpos_(left_wall_xpos),
        right_wall_xpos_(right_wall_xpos),
        pole_length_(pole_length) {
    this->DeclareVectorInputPort("cartpole_state", 4);
    this->DeclareAbstractOutputPort(
        "spatial_forces", &SoftWallReactionForce::CalcSoftWallSpatialForce);
    pole_body_ = &cartpole_softwalls_->GetBodyByName("Pole");
  }

  double get_wall_stiffness() { return wall_stiffness_; }

 private:
  void CalcSoftWallSpatialForce(
      const drake::systems::Context<double>& context,
      std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>*
          output) const {
    output->resize(1);
    (*output)[0].body_index = pole_body_->index();
    (*output)[0].p_BoBq_B = pole_body_->default_com();

    // Get Input
    const auto& cartpole_state = get_input_port(0).Eval(context);

    // Calculate wall force
    double pole_tip_xpos =
        cartpole_state[0] - pole_length_ * sin(cartpole_state[1]);
    double left_wall_force =
        std::max(0.0, left_wall_xpos_ - pole_tip_xpos) * wall_stiffness_;
    double right_wall_force =
        std::min(0.0, right_wall_xpos_ - pole_tip_xpos) * wall_stiffness_;
    double wall_force = 0.0;
    if (left_wall_force != 0)
      wall_force = left_wall_force;
    else if (right_wall_force != 0)
      wall_force = right_wall_force;

    // Set force value
    (*output)[0].F_Bq_W = drake::multibody::SpatialForce<double>(
        drake::Vector3<double>::Zero() /* no torque */,
        drake::Vector3<double>(wall_force, 0, 0));
  }
  const MultibodyPlant<double>* cartpole_softwalls_{nullptr};
  const drake::multibody::RigidBody<double>* pole_body_;
  double wall_stiffness_;
  double left_wall_xpos_;
  double right_wall_xpos_;
  double pole_length_;
};

int RunCartpoleTest() {
  // Initialize the C3 cartpole problem. Assuming SDF matches default values in
  // problem.
  auto c3_cartpole_problem = C3CartpoleProblem();
  c3_cartpole_problem.UseC3Plus();  // Use C3+ controller.

  DiagramBuilder<double> plant_builder;
  auto [plant_for_lcs, scene_graph_for_lcs] =
      AddMultibodyPlantSceneGraph(&plant_builder, 0.01);
  Parser parser_for_lcs(&plant_for_lcs, &scene_graph_for_lcs);
  const std::string file_for_lcs =
      "examples/resources/cartpole_softwalls/cartpole_softwalls.sdf";
  parser_for_lcs.AddModels(file_for_lcs);
  plant_for_lcs.Finalize();

  //   // LCS Factory System
  auto plant_diagram = plant_builder.Build();

  std::vector<drake::geometry::GeometryId> left_wall_contact_points =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("left_wall"));

  std::vector<drake::geometry::GeometryId> right_wall_contact_points =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("right_wall"));

  std::vector<drake::geometry::GeometryId> pole_point_geoms =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("Pole"));
  std::unordered_map<std::string, std::vector<drake::geometry::GeometryId>>
      contact_geoms;
  contact_geoms["LEFT_WALL"] = left_wall_contact_points;
  contact_geoms["RIGHT_WALL"] = right_wall_contact_points;
  contact_geoms["POLE_POINT"] = pole_point_geoms;

  std::vector<SortedPair<GeometryId>> contact_pairs;
  contact_pairs.emplace_back(contact_geoms["LEFT_WALL"][0],
                             contact_geoms["POLE_POINT"][0]);
  contact_pairs.emplace_back(contact_geoms["RIGHT_WALL"][0],
                             contact_geoms["POLE_POINT"][0]);

  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.01);
  Parser parser(&plant, &scene_graph);
  const std::string file =
      "examples/resources/cartpole_softwalls/"
      "cartpole_softwalls_no_collision_walls.sdf";
  parser.AddModels(file);
  plant.Finalize();

  C3ControllerOptions options = drake::yaml::LoadYamlFile<C3ControllerOptions>(
      "examples/resources/cartpole_softwalls/"
      "c3_controller_cartpole_options.yaml");
  options.projection_type = "C3+";  // Use C3+ controller.

  std::unique_ptr<drake::systems::Context<double>> plant_diagram_context =
      plant_diagram->CreateDefaultContext();

  auto plant_autodiff =
      drake::systems::System<double>::ToAutoDiffXd(plant_for_lcs);

  auto& plant_for_lcs_context = plant_diagram->GetMutableSubsystemContext(
      plant_for_lcs, plant_diagram_context.get());

  auto plant_context_autodiff = plant_autodiff->CreateDefaultContext();
  auto lcs_factory_system = builder.AddSystem<LCSFactorySystem>(
      plant_for_lcs, plant_for_lcs_context, *plant_autodiff,
      *plant_context_autodiff, contact_pairs, options.lcs_factory_options);

  // Add the C3 controller.
  auto c3_controller = builder.AddSystem<C3Controller>(
      plant_for_lcs, c3_cartpole_problem.cost, options);

  // Add constant vector source for the desired state.
  auto xdes = builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
      c3_cartpole_problem.xdesired.at(0));

  // Add vector-to-timestamped-vector converter.
  auto vector_to_timestamped_vector =
      builder.AddSystem<Vector2TimestampedVector>(4);
  builder.Connect(plant.get_state_output_port(),
                  vector_to_timestamped_vector->get_input_port_state());

  // Connect controller inputs.
  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      c3_controller->get_input_port_lcs_state());
  builder.Connect(lcs_factory_system->get_output_port_lcs(),
                  c3_controller->get_input_port_lcs());
  builder.Connect(xdes->get_output_port(),
                  c3_controller->get_input_port_target());

  // Add and connect C3 solution input system.
  auto c3_input = builder.AddSystem<C3Solution2Input>(1);
  builder.Connect(c3_controller->get_output_port_c3_solution(),
                  c3_input->get_input_port_c3_solution());
  builder.Connect(c3_input->get_output_port_c3_input(),
                  plant.get_actuation_input_port());

  // Add a ZeroOrderHold system for state updates.
  auto state_zero_order_hold =
      builder.AddSystem<drake::systems::ZeroOrderHold<double>>(
          1 / options.publish_frequency, c3_cartpole_problem.k);
  builder.Connect(c3_input->get_output_port_c3_input(),
                  state_zero_order_hold->get_input_port());

  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      lcs_factory_system->get_input_port_lcs_state());
  builder.Connect(state_zero_order_hold->get_output_port(),
                  lcs_factory_system->get_input_port_lcs_input());

  // Add the SoftWallReactionForce system.
  auto soft_wall_reaction_force = builder.AddSystem<SoftWallReactionForce>(
      &plant_for_lcs, c3_cartpole_problem.ks, c3_cartpole_problem.d2,
      c3_cartpole_problem.d1, c3_cartpole_problem.len_p);
  // Connect the SoftWallReactionForce system to the LCSFactorySystem.
  builder.Connect(plant.get_state_output_port(),
                  soft_wall_reaction_force->get_input_port());
  // Connect the SoftWallReactionForce output to the plant's external forces.
  builder.Connect(soft_wall_reaction_force->get_output_port(),
                  plant.get_applied_spatial_force_input_port());

  // Set up Meshcat visualizer.
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  drake::geometry::MeshcatVisualizerParams params;
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, std::move(params));

  drake::multibody::meshcat::ContactVisualizer<double>::AddToBuilder(
      &builder, plant, meshcat,
      drake::multibody::meshcat::ContactVisualizerParams());

  auto diagram = builder.Build();

  if (!FLAGS_diagram_path.empty())
    c3::systems::common::DrawAndSaveDiagramGraph(*diagram, FLAGS_diagram_path);

  // Create a default context for the diagram.
  auto diagram_context = diagram->CreateDefaultContext();

  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  plant.SetPositionsAndVelocities(&plant_context, c3_cartpole_problem.x0);
  // Create and configure the simulator.
  drake::systems::Simulator<double> simulator(*diagram,
                                              std::move(diagram_context));

  simulator.set_target_realtime_rate(
      0.25);  // Run simulation at real-time speed.
  simulator.Initialize();
  simulator.AdvanceTo(10.0);  // Run
  //   simulation for 10 seconds.

  return 0;
}

int RunPivotingTest() {
  // Build the plant and scene graph for the pivoting system.
  DiagramBuilder<double> plant_builder;
  auto [plant_for_lcs, scene_graph_for_lcs] =
      AddMultibodyPlantSceneGraph(&plant_builder, 0.0);
  Parser parser_for_lcs(&plant_for_lcs, &scene_graph_for_lcs);
  const std::string file_for_lcs =
      "examples/resources/cube_pivoting/cube_pivoting.sdf";
  parser_for_lcs.AddModels(file_for_lcs);
  plant_for_lcs.Finalize();

  // Build the plant diagram.
  auto plant_diagram = plant_builder.Build();

  // Retrieve collision geometries for relevant bodies.
  std::vector<drake::geometry::GeometryId> platform_collision_geoms =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("platform"));
  std::vector<drake::geometry::GeometryId> cube_collision_geoms =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("cube"));
  std::vector<drake::geometry::GeometryId> left_finger_collision_geoms =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("left_finger"));
  std::vector<drake::geometry::GeometryId> right_finger_collision_geoms =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("right_finger"));

  // Map collision geometries to their respective components.
  std::unordered_map<std::string, std::vector<drake::geometry::GeometryId>>
      contact_geoms;
  contact_geoms["PLATFORM"] = platform_collision_geoms;
  contact_geoms["CUBE"] = cube_collision_geoms;
  contact_geoms["LEFT_FINGER"] = left_finger_collision_geoms;
  contact_geoms["RIGHT_FINGER"] = right_finger_collision_geoms;

  // Define contact pairs for the LCS system.
  std::vector<SortedPair<GeometryId>> contact_pairs;
  contact_pairs.emplace_back(contact_geoms["CUBE"][0],
                             contact_geoms["LEFT_FINGER"][0]);
  contact_pairs.emplace_back(contact_geoms["CUBE"][0],
                             contact_geoms["PLATFORM"][0]);
  contact_pairs.emplace_back(contact_geoms["CUBE"][0],
                             contact_geoms["RIGHT_FINGER"][0]);

  // Build the main diagram.
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.01);
  Parser parser(&plant, &scene_graph);
  const std::string file = "examples/resources/cube_pivoting/cube_pivoting.sdf";
  parser.AddModels(file);
  plant.Finalize();

  // Load controller options and cost matrices.
  C3ControllerOptions options = drake::yaml::LoadYamlFile<C3ControllerOptions>(
      "examples/resources/cube_pivoting/"
      "c3_controller_pivoting_options.yaml");
  C3::CostMatrices cost = C3::CreateCostMatricesFromC3Options(
      options.c3_options, options.lcs_factory_options.N);

  // Create contexts for the plant and LCS factory system.
  std::unique_ptr<drake::systems::Context<double>> plant_diagram_context =
      plant_diagram->CreateDefaultContext();
  auto plant_autodiff =
      drake::systems::System<double>::ToAutoDiffXd(plant_for_lcs);
  auto& plant_for_lcs_context = plant_diagram->GetMutableSubsystemContext(
      plant_for_lcs, plant_diagram_context.get());
  auto plant_context_autodiff = plant_autodiff->CreateDefaultContext();

  // Add the LCS factory system.
  auto lcs_factory_system = builder.AddSystem<LCSFactorySystem>(
      plant_for_lcs, plant_for_lcs_context, *plant_autodiff,
      *plant_context_autodiff, contact_pairs, options.lcs_factory_options);

  // Add the C3 controller.
  auto c3_controller =
      builder.AddSystem<C3Controller>(plant_for_lcs, cost, options);
  c3_controller->set_name("c3_controller");

  // Add linear constraints to the controller.
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(14, 14);
  A(3, 3) = 1;
  A(4, 4) = 1;
  A(5, 5) = 1;
  A(6, 6) = 1;
  Eigen::VectorXd lower_bound(14);
  Eigen::VectorXd upper_bound(14);
  lower_bound << 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  upper_bound << 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0;
  c3_controller->AddLinearConstraint(A, lower_bound, upper_bound,
                                     ConstraintVariable::STATE);

  // Add a constant vector source for the desired state.
  Eigen::VectorXd xd(14);
  xd << 0, 0.75, 0.785, -0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0;
  auto xdes =
      builder.AddSystem<drake::systems::ConstantVectorSource<double>>(xd);

  // Add a vector-to-timestamped-vector converter.
  auto vector_to_timestamped_vector =
      builder.AddSystem<Vector2TimestampedVector>(14);
  builder.Connect(plant.get_state_output_port(),
                  vector_to_timestamped_vector->get_input_port_state());

  // Connect controller inputs.
  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      c3_controller->get_input_port_lcs_state());
  builder.Connect(lcs_factory_system->get_output_port_lcs(),
                  c3_controller->get_input_port_lcs());
  builder.Connect(xdes->get_output_port(),
                  c3_controller->get_input_port_target());

  // Add and connect the C3 solution input system.
  auto c3_input = builder.AddSystem<C3Solution2Input>(4);
  builder.Connect(c3_controller->get_output_port_c3_solution(),
                  c3_input->get_input_port_c3_solution());
  builder.Connect(c3_input->get_output_port_c3_input(),
                  plant.get_actuation_input_port());

  // Add a ZeroOrderHold system for state updates.
  auto input_zero_order_hold =
      builder.AddSystem<drake::systems::ZeroOrderHold<double>>(
          1 / options.publish_frequency, 4);
  builder.Connect(c3_input->get_output_port_c3_input(),
                  input_zero_order_hold->get_input_port());
  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      lcs_factory_system->get_input_port_lcs_state());
  builder.Connect(input_zero_order_hold->get_output_port(),
                  lcs_factory_system->get_input_port_lcs_input());

  // Set up Meshcat visualizer.
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  drake::geometry::MeshcatVisualizerParams params;
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, std::move(params));
  drake::multibody::meshcat::ContactVisualizer<double>::AddToBuilder(
      &builder, plant, meshcat,
      drake::multibody::meshcat::ContactVisualizerParams());

  // Build the diagram.
  auto diagram = builder.Build();

  if (!FLAGS_diagram_path.empty())
    c3::systems::common::DrawAndSaveDiagramGraph(*diagram, FLAGS_diagram_path);

  // Create a default context for the diagram.
  auto diagram_context = diagram->CreateDefaultContext();

  // Set the initial state of the system.
  Eigen::VectorXd x0(14);
  x0 << 0, 0.75, 0, -0.6, 0.75, 0.1, 0.125, 0, 0, 0, 0, 0, 0, 0;
  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());
  plant.SetPositionsAndVelocities(&plant_context, x0);

  // Create and configure the simulator.
  drake::systems::Simulator<double> simulator(*diagram,
                                              std::move(diagram_context));
  simulator.set_target_realtime_rate(1);  // Run simulation at real-time speed.
  simulator.Initialize();
  simulator.AdvanceTo(10.0);  // Run simulation for 10 seconds.

  return 0;
}

int RunPlateTest() {
  // Build the plant and scene graph for the pivoting system.
  DiagramBuilder<double> plant_builder;
  auto [plant_for_lcs, scene_graph_for_lcs] =
      AddMultibodyPlantSceneGraph(&plant_builder, 0);
  Parser parser_for_lcs(&plant_for_lcs, &scene_graph_for_lcs);

  const std::string plate_file_lcs = "examples/resources/plate/plate.sdf";
	const std::string cube_file_lcs = "examples/resources/plate/cube.sdf";

  parser_for_lcs.AddModels(plate_file_lcs);
  parser_for_lcs.AddModels(cube_file_lcs);

  plant_for_lcs.Finalize();

  // Build the plant diagram.
  auto plant_diagram = plant_builder.Build();

  // Retrieve collision geometries for relevant bodies.
  drake::geometry::GeometryId plate_collision_geom =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("plate"))[0];
	std::vector<drake::geometry::GeometryId> cube_collision_geoms;
  for (int i = 1; i <= 8; i++) {
		cube_collision_geoms.push_back(
			plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("cube"))[i]);
	}

  // Define contact pairs for the LCS system.
  std::vector<SortedPair<GeometryId>> contact_pairs;

	for (GeometryId geom_id : cube_collision_geoms) {
		contact_pairs.emplace_back(plate_collision_geom, geom_id);
	}

  // Build the main diagram.
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.001);
  Parser parser(&plant, &scene_graph);
  const std::string plate_file = "examples/resources/plate/plate.sdf";
	const std::string cube_file = "examples/resources/plate/cube.sdf";

  parser.AddModels(plate_file);
  parser.AddModels(cube_file);

  plant.Finalize();

  // Load controller options and cost matrices.
  C3ControllerOptions options = drake::yaml::LoadYamlFile<C3ControllerOptions>(
      "examples/resources/plate/c3_controller_plate_options.yaml");
  C3::CostMatrices cost = C3::CreateCostMatricesFromC3Options(
      options.c3_options, options.lcs_factory_options.N);

  // Create contexts for the plant and LCS factory system.
  std::unique_ptr<drake::systems::Context<double>> plant_diagram_context =
      plant_diagram->CreateDefaultContext();
  auto plant_autodiff =
      drake::systems::System<double>::ToAutoDiffXd(plant_for_lcs);
  auto& plant_for_lcs_context = plant_diagram->GetMutableSubsystemContext(
      plant_for_lcs, plant_diagram_context.get());
  auto plant_context_autodiff = plant_autodiff->CreateDefaultContext();

  // Add the LCS factory system.
  auto lcs_factory_system = builder.AddSystem<LCSFactorySystem>(
      plant_for_lcs, plant_for_lcs_context, *plant_autodiff,
      *plant_context_autodiff, contact_pairs, options.lcs_factory_options);

	for (const auto& pname : plant.GetPositionNames()) {
		std::cout << pname << std::endl;
	}
	for (const auto& vname : plant.GetVelocityNames()) {
		std::cout << vname << std::endl;
	}
	std::cout << "Before add C3 controller" << std::endl;

  // Add the C3 controller.
  auto c3_controller =
      builder.AddSystem<C3Controller>(plant_for_lcs, cost, options);
  c3_controller->set_name("c3_controller");

	std::cout << "After add C3 controller" << std::endl;

  // Add linear constraints to the controller.
  // Eigen::MatrixXd A = Eigen::MatrixXd::Zero(26, 26);
  // A(4, 4) = 1;
  // A(5, 5) = 1;
  // A(6, 6) = 1;
	// A(11, 11) = 1;
  // A(12, 12) = 1;
  // A(13, 13) = 1;
  // Eigen::VectorXd lower_bound(26);
  // Eigen::VectorXd upper_bound(26);
  // lower_bound << 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  // upper_bound << 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  // c3_controller->AddLinearConstraint(A, lower_bound, upper_bound,
  //                                    ConstraintVariable::STATE);

  Eigen::VectorXd xd(23);
  //xd << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	xd << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  std::vector<double> x_des = *options.x_des;
  xd = Eigen::Map<Eigen::VectorXd>(x_des.data(), x_des.size()); 

	std::cout << "xd: " << xd.transpose() << std::endl;

  auto xdes =
      builder.AddSystem<drake::systems::ConstantVectorSource<double>>(xd);

  // Add a vector-to-timestamped-vector converter.
  auto vector_to_timestamped_vector =
      builder.AddSystem<Vector2TimestampedVector>(23);
  builder.Connect(plant.get_state_output_port(),
                  vector_to_timestamped_vector->get_input_port_state());

	auto gate = builder.AddSystem<TimedGravityCompGate>(plant, 1.0, 5);
  // Connect controller inputs.
  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      c3_controller->get_input_port_lcs_state());

	builder.Connect(vector_to_timestamped_vector->get_output_port_timestamped_state(),
                gate->get_input_port(0));

  builder.Connect(lcs_factory_system->get_output_port_lcs(),
                  c3_controller->get_input_port_lcs());
  builder.Connect(xdes->get_output_port(),
                  c3_controller->get_input_port_target());

  // Add and connect the C3 solution input system.
  auto c3_input = builder.AddSystem<C3Solution2Input>(5);
  builder.Connect(c3_controller->get_output_port_c3_solution(),
                  c3_input->get_input_port_c3_solution());

	builder.Connect(c3_input->get_output_port_c3_input(),
									gate->get_input_port(1));
	builder.Connect(gate->get_output_port(),
									plant.get_actuation_input_port());

  // Add a ZeroOrderHold system for state updates.
  auto input_zero_order_hold =
      builder.AddSystem<drake::systems::ZeroOrderHold<double>>(
          1 / options.publish_frequency, 5);
  builder.Connect(c3_input->get_output_port_c3_input(),
                  input_zero_order_hold->get_input_port());
  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      lcs_factory_system->get_input_port_lcs_state());
  builder.Connect(input_zero_order_hold->get_output_port(),
                  lcs_factory_system->get_input_port_lcs_input());



	Eigen::Vector4d q_vec = xd.segment(5, 4);
	Eigen::Quaterniond q(q_vec(0), q_vec(1), q_vec(2), q_vec(3));
	q.normalize();
  RotationMatrixd R_target(q);
	RigidTransformd X_WF(R_target, xd.segment(9, 3));

  // Set up Meshcat visualizer.
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  drake::geometry::MeshcatVisualizerParams params;

  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, std::move(params));

  drake::multibody::meshcat::ContactVisualizer<double>::AddToBuilder(
      &builder, plant, meshcat,
      drake::multibody::meshcat::ContactVisualizerParams());

	const double axis_len = 0.2;
	const double radius = 0.01;

	meshcat->SetObject("target_pose/x_axis", drake::geometry::Cylinder(radius, axis_len), drake::geometry::Rgba(1, 0, 0, 1));
	RigidTransformd X_FX(
		RotationMatrixd::MakeYRotation(-M_PI / 2.0),
		Eigen::Vector3d(axis_len / 2.0, 0, 0));
	meshcat->SetTransform("target_pose/x_axis", X_WF * X_FX);

	meshcat->SetObject("target_pose/y_axis", drake::geometry::Cylinder(radius, axis_len), drake::geometry::Rgba(0, 1, 0, 1));
	RigidTransformd X_FY(
		RotationMatrixd::MakeXRotation(M_PI / 2.0),
		Eigen::Vector3d(0, axis_len / 2.0, 0));
	meshcat->SetTransform("target_pose/y_axis", X_WF * X_FY);

	meshcat->SetObject("target_pose/z_axis", drake::geometry::Cylinder(radius, axis_len), drake::geometry::Rgba(0, 0, 1, 1));
	RigidTransformd X_FZ(
		RotationMatrixd::Identity(),
		Eigen::Vector3d(0, 0, axis_len / 2.0));
	meshcat->SetTransform("target_pose/z_axis", X_WF * X_FZ);


  // Build the diagram.
  auto diagram = builder.Build();

  if (!FLAGS_diagram_path.empty())
    c3::systems::common::DrawAndSaveDiagramGraph(*diagram, FLAGS_diagram_path);

  // Create a default context for the diagram.
  auto diagram_context = diagram->CreateDefaultContext();

  // Set the initial state of the system.
  Eigen::VectorXd x0(23);
	x0 << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  // x0 << 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  std::vector<double> x_init = *options.x_init;
  x0 = Eigen::Map<Eigen::VectorXd>(x_init.data(), x_init.size());
	std::cout << "x0: " << x0.transpose() << std::endl;

	auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());
  plant.SetPositionsAndVelocities(&plant_context, x0);

  // Create and configure the simulator.
  drake::systems::Simulator<double> simulator(*diagram,
                                              std::move(diagram_context));
  simulator.set_target_realtime_rate(0.25);  // Run simulation at real-time speed.
  simulator.Initialize();
  simulator.AdvanceTo(120.0);  // Run simulation for 10 seconds.

  return 0;
}

std::atomic<bool> g_run{true};
void SigIntHandler(int) { g_run.store(false); }

int RunPlateTestiC3(drake::lcm::DrakeLcm& lcm) {
  // Build the plant and scene graph for the pivoting system.
  DiagramBuilder<double> plant_builder;
  auto [plant_for_lcs, scene_graph_for_lcs] =
      AddMultibodyPlantSceneGraph(&plant_builder, 0);
  Parser parser_for_lcs(&plant_for_lcs, &scene_graph_for_lcs);

  const std::string plate_file_lcs = "examples/resources/plate/plate.sdf";
	const std::string cube_file_lcs = "examples/resources/plate/cube.sdf";

  parser_for_lcs.AddModels(plate_file_lcs);
  parser_for_lcs.AddModels(cube_file_lcs);

  plant_for_lcs.Finalize();

  // Build the plant diagram.
  auto plant_diagram = plant_builder.Build();

  // Retrieve collision geometries for relevant bodies.
  drake::geometry::GeometryId plate_collision_geom =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("plate"))[0];
	std::vector<drake::geometry::GeometryId> cube_collision_geoms;
  for (int i = 1; i <= 8; i++) {
		cube_collision_geoms.push_back(
			plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("cube"))[i]);
	}

  // Define contact pairs for the LCS system.
  std::vector<SortedPair<GeometryId>> contact_pairs;

	for (GeometryId geom_id : cube_collision_geoms) {
		contact_pairs.emplace_back(plate_collision_geom, geom_id);
	}

  // Build the main diagram.
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.001);
  Parser parser(&plant, &scene_graph);
  const std::string plate_file = "examples/resources/plate/plate.sdf";
	const std::string cube_file = "examples/resources/plate/cube.sdf";

  parser.AddModels(plate_file);
  parser.AddModels(cube_file);

  plant.Finalize();

  // Load controller options and cost matrices.
  C3ControllerOptions options = drake::yaml::LoadYamlFile<C3ControllerOptions>(
      "examples/resources/plate/c3_controller_plate_options.yaml");
  C3::CostMatrices cost = C3::CreateCostMatricesFromC3Options(
      options.c3_options, options.lcs_factory_options.N);

  // Create contexts for the plant and LCS factory system.
  std::unique_ptr<drake::systems::Context<double>> plant_diagram_context =
      plant_diagram->CreateDefaultContext();
  auto plant_autodiff =
      drake::systems::System<double>::ToAutoDiffXd(plant_for_lcs);
  auto& plant_for_lcs_context = plant_diagram->GetMutableSubsystemContext(
      plant_for_lcs, plant_diagram_context.get());
  auto plant_context_autodiff = plant_autodiff->CreateDefaultContext(); 

  auto ic3_controller = systems::iC3(plant_for_lcs, *plant_autodiff, cost, options);
  
  std::vector<MatrixXd> ic3_trajs = ic3_controller.ComputeTrajectory(plant_for_lcs_context, *plant_context_autodiff, contact_pairs);

  // Publishes input std::vector<MatrixXd> as a lcmt_timestamped_saved_traj
  auto traj_source = builder.AddSystem<TrajToLcmSystem>(ic3_trajs);
  traj_source->set_name("traj_source");

  auto traj_publisher = builder.AddSystem(
      LcmPublisherSystem::Make<c3::lcmt_timestamped_saved_traj>(
          "iC3_TRAJECTORY", &lcm,
          TriggerTypeSet({TriggerType::kForced})));

  builder.Connect(traj_source->get_output_port(),
                    traj_publisher->get_input_port());

	// Eigen::Vector4d q_vec = xd.segment(5, 4);
	// Eigen::Quaterniond q(q_vec(0), q_vec(1), q_vec(2), q_vec(3));
	// q.normalize();
  // RotationMatrixd R_target(q);
	// RigidTransformd X_WF(R_target, xd.segment(9, 3));

  // Set up Meshcat visualizer.
  // auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  // drake::geometry::MeshcatVisualizerParams params;

  // auto& visualizer = drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
  //     &builder, scene_graph, meshcat, std::move(params));

  // drake::multibody::meshcat::ContactVisualizer<double>::AddToBuilder(
  //     &builder, plant, meshcat,
  //     drake::multibody::meshcat::ContactVisualizerParams());

	// const double axis_len = 0.2;
	// const double radius = 0.01;

	// meshcat->SetObject("target_pose/x_axis", drake::geometry::Cylinder(radius, axis_len), drake::geometry::Rgba(1, 0, 0, 1));
	// RigidTransformd X_FX(
	// 	RotationMatrixd::MakeYRotation(-M_PI / 2.0),
	// 	Eigen::Vector3d(axis_len / 2.0, 0, 0));
	// meshcat->SetTransform("target_pose/x_axis", X_WF * X_FX);

	// meshcat->SetObject("target_pose/y_axis", drake::geometry::Cylinder(radius, axis_len), drake::geometry::Rgba(0, 1, 0, 1));
	// RigidTransformd X_FY(
	// 	RotationMatrixd::MakeXRotation(M_PI / 2.0),
	// 	Eigen::Vector3d(0, axis_len / 2.0, 0));
	// meshcat->SetTransform("target_pose/y_axis", X_WF * X_FY);

	// meshcat->SetObject("target_pose/z_axis", drake::geometry::Cylinder(radius, axis_len), drake::geometry::Rgba(0, 0, 1, 1));
	// RigidTransformd X_FZ(
	// 	RotationMatrixd::Identity(),
	// 	Eigen::Vector3d(0, 0, axis_len / 2.0));
	// meshcat->SetTransform("target_pose/z_axis", X_WF * X_FZ);

  // Build the diagram.
  auto diagram = builder.Build();

  if (!FLAGS_diagram_path.empty())
    c3::systems::common::DrawAndSaveDiagramGraph(*diagram, FLAGS_diagram_path);

  // Create a default context for the diagram.
  auto diagram_context = diagram->CreateDefaultContext();

  // Set the initial state of the system.
  Eigen::VectorXd x0(23);
	x0 << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  // x0 << 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  std::vector<double> x_init = *options.x_init;
  x0 = Eigen::Map<Eigen::VectorXd>(x_init.data(), x_init.size());
	std::cout << "x0: " << x0.transpose() << std::endl;

	auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());
  plant.SetPositionsAndVelocities(&plant_context, x0);


  // for (const auto& x : ic3_traj) {
  //   plant.SetPositions(&plant_context, x.head(plant.num_positions()));
  //   visualizer.ExecuteForcedEvents(diagram_context.get(), true);    
  //   std::this_thread::sleep_for(
  //     std::chrono::duration<double>(options.lcs_factory_options.dt));
  // }

  // // Create and configure the simulator.
  // drake::systems::Simulator<double> simulator(*diagram,
  //                                             std::move(diagram_context));
  // simulator.set_target_realtime_rate(0.25);  // Run simulation at real-time speed.
  // simulator.Initialize();
  // simulator.AdvanceTo(120.0);  // Run simulation for 10 seconds.

  std::signal(SIGINT, SigIntHandler);
  auto output = diagram->AllocateOutput();

  const std::chrono::milliseconds period(1000);
  while (g_run.load()) {
    diagram->CalcOutput(*diagram_context, output.get()); 
    diagram->ForcedPublish(*diagram_context);
    std::this_thread::sleep_for(period);
  }


  return 0;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::lcm::DrakeLcm lcm(FLAGS_lcm_url);

  if (FLAGS_experiment_type == "cartpole_softwalls") {
    std::cout << "Running Cartpole Softwalls Test..." << std::endl;
    return RunCartpoleTest();
  } else if (FLAGS_experiment_type == "cube_pivoting") {
    std::cout << "Running Cube Pivoting Test..." << std::endl;
    return RunPivotingTest();
  } else if (FLAGS_experiment_type == "plate") {
    std::cout << "Running Plate Test..." << std::endl;
    return RunPlateTest();
  } else if (FLAGS_experiment_type == "iC3") {
    std::cout << "Running iC3 Test..." << std::endl;
    return RunPlateTestiC3(lcm);
  } else {
    std::cerr
        << "Unknown experiment type: " << FLAGS_experiment_type
        << ". Supported types are 'cartpole_softwalls' and 'cube_pivoting'."
        << std::endl;
    return -1;
  }
}
