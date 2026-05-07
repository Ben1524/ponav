from crowd_sim_plus.envs.policy.policy_factory import policy_factory
from sicnav.policy.dwa import DynamicWindowApproach
from sicnav.policy.campc import CollisionAvoidMPC
from sicnav.policy.NewMPC import NewMPC
from sicnav.policy.NewMPCChanging import NewMPCChanging
from sicnav.policy.navrl import PONav
from sicnav.policy.sarl import SARLNav
# from sicnav.policy.sicnav_acados import SICNavAcados
from sicnav.policy.cnn_policy import CNNRL
from sicnav.policy.dsrnn import DSRNNPolicy
from sicnav.policy.selfattn_merge_srnn import SelfAttnMergeSRNNPolicy
from sicnav.policy.selfattn_merge_srnn_lidar import SelfAttnMergeSRNNLidarPolicy
from sicnav.policy.homo_transformer_obs import HomoTransformerObsPolicy
from sicnav.policy.lidar_gru import LIDARGRUPolicy
from sicnav.policy.om_gru import OMGRUPolicy

policy_factory['dwa'] = DynamicWindowApproach
policy_factory['campc'] = CollisionAvoidMPC
policy_factory['NewMPC']= NewMPC
policy_factory['NewMPCChanging']= NewMPCChanging
policy_factory['ponav'] = PONav
policy_factory['sarl'] = SARLNav
# policy_factory['sicnav_acados'] = SICNavAcados
policy_factory['cnn_policy'] = CNNRL
policy_factory['dsrnn_obs_vertex'] = DSRNNPolicy
policy_factory['selfAttn_merge_srnn'] = SelfAttnMergeSRNNPolicy
policy_factory['selfAttn_merge_srnn_lidar'] = SelfAttnMergeSRNNLidarPolicy
policy_factory['homo_transformer_obs'] = HomoTransformerObsPolicy
policy_factory['lidar_gru'] = LIDARGRUPolicy
policy_factory['om_gru'] = OMGRUPolicy