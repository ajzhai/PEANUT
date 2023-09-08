import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='PEANUT')

    # General arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--sem_gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--start_ep', type=int, default=0,
                        help='start episode for data collection')
    parser.add_argument('--end_ep', type=int, default=-1,
                        help='end episode for data collection')

    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""1: Show visualization on screen
                                2: Dump visualizations as image files
                                (default: 0)""")
    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='experiment name (default: exp1)')
    parser.add_argument('-d', '--dump_location', type=str, default="./data/tmp/",
                        help='path to dump models and log (default: ./data/tmp/)')
    
    # Segmentation model
    parser.add_argument('--seg_model_wts', type=str, default='nav/agent/utils/mask_rcnn_R_101_cat9.pth',
                        help='path to segmentation model')
    parser.add_argument('--segformer_ckpt', type=str, default='/workspaces/peanut-temp/nav/agent/utils/HM3D Finetuned SegFormer - ObjectGoalNav - sqrt weights - Shuffled - full finetune/checkpoint-62216',
            help='path to segformer segmentation model checkpoint')
    parser.add_argument('--seg_type', type=str, default='Mask-RCNN',
        help='Which semantic segmentation model to use - valid choices are [Mask-RCNN,Segformer]')
    parser.add_argument('--fusion_type', type=str, default='Averaging',
        help='Which semantic fusion method to use - valid choices are [Averaging,Bayesian,Geometric,Histogram]')
    # Prediction model
    parser.add_argument('--pred_model_wts', type=str, default="./nav/pred_model_wts.pth",
                        help='path to prediction model weights')
    parser.add_argument('--pred_model_cfg', type=str, default="./nav/pred_model_cfg.py",
                        help='path to prediction model config')
    parser.add_argument('--prediction_window', type=int, default=720,
                        help='size of prediction (in pixels)')
    parser.add_argument('--mapping_strategy', type=str, default="neural",
                        help='The type of mapping strategy to use, valid = [neural,traditional], neural is the original PONI one')
    
    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=640,
                        help='Frame width (default:640)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=480,
                        help='Frame height (default:480)')
    parser.add_argument('-fw', '--frame_width', type=int, default=640,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=480,
                        help='Frame height (default:120)')
    parser.add_argument('-el', '--max_episode_length', type=int, default=500,
                        help="""Maximum episode length""")
    parser.add_argument("--task_config", type=str,
                        default="tasks/objectnav_gibson.yaml",
                        help="path to config yaml containing task information")

    parser.add_argument('--camera_height', type=float, default=0.88,
                        help="agent camera height in metres")
    parser.add_argument('--hfov', type=float, default=79.0,
                        help="horizontal field of view in degrees")
    parser.add_argument('--turn_angle', type=float, default=30,
                        help="Agent turn angle in degrees")
    parser.add_argument('--min_depth', type=float, default=0.5,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=5.0,
                        help="Maximum depth for depth sensor in meters")


    parser.add_argument('--num_local_steps', type=int, default=20,
                        help="""Number of steps between local map position updates""")

    # Mapping
    parser.add_argument('--num_sem_categories', type=int, default=10)
    parser.add_argument('--sem_pred_prob_thr', type=float, default=0.95)
    parser.add_argument('--goal_thr', type=float, default=0.985)
    parser.add_argument('--global_downscaling', type=int, default=2)  # local map relative size
    parser.add_argument('--vision_range', type=int, default=100)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=1)
    parser.add_argument('--map_size_cm', type=int, default=4800)  # the global downscaling may also need to change
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=0.1)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    parser.add_argument('--map_trad_detection_threshold', type=float, default=0.55)
    parser.add_argument('--col_rad', type = float, default = 4) 
    parser.add_argument('--goal_erode', type = int, default = 3) 
    parser.add_argument('--collision_threshold', type=float, default=0.20)
    parser.add_argument(
        "--evaluation", type=str, required=False, choices=["local", "remote"]
    )
    
    # Small details from Stubborn
    parser.add_argument('--timestep_limit', type = int, default=499)
    parser.add_argument('--grid_resolution',type = int, default = 24)
    parser.add_argument('--magnify_goal_when_hard',type = int, default = 100) #originally 100
    parser.add_argument("--move_forward_after_stop",type = int, default = 1) #originally 1

    # Long-term goal selection
    parser.add_argument('--dist_weight_temperature', type = float, default = 500,
                        help="Temperature for exponential distance weight (lambda in paper)")
    parser.add_argument('--goal_reached_dist', type = float, default = 75,
                        help="Distance at which goal is considered reached")
    parser.add_argument('--update_goal_freq', type = float, default = 10,
                        help="How often to update long-term goal")
    parser.add_argument('--switch_step', type = float, default = 0,
                        help="For switching from Stubborn goal selection to PEANUT")
    
    # For data collection 
    parser.add_argument('--use_gt_seg', type = int, default = 0)
    parser.add_argument('--only_explore', type = int, default = 0)
    parser.add_argument("--perf_log_name", type=str,
                        default="log.csv",
                        help="where to log the performance of the runs")    # parse arguments
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
