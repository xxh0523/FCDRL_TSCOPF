import numpy as np
import torch
import argparse
import pathlib

from utils import ReplayBuffer, Logger, Picture_Drawer
from rl_policy import Agent_DDPG

from gym_psops.envs import sopf_optimum_Env, worker_sopf_optimum
import ray
import time
from tqdm import tqdm

neg_mid = -500.
neg_max = -1000.


# this code is developed based on https://github.com/sfujim/TD3/blob/master/main.py 


def eval_policy(logger: Logger, workers, policy, cur_episode=None, eval_episodes=10, state_pos=""):
    num_workers = len(workers)
    
    avg_reward = 0.
    reward = list()
    static_violation = 0
    dynamic_violation = 0
    vio_list = list()
    
    if state_pos != "": saved_states = np.load(state_pos, allow_pickle=True)

    for k in tqdm(range(eval_episodes)):
        if (cur_episode is None):
            if state_pos == "": states = ray.get([worker.set_insecure_start.remote() for worker in workers])
            else: states = ray.get([workers[w_no].set_custom_start.remote(saved_states[k*num_workers+w_no][0], 
                                                                          saved_states[k*num_workers+w_no][1], 
                                                                          saved_states[k*num_workers+w_no][2], 
                                                                          saved_states[k*num_workers+w_no][3]) for w_no in range(num_workers)])
            violations = np.array(ray.get([worker.check_dynamic_constraints.remote() for worker in workers]))
            vio_list.append(violations)
            dynamic_violation += np.where(violations > 0.0)[0].shape[0]
            violations = np.array(ray.get([worker.check_static_constraints.remote() for worker in workers]))
            static_violation += np.where(violations > 0.0)[0].shape[0]
        else:
            states = ray.get([worker.set_random_start.remote() for worker in workers])
        actions = [policy.select_action(np.array(state), True) for state in states]
        results = ray.get([worker.step.remote(action) for worker, action in zip(workers, actions)])
        for i in range(len(workers)):
            reward.append(results[i][1])
    avg_reward = sum(reward) / eval_episodes / len(workers)

    logger.testing_log('======================================================================================')
    if cur_episode is None: 
        reward = np.array(reward)
        logger.testing_log(f'Dynamic stability before: {vio_list}')
        logger.testing_log(f'All reward: {reward}')
        logger.testing_log(f'Total scenarios: {reward.shape[0]}, static violation: {static_violation}, dynamic violation: {dynamic_violation}.')
        logger.testing_log(f'Average reward: {avg_reward:.10f}.')
        logger.testing_log(f'Control success: {float(np.where(reward >= 0)[0].shape[0]) / float(reward.shape[0]) * 100.}%.')
        logger.testing_log(f'Static instability: {float(np.where(reward[reward < 0] >= (neg_mid+1))[0].shape[0]) / float(reward.shape[0]) * 100.}%.')
        logger.testing_log(f'Dynamic instability: {float(np.where(reward[reward < (neg_mid+1)] >= (neg_max+1))[0].shape[0]) / float(reward.shape[0]) * 100.}%.')
        logger.testing_log(f'Not converge: {float(np.where(reward < (neg_max+1))[0].shape[0]) / float(reward.shape[0]) * 100.}%.')
        reward = reward.tolist()
    else: logger.testing_log(f"Training Episode: {cur_episode} Evaluation over {eval_episodes} episodes, average reward: {avg_reward:.10f}, all reward: {reward}")
    logger.testing_log('======================================================================================')
    
    return reward


def compare_policy(logger: Logger,env, workers, policy, eval_episodes=10, state_pos=""):
    num_workers = len(workers)
    reward = list()
    pso_solution = list()
    t_rl_decision = 0.0
    t_rl_checking = 0.0
    t_pso = 0.0
    static_violation = 0
    dynamic_violation = 0
    static_after_rl = 0
    dynamic_after_rl = 0
    unconv_after_rl = 0
    static_after_pso = 0
    dynamic_after_pso = 0
    unconv_after_pso = 0
    if state_pos != "": saved_states = np.load(state_pos, allow_pickle=True)
    for k in tqdm(range(eval_episodes)):
        # get state
        if state_pos == "": states = ray.get([worker.set_insecure_start.remote() for worker in workers])
        else: states = ray.get([workers[w_no].set_custom_start.remote(saved_states[k*num_workers+w_no][0], 
                                                                      saved_states[k*num_workers+w_no][1], 
                                                                      saved_states[k*num_workers+w_no][2], 
                                                                      saved_states[k*num_workers+w_no][3]) for w_no in range(num_workers)])
        
        # violation check
        violations = np.array(ray.get([worker.check_static_constraints.remote() for worker in workers]))
        static_violation += np.where(violations > 0.0)[0].shape[0]
        violations = np.array(ray.get([worker.check_dynamic_constraints.remote() for worker in workers]))
        dynamic_violation += np.where(violations > 0.0)[0].shape[0]
        
        # rl
        t_rl_decision -= time.time()
        actions = [policy.select_action(np.array(state)) for state in states]
        t_rl_decision += time.time()
        t_rl_checking -= time.time()
        results = ray.get([worker.step.remote(action) for worker, action in zip(workers, actions)])
        t_rl_checking += time.time()
        for i in range(len(workers)): 
            reward.append(results[i][1])
            if results[i][1] == neg_max: unconv_after_rl += 1
            elif results[i][1] < (neg_mid+1): dynamic_after_rl += 1
            elif results[i][1] < 0: static_after_rl += 1
        
        # pso
        ray.get([worker.reset.remote() for worker in workers])
        t_pso -= time.time()
        stats_pso = ray.get([worker.cal_optimum.remote() for worker in workers])
        t_pso += time.time()
        for i in range(len(workers)): 
            pso_solution.append(stats_pso[i][1][0])
            if stats_pso[i][1][0] == neg_max: unconv_after_pso += 1
            elif stats_pso[i][1][0] < (neg_mid+1): dynamic_after_pso += 1
            elif stats_pso[i][1][0] < 0: static_after_pso += 1

    logger.testing_log("---------------------------------------")
    if len(reward) != 0: logger.testing_log(f'rl reward: {reward}')
    if len(pso_solution) != 0: logger.testing_log(f'pso reward: {pso_solution}')
    logger.testing_log("---------------------------------------")
    logger.testing_log(f'static violation {static_violation}, dynamic violation {dynamic_violation}')
    if len(reward) != 0: logger.testing_log(f'rl: average reward: {sum(reward) / len(reward):.10f}, static after {static_after_rl}, dynamic after {dynamic_after_rl}, unconv after: {unconv_after_rl}')
    if len(pso_solution) != 0: logger.testing_log(f'pso average reward: {sum(pso_solution) / len(pso_solution):.10f}, static after {static_after_pso}, dynamic after {dynamic_after_pso}, unconv after: {unconv_after_pso}')
    logger.testing_log("---------------------------------------")
    logger.testing_log(f'rl cal time: {t_rl_decision}, rl check time: {t_rl_checking}, pso cal time: {t_pso}')
    logger.testing_log("---------------------------------------")
    
    return reward, pso_solution


if __name__ == "__main__":
    t1 = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str,
                        help='Environment type.',
                        required=False, default='optimum')
    parser.add_argument('--sampler', type=str,
                        help='Power flow sampler.',
                        required=False, default='stepwise')
    parser.add_argument('--static', type=str,
                        help='Static check type.',
                        required=False, default='all')
    parser.add_argument('--observation', type=str,
                        help='Observation type.',
                        required=False, default='minimum')
    parser.add_argument('--action', type=str,
                        help='Action type.',
                        required=False, default='absolute')
    parser.add_argument('--reward', type=str,
                        help='Reward type.',
                        required=False, default='unstabletime')
    parser.add_argument('--volcheck', type=str,
                        help='Flag for checking voltage of power sample.',
                        required=False, default='true')
    parser.add_argument('--criterion', type=float,
                        help='Rotor angle criterion. Default is 180.0.',
                        required=False, default=180.0)
    parser.add_argument('--coreward', type=float,
                        help='Reward coefficient. Default is 2000.0.',
                        required=False, default=2000.0)

    parser.add_argument('--training', action='store_true',
                        help='Call training process.',
                        required=False)
    parser.add_argument('--testing', action='store_true',
                        help='Call testing process.',
                        required=False)
    parser.add_argument('--comparing', action='store_true',
                        help='Call compare process.',
                        required=False)
    parser.add_argument('--pres', type=str,
                        help='Prepared states path, if given, file must exists for evaluation.',
                        required=False, default="")

    parser.add_argument('--model', type=str,
                        help='Model dump/load path, directory can be automatically created, but model file must exists for evaluation.',
                        required=False, default='00saved_results/models/scopf_agent/test')

    parser.add_argument('--insecure', type=str,
                        help='Flag for insecure training settings.',
                        required=False, default='false')
    parser.add_argument('--policy', type=str,
                        help='Policy type: ddpg, ddpg, ... Default is ddpg',
                        required=False, default='ddpg')
    parser.add_argument('--worker', type=int,
                        help='Set the number of workers. Default is 1',
                        required=False, default=1)
    parser.add_argument('--seed', type=int,
                        help='Set random seed for the environment. Default is 42',
                        required=False, default=42)
    parser.add_argument('--warm', type=int,
                        help='Set warm-up traing steps. Default is 0',
                        required=False, default=0)
    parser.add_argument('--eval', type=int,
                        help='Set evaluation period (steps). Default is 100',
                        required=False, default=100)
    parser.add_argument('--round', type=int,
                        help='Set the number of episodes tested during the evaluation process. Default is 10',
                        required=False, default=100)
    parser.add_argument('--max', type=int,
                        help='Set max training steps. Default is 20000',
                        required=False, default=20000)
    parser.add_argument('--batch', type=int,
                        help='Set mini-batch size for training actor and critic. Default is 256',
                        required=False, default=256)
    parser.add_argument('--explore', type=float,
                        help='Set the std error of the Gaussian exploration noise. \
                            When equal to 1.0, it will start at 1.0 and damping to 0.01. When smaller than 1.0, the exploration noise is constant. Default is 1.0.',
                        required=False, default=1.0)
    parser.add_argument('--discount', type=float,
                        help='Set discount factor. The discount factor is useless in this program. Default is 0.0',
                        required=False, default=0.0)

    parser.add_argument('--hidden', type=int,
                        help='Set hidden dimentionality. Default value is 256.',
                        required=False, default=256)
    parser.add_argument('--actor', type=float,
                        help='Set the actor learning rate. Default is 1e-4',
                        required=False, default=1e-4)
    parser.add_argument('--critic', type=float,
                        help='Set the critic learning rate. Default is 1e-3',
                        required=False, default=1e-3)
    parser.add_argument('--tau', type=float,
                        help='Set target network update rate. Default is 0.05',
                        required=False, default=0.05)
    parser.add_argument('--pnoise', type=float,
                        help='Set the std error of the Gaussian noise added to target policy during critic update. Default is 0.1',
                        required=False, default=0.1)
    parser.add_argument('--clip', type=float,
                        help='Set the range to clip target policy noise. Default is 0.2',
                        required=False, default=0.2)
    parser.add_argument('--freq', type=int,
                        help='Set frequency of delayed policy updates. Default is 2',
                        required=False, default=2)
    parser.add_argument('--update', type=int,
                        help='Set the number of training epochs during policy update. Default is 1',
                        required=False, default=1)
    args = parser.parse_args()
    
    flg_check_voltage = True if args.volcheck == 'true' else False
    if args.env == 'optimum': 
        env = sopf_optimum_Env(
            flg=0, 
            sampler=args.sampler, 
            static_check=args.static, 
            observation_type=args.observation,
            action_type=args.action,
            reward_type=args.reward,
            check_voltage=flg_check_voltage,
            criterion=args.criterion,
            co_reward=args.coreward
        )
        Cur_Worker = worker_sopf_optimum
    else:
        raise Exception('Unknown environment.')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden,
        "discount": args.discount,
        "actor_lr": args.actor,
        "critic_lr": args.critic
    }

    if args.policy == "ddpg":
        kwargs["tau"] = args.tau
        kwargs["total_step"] = int(args.max * args.update)
        policy = Agent_DDPG(**kwargs)
    else:
        raise Exception('Unknown policy type. Please check using --help.')

    ray.init(num_cpus=args.worker, ignore_reinit_error=True)
    workers = [Cur_Worker.remote(
        flg=i, 
        sampler=args.sampler, 
        static_check=args.static, 
        observation_type=args.observation,
        action_type=args.action,
        reward_type=args.reward,
        check_voltage=flg_check_voltage,
        criterion=args.criterion,
        co_reward=args.coreward
    ) for i in range(args.worker)]

    ray.get([worker.seed.remote(64*i+args.seed) for worker, i in zip(workers, range(len(workers)))])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.training:
        model_path = pathlib.Path(args.model)

        if not model_path.exists(): model_path.mkdir()
        if model_path.is_dir() is False:
            assert model_path.exists(), f'{model_path} does not exist!'
            policy.load(model_path)
            model_path = model_path.parent / (model_path.name + '_branch')
            if not model_path.exists(): model_path.mkdir()
        
        logger = Logger(model_path, 'training.log', 'testing.log')
        logger.training_log(f'Environment: {args.env}, Sampler: {args.sampler}, Static_check: {args.static}, Observation_type: {args.observation}, Action_type: {args.action}, Reward_type: {args.reward}, Voltage_check: {args.volcheck}, Rotor_angle_criterion: {args.criterion}, Reward_coefficient: {args.coreward}')
        for fault in env.anticipated_fault_set: logger.training_log(f'Line {fault}: {env.psops.get_acline_info(fault[0])}')
        logger.training_log(f'agent_model_path: {args.model}, insecure_training: {args.insecure}, policy_type: {args.policy}, worker_num: {args.worker}, random_seed: {args.seed}, warm_up_step: {args.warm}, eval_period: {args.eval},' +  
                            f'episodes_for_evaluation: {args.round}, max_time_step: {args.max}, mini_batch_size: {args.batch}, explore_noise: {args.explore}, discount_factor: {args.discount}, training_epoch_each_episode: {args.update}')
        logger.training_log(f'model_parameters: {kwargs}')

        replay_buffer = ReplayBuffer(state_dim, action_dim)

        evaluations = [eval_policy(logger=logger, workers=workers, policy=policy, eval_episodes=args.round)]
        
        expl_noise = args.explore
        if expl_noise == 1.0:
            minimum_noise = 0.1
            step_noise = expl_noise / float(args.max)
        else:
            minimum_noise = expl_noise
            step_noise = 0.0

        states = ray.get([worker.set_random_start.remote() for worker in workers])
        done = [False] * len(workers)
        episode_timesteps = 0
        total_rewards = list()

        for t in tqdm(range(1, int(args.max)+1)):

            if args.insecure == "true" or args.insecure == "True": states = ray.get([worker.set_insecure_start.remote() for worker in workers])
            else: states = ray.get([worker.set_random_start.remote() for worker in workers])

            if t < args.warm and t % 2 == 1:
                actions = ray.get([worker.get_random_action.remote() for worker in workers])
            else:
                actions = [(policy.select_action(np.array(state)) + np.random.normal(0, expl_noise, size=action_dim)).clip(-1.0, 1.0) for state in states]
            
            results = ray.get([worker.step.remote(action) for worker, action in zip(workers, actions)])
            
            rewards = 0.
            for state, action, result in zip(states, actions, results):
                done_bool = True
                replay_buffer.add(state, action, result[0], result[1], done_bool)
                rewards += result[1]
                total_rewards.append(result[1])
            rewards /= len(workers)

            policy.train(replay_buffer, args.batch, args.update)

            if t % args.eval == 0:
                evaluations.append(eval_policy(logger=logger, workers=workers, policy=policy, cur_episode=t+1, eval_episodes=args.round))
                policy.save(model_path / 'model.pth')
                np.savez(model_path / 'train_and_eval.npz', train=total_rewards, eval=evaluations)
                Picture_Drawer.draw_sopf_train_test(data_path=model_path / 'train_and_eval.npz', n_processor=len(workers), total_step=t, interval=args.eval)
                if t % 10000 == 0: policy.save(model_path / f'model.checkpoint_{t}.pth')
            
            expl_noise -= step_noise
            expl_noise = minimum_noise if expl_noise < minimum_noise else expl_noise
        
        evaluations.append(eval_policy(logger=logger, workers=workers, policy=policy, eval_episodes=args.round))
        policy.save(model_path / f'final_model.pth')
        np.savez(model_path / f'train_and_eval.npz', train=total_rewards, eval=evaluations)

        logger.training_log(f'Total calculation time: {time.time() - t1} seconds.')

    elif args.testing:
        model_path = pathlib.Path(args.model)
        assert model_path.is_dir() is False and model_path.exists(), f'{model_path} is not a file or does not exist!'
        policy.load(model_path)
        
        logger = Logger(logfile_path=model_path.parent, test_log_name=f'{model_path.name}_evaluation.log')
        for fault in env.anticipated_fault_set: logger.testing_log(f'Line {fault}: {env.psops.get_acline_info(fault[0])}')
        logger.testing_log("load policy successful")

        results = eval_policy(logger=logger, workers=workers, policy=policy, eval_episodes=args.round, state_pos=args.pres)
        logger.testing_log(f'Total calculation time: {time.time() - t1} seconds.')

        np.save(model_path.parent / f'{model_path.name}_reward_eval.npy', np.array(results))

    elif args.comparing:
        model_path = pathlib.Path(args.model)
        assert model_path.is_dir() is False and model_path.exists(), f'{model_path} is not a file or does not exist!'
        policy.load(model_path)
        
        logger = Logger(logfile_path=model_path.parent, test_log_name=f'{model_path.name}_comparison.log')
        for fault in env.anticipated_fault_set: logger.testing_log(f'Line {fault}: {env.psops.get_acline_info(fault[0])}')
        logger.testing_log("load policy successful")

        r, p = compare_policy(logger=logger, env=env, workers=workers, policy=policy, eval_episodes=args.round, state_pos=args.pres)

        np.savez(model_path.parent / f'eval_compare_{model_path.name}.npz', rl = r, pso = p)
    else:
        raise Exception('Unknown task. Set "--training" or "--testing".')

