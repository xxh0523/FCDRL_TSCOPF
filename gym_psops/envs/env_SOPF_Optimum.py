
from gym import spaces, core
from gym_psops.envs.psops import Py_PSOPS
import numpy as np
import ray
import time
from optimizer import My_PSO
from timebudget import timebudget

# core.Env是gym的环境基类,自定义的环境就是根据自己的需要重写其中的方法；
# 必须要重写的方法有:
# __init__()：构造函数
# reset()：初始化环境
# step()：环境动作,即环境对agent的反馈
# render()：如果要进行可视化则实现

class sopf_optimum_Env(core.Env):
    def __init__(
            self,
            flg=0, 
            rng=None,
            sys_no=-1,
            act_gen_v=None,
            act_gen_p=None,
            sampler='stepwise',
            static_check='all',
            observation_type = 'minimum', # minimum state, all state
            action_type = 'absolute', # absolute action, or delta action
            reward_type = 'unstabletime', # maximum rotor angle, TSI, unstable duration
            check_voltage=True,
            check_slack=True,
            upper_load=1.2,
            lower_load=0.7,
            criterion=180.,
            fault_set=[[26, 0, 0.1], [26, 100, 0.1]],
            disturbance_set=[],
            co_reward=1000.
    ):
        self.co_static_violation = -100.
        self.neg_mid = -500.
        self.neg_max = -1000.
        # settings
        self.sampler = sampler
        assert (self.sampler == 'stepwise' or self.sampler == 'simple'), f'Unknown sampler type: {self.sampler}. Please check. '
        self.static_check = static_check
        assert (self.static_check == 'all' or self.static_check == 'important' or self.static_check == 'none'), f'Unknown static check type: {self.static_check}. Please check.'
        self.observation_type = observation_type
        assert (self.observation_type == 'minimum' or self.observation_type == 'all'), f'Unknown observation type: {self.observation_type}. Please check.'
        self.action_type = action_type
        assert (self.action_type == 'absolute' or self.action_type == 'delta'), f'Unknown action type: {self.action_type}. Please check.'
        self.reward_type = reward_type
        assert (self.reward_type == 'unstabletime' or self.reward_type == 'maxrotordiff' or self.reward_type == 'tsi'), f'Unknown reward type: {self.reward_type}. Please check.'
        self.check_voltage = check_voltage
        self.check_slack = check_slack
        self.criterion = criterion
        self.co_reward = co_reward
        # cost coefficients
        self.ck1 = 0.2
        self.ck2 = 30.
        self.ck3 = 100.
        # flg and system numer
        self.flg = flg
        self.sys = sys_no
        # psops api
        self.psops = Py_PSOPS(flg=self.flg, rng=rng)
        api = self.psops
        # load limit
        self.load_lower_limit = lower_load
        self.load_upper_limit = upper_load
        # anticipated contingency set
        if api.get_bus_number() == 39:
            fault_set=[[26, 0, 0.1], [26, 0, 0.1]]
        elif api.get_bus_number() == 710:
            fault_set=[[100, 0, 0.1], [100, 100, 0.1]]
        self.anticipated_fault_set = fault_set
        self.anticipated_disturbance_set = disturbance_set
        # dynamic constraints
        if self.reward_type == 'unstabletime':
            self.c_fault = (self.neg_mid-self.neg_max) / (len(self.anticipated_fault_set) + len(self.anticipated_disturbance_set)) / ((api.get_info_ts_max_step() - 1) * api.get_info_ts_delta_t())
        elif self.reward_type == 'maxrotordiff':
            self.c_fault = (self.neg_mid-self.neg_max) / (len(self.anticipated_fault_set) + len(self.anticipated_disturbance_set)) / 500.
        elif self.reward_type == 'tsi':
            self.c_fault = (self.neg_mid-self.neg_max) / (len(self.anticipated_fault_set) + len(self.anticipated_disturbance_set))
        if self.sampler == 'stepwise': self.pf_sampler = api.get_power_flow_sample_stepwise
        elif self.sampler == 'simple': self.pf_sampler = api.get_power_flow_sample_simple_random
        # print(self.c_fault)
        # random generator
        self.set_random_state(np.random.default_rng() if rng is None else rng)
        # save original state
        self.acline_connectivity = api.get_network_acline_all_connectivity()
        self.generator_connectivity = api.get_network_generator_all_connectivity()
        self.generator_v_set = api.get_generator_all_v_set()
        self.generator_p_set = api.get_generator_all_p_set()
        self.load_p_set = api.get_load_all_p_set()
        self.load_q_set = api.get_load_all_q_set()
        # state, V, theta, PG, QG, Pl, Ql
        self.RADIAN = 180.0 / 3.1415926535897932384626433832795
        ob_bus_vmax = api.get_bus_all_vmax()
        ob_bus_vmin = api.get_bus_all_vmin()
        ob_generator_pmax = api.get_generator_all_pmax()
        ob_generator_pmin = api.get_generator_all_pmin()
        ob_generator_qmax = api.get_generator_all_qmax()
        ob_generator_qmin = api.get_generator_all_qmin()
        ob_load_pmax = api.get_load_all_p_set() * self.load_upper_limit
        ob_load_pmin = api.get_load_all_p_set() * self.load_lower_limit
        ob_load_qmax = api.get_load_all_q_set() * self.load_upper_limit
        ob_load_qmin = api.get_load_all_q_set() * self.load_lower_limit
        lower = np.concatenate((ob_bus_vmin, ob_generator_pmin, ob_generator_qmin, ob_load_pmin, ob_load_qmin))
        upper = np.concatenate((ob_bus_vmax, ob_generator_pmax, ob_generator_qmax, ob_load_pmax, ob_load_qmax))
        idx = lower > upper
        lower[idx], upper[idx] = upper[idx], lower[idx]
        assert True not in (upper < lower), 'State upper is smaller than lower, please check'
        self.state_space = spaces.Box(low=lower, high=upper)
        self.state = self.state_space.sample()
        self.bus_no_gen_load = np.unique(np.sort(np.concatenate([api.get_load_all_bus_no(), api.get_generator_all_bus_no()])))
        self.important_state_idx = np.concatenate([self.bus_no_gen_load, np.arange(ob_bus_vmin.shape[0], lower.shape[0])])
        # observation, Pl, Ql
        if self.observation_type == 'minimum':
            self.observation_space = spaces.Box(low=lower[-self.psops.get_load_number()*2:], high=upper[-self.psops.get_load_number()*2:])
        elif self.observation_type == 'all':
            self.observation_space = spaces.Box(low=lower, high=upper)
        # action, gen_V, gen_P
        self.ctrl_v_gen = np.arange(api.get_generator_number()) if act_gen_v is None else act_gen_v
        self.ctrl_p_gen = api.get_generator_all_ctrl() if act_gen_p is None else act_gen_p
        act_generator_vmax = api.get_generator_all_vmax(self.ctrl_v_gen, self.sys)
        act_generator_vmin = api.get_generator_all_vmin(self.ctrl_v_gen, self.sys)
        act_generator_pmax = api.get_generator_all_pmax()[self.ctrl_p_gen]
        act_generator_pmin = api.get_generator_all_pmin()[self.ctrl_p_gen]
        lower = np.concatenate((act_generator_vmin, act_generator_pmin))
        upper = np.concatenate((act_generator_vmax, act_generator_pmax))
        idx = lower > upper
        lower[idx], upper[idx] = upper[idx], lower[idx]
        assert True not in (upper < lower), 'action upper is smaller than lower, please check'
        self.centralCtrl = 0.5 * (upper + lower)
        self.deltaCtrl = 0.5 * (upper - lower)
        self.originCtrl = self.get_ctrl()
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.deltaCtrl.shape[0],))  # 动作空间
        # max cost
        g_max = api.get_generator_all_pmax()
        self.max_cost = sum(self.ck1 + self.ck2 * g_max + self.ck3 * g_max * g_max)
        # reset state
        self.reset()

    def set_flg(self, flg):
        self.flg = flg
    
    def set_random_state(self, rng):
        self.rng = rng
        self.psops.set_random_state(self.rng)

    def seed(self, sd):
        self.rng = np.random.default_rng(sd)
        self.psops.set_random_state(self.rng)

    def get_psops(self):
        return self.psops

    def set_back_to_origin_state(self):
        api = self.psops
        api.set_network_acline_all_connectivity(self.acline_connectivity)
        api.set_network_generator_all_connectivity(self.generator_connectivity)
        api.set_generator_all_v_set(self.generator_v_set)
        api.set_generator_all_p_set(self.generator_p_set)
        api.set_load_all_p_set(self.load_p_set)
        api.set_load_all_q_set(self.load_q_set)

    def get_ctrl(self):
        api = self.psops
        return np.concatenate((api.get_generator_all_v_set(self.ctrl_v_gen), api.get_generator_all_p_set(self.ctrl_p_gen)))

    def sample_a_power_flow(self):
        self.pf_sampler(num=1,
                        generator_v_list=self.ctrl_v_gen,
                        generator_p_list=self.ctrl_p_gen,
                        load_max=self.load_upper_limit,
                        load_min=self.load_lower_limit,
                        check_voltage=self.check_voltage,
                        check_slack=self.check_slack
                        )

    def set_random_start(self):
        self.sample_a_power_flow()
        obs, _ = self._get_observation()
        self.originCtrl = self.get_ctrl()
        return obs
    
    def set_custom_start(
        self,
        generator_v_set,
        generator_p_set,
        load_p_set,
        load_q_set,
        acline_connectivity=None,
        generator_connectivity=None
        ):
        api = self.psops
        if acline_connectivity is None:
            api.set_network_acline_all_connectivity(self.acline_connectivity)
        else: api.set_network_acline_all_connectivity(acline_connectivity)
        if generator_connectivity is None:
            api.set_network_generator_all_connectivity(self.generator_connectivity)
        else: api.set_network_generator_all_connectivity(generator_connectivity)
        api.set_generator_all_v_set(generator_v_set, self.ctrl_v_gen)
        api.set_generator_all_p_set(generator_p_set, self.ctrl_p_gen)
        api.set_load_all_p_set(load_p_set)
        api.set_load_all_q_set(load_q_set)
        obs, _ = self._get_observation()
        self.originCtrl = self.get_ctrl()
        return obs

    def set_insecure_start(self):
        while (1): 
            self.sample_a_power_flow()
            if self.check_dynamic_constraints() > 0: break
        obs, _ = self._get_observation()
        self.originCtrl = self.get_ctrl()
        return obs

    def get_current_state_settings(self):
        api = self.psops
        st = list()
        st.append(api.get_generator_all_v_set(self.ctrl_v_gen)) 
        st.append(api.get_generator_all_p_set(self.ctrl_p_gen))
        st.append(api.get_load_all_p_set())
        st.append(api.get_load_all_q_set())
        return np.array(st, dtype=object)

    def reset(self):
        self.set_ctrl(self.originCtrl)
        obs, _ = self._get_observation()
        return obs

    def set_ctrl(self, ctrl):
        api = self.psops
        api.set_generator_all_v_set(ctrl[:self.ctrl_v_gen.shape[0]], self.ctrl_v_gen, self.sys)
        api.set_generator_all_p_set(ctrl[self.ctrl_v_gen.shape[0]:], self.ctrl_p_gen, self.sys)
    
    # get observation
    def _get_observation(self):
        api = self.psops
        converge = api.cal_power_flow_basic_nr()
        if converge > 0:
            bus_result = api.get_bus_all_lf_result()[:,0]
            gen_result = api.get_generator_all_lf_result().reshape(api.get_generator_number()*2, order='F')
            load_result = api.get_load_all_lf_result().reshape(api.get_load_number()*2, order='F')
            self.state = np.concatenate([bus_result, gen_result, load_result])
            obs = self.state[-self.psops.get_load_number()*2:]
        else:
            v = np.zeros(api.get_bus_number())
            gen_p = api.get_generator_all_p_set()
            gen_p[api.get_generator_all_slack()] = 0.0
            gen_q = np.zeros(api.get_generator_number())
            load_p = api.get_load_all_p_set()
            load_q = api.get_load_all_q_set()
            self.state = np.concatenate([v, gen_p, gen_q, load_p, load_q])
        if self.observation_type == 'minimum':
            obs = self.state[-self.psops.get_load_number()*2:]
        elif self.observation_type == 'all':
            obs = self.state
        return obs, converge

    def get_random_action(self):
        self.pf_sampler(num=1,
                        generator_v_list=self.ctrl_v_gen,
                        generator_p_list=self.ctrl_p_gen,
                        load_max=-1,
                        load_min=-1,
                        check_voltage=self.check_voltage,
                        check_slack=self.check_slack
                        )
        return (self.get_ctrl() - self.centralCtrl) / self.deltaCtrl

    def step(self, cur_action):
        act = cur_action * self.deltaCtrl + self.centralCtrl
        self.set_ctrl(act)
        obs, converge = self._get_observation()
        rew = self._get_reward(obs, converge)
        don = True
        inf = None
        return obs, rew, don, inf

    def cal_dynamic_criterion(self):
        api = self.psops
        stability_result = api.get_acsystem_all_ts_result()[0, :, 1]
        flg = False
        if np.any(stability_result == 0.0): # early termination
            stop_step = np.where(stability_result == 0)[0]
            if stop_step[0] < 50: flg = True
        if flg == True:
            if self.reward_type == 'unstabletime': criterion_value = (api.get_info_ts_max_step() - 1) * api.get_info_ts_delta_t()
            elif self.reward_type == 'maxrotordiff': criterion_value = 500.
            elif self.reward_type == 'tsi': criterion_value = 1.
        else:
            criterion_value = 0
            if self.reward_type == 'unstabletime':
                stability_result = np.where(stability_result > self.criterion)[0]
                if stability_result.shape[0] != 0: criterion_value += ((api.get_info_ts_max_step() - 1) - stability_result[0]) * api.get_info_ts_delta_t()
            elif self.reward_type == 'maxrotordiff':
                delta_max = min(self.criterion + 500., abs(stability_result).max())
                if delta_max > 0: criterion_value = delta_max - self.criterion
            elif self.reward_type == 'tsi':
                delta_max = min(999999.9, abs(stability_result).max())
                stability_result = (self.criterion - delta_max) / (self.criterion + delta_max)
                if stability_result < 0: criterion_value = 0 - stability_result
        return criterion_value

    def simulate_acline_fault(self, fault):
        api = self.psops
        acline_no = int(fault[0])
        terminal = int(fault[1])
        f_time = float(fault[2])
        assert acline_no < api.get_acline_number() and terminal in [0, 100] and f_time >= 0.0, f'acline fault set error, acline no {acline_no}, terminal {terminal}, f_time {f_time}'
        api.set_fault_disturbance_clear_all()
        api.set_fault_disturbance_add_acline(0, terminal, 0.0, f_time, acline_no)
        api.set_fault_disturbance_add_acline(1, terminal, f_time, 10.0, acline_no)
        api.cal_transient_stability_simulation_ti_sv()

    def simulate_disturbance(self, disturbance):
        api = self.psops
        dis_type = int(disturbance[0])
        dis_time = float(disturbance[1])
        dis_per = float(disturbance[2])
        ele_type = int(disturbance[3]) 
        ele_pos = int(disturbance[4])
        # assert , f'disturabnce set error, {disturbance}'
        api.set_fault_disturbance_clear_all()
        api.set_disturbance(dis_type, dis_time, dis_per, ele_type, ele_pos, False)
        api.cal_transient_stability_simulation_ti_sv()

    def check_dynamic_constraints(self):
        criterion_value = 0.0
        # check stability of anticipated fault set
        for fault in self.anticipated_fault_set:
            self.simulate_acline_fault(fault=fault)
            criterion_value += self.cal_dynamic_criterion()
        # check stability of anticipated disturbance set
        for disturbance in self.anticipated_disturbance_set:
            self.simulate_disturbance(disturbance=disturbance)
            criterion_value += self.cal_dynamic_criterion()
        return criterion_value

    def check_static_constraints(self, obs=None):
        if obs is None: obs, _ = self._get_observation()
        lower = self.state_space.low
        upper = self.state_space.high
        if self.static_check == 'all':
            limit = sum((self.state - upper)[self.state > upper]) + sum((lower - self.state)[self.state < lower])
        elif self.static_check == 'important':
            idx = self.important_state_idx
            limit = sum((self.state[idx] - upper[idx])[self.state[idx] > upper[idx]]) + sum((lower[idx] - self.state[idx])[self.state[idx] < lower[idx]])
        elif self.static_check == 'none':
            limit = 0.0
        else:
            limit = sum((self.state - upper)[self.state > upper]) + sum((lower - self.state)[self.state < lower])
        return limit

    def _get_reward(self, obs, converge):
        api = self.psops
        if converge < 0: # not converge
            re = self.neg_max
        else:
            # check stability
            finish_time = self.check_dynamic_constraints()
            if finish_time > 0: # dynamic constraint violation
                re = max(self.neg_mid+1-self.c_fault*finish_time, self.neg_max+1)
                # print(self.c_fault, finish_time, re)
            else:
                limit = self.check_static_constraints(obs=obs)
                if limit > 0: # static constraint violation
                    re = max(self.co_static_violation*limit, self.neg_mid+1)
                else: # secure state
                    g = self.state[api.get_bus_number():api.get_bus_number()+api.get_generator_number()]
                    # cost = sum(0.2 + 30. * g + 100. * g * g)
                    cost = sum(self.ck1 + self.ck2 * g + self.ck3 * g * g)
                    re = max(self.co_reward * (1 - cost / self.max_cost), 0)
                    # re = max(1000. - 0.01 * cost, 0.0)
                    # re = max(3200 - 0.0001 * cost, 0.0)
        return re

    def get_max_cost(self):
        return self.max_cost

    def cal_action(self, x):
        if x.ndim == 1:
            _, y, _, _ = self.step(x)
            y = -y
        elif x.ndim == 2:
            y = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                _, y[i], _, _ = self.step(x[i])
            y = -y
        else:
            raise Exception("wrong dimension")
        return y

    def cal_optimum(self):
        optimizer = My_PSO(func=self.cal_action, 
                           n_dim=self.action_space.shape[0], 
                           pop=200, 
                           max_iter=150, 
                           lb=self.action_space.low, 
                           ub=self.action_space.high, 
                           w=0.8, c1=0.5, c2=0.5,
                           func_transform=False
                           )
        optimizer.run()
        solutions = optimizer.gbest_x * self.deltaCtrl + self.centralCtrl
        result = -optimizer.gbest_y
        # if optimizer.gbest_y <= 0: result = (1. + optimizer.gbest_y / self.co_reward) * self.max_cost
        # else: result = optimizer.gbest_y
        return [solutions, result]


@ray.remote
class worker_sopf_optimum(sopf_optimum_Env):
    def __init__(
            self,
            flg=0, 
            rng=None,
            sys_no=-1,
            act_gen_v=None,
            act_gen_p=None,
            sampler='stepwise',
            static_check='all',
            observation_type = 'minimum', # minimum state, all state
            action_type = 'absolute', # absolute action, or delta action
            reward_type = 'unstabletime', # maximum rotor angle, TSI, unstable duration
            check_voltage=True,
            check_slack=True,
            upper_load=1.2,
            lower_load=0.7,
            criterion=180.,
            fault_set=[[26, 0, 0.1], [26, 100, 0.1]],
            disturbance_set=[],
            co_reward=1000.
    ):
        super().__init__(
            flg=flg, 
            rng=rng, 
            sys_no=sys_no, 
            act_gen_v=act_gen_v, 
            act_gen_p=act_gen_p, 
            sampler=sampler, 
            static_check=static_check, 
            check_voltage=check_voltage, 
            observation_type=observation_type,
            action_type=action_type,
            reward_type=reward_type,
            check_slack=check_slack,
            upper_load=upper_load,
            lower_load=lower_load,
            criterion=criterion,
            fault_set=fault_set,
            disturbance_set=disturbance_set,
            co_reward=co_reward
        )
        self.__worker_no=flg
    
    def get_work_no(self):
        return self.__worker_no


class pso_sopf_optimum():
    def __init__(
            self, 
            n_processor, 
            seed=0,
            rng=None,
            sys_no=-1,
            act_gen_v=None,
            act_gen_p=None,
            sampler='stepwise',
            static_check='all',
            observation_type = 'minimum', # minimum state, all state
            action_type = 'absolute', # absolute action, or delta action
            reward_type = 'unstabletime', # maximum rotor angle, TSI, unstable duration
            check_voltage=True,
            check_slack=True,
            upper_load=1.2,
            lower_load=0.7,
            criterion=180.,
            fault_set=[[26, 0, 0.1], [26, 100, 0.1]],
            disturbance_set=[],
            co_reward=1000.
    ):
        rng = np.random.default_rng(seed)
        self.env = sopf_optimum_Env(
            rng=rng, 
            sys_no=sys_no, 
            act_gen_v=act_gen_v, 
            act_gen_p=act_gen_p, 
            sampler=sampler, 
            static_check=static_check, 
            check_voltage=check_voltage, 
            observation_type=observation_type,
            action_type=action_type,
            reward_type=reward_type,
            check_slack=check_slack,
            upper_load=upper_load,
            lower_load=lower_load,
            criterion=criterion,
            fault_set=fault_set,
            disturbance_set=disturbance_set,
            co_reward=co_reward
        )
        self.workers = [worker_sopf_optimum.remote(
            flg=i, 
            rng=rng, 
            sys_no=sys_no, 
            act_gen_v=act_gen_v, 
            act_gen_p=act_gen_p, 
            sampler=sampler, 
            static_check=static_check, 
            check_voltage=check_voltage, 
            observation_type=observation_type,
            action_type=action_type,
            reward_type=reward_type,
            check_slack=check_slack,
            upper_load=upper_load,
            lower_load=lower_load,
            criterion=criterion,
            fault_set=fault_set,
            disturbance_set=disturbance_set,
            co_reward=co_reward
        ) for i in range(n_processor)]
    
    def set_start(self):
        env = self.env
        api = self.env.psops
        workers = self.workers
        env.set_insecure_start()
        gen_v_set = api.get_generator_all_v_set()[env.ctrl_v_gen]
        gen_p_set = api.get_generator_all_p_set()[env.ctrl_p_gen]
        load_p_set = api.get_load_all_p_set()
        load_q_set = api.get_load_all_q_set()
        obs = ray.get([worker.set_custom_start.remote(gen_v_set, gen_p_set, load_p_set, load_q_set) for worker in workers])
        return obs

    # @timebudget
    def cal_action(self, x):
        workers = self.workers
        if x.ndim == 1:
            _, y, _, _ = workers[0].cal_action(x)
        elif x.ndim == 2:
            n_act = x.shape[0]
            n_round = int(n_act / len(workers))
            mod = n_act % n_round
            all_y = list()
            for i in range(len(workers)):
                if i < mod: all_y.append(workers[i].cal_action.remote(x[(n_round+1)*i:(n_round+1)*(i+1)]))
                else: all_y.append(workers[i].cal_action.remote(x[n_round*i+mod:n_round*(i+1)+mod]))
            all_y = ray.get(all_y)
            y = np.concatenate(all_y, axis=-1)
        else:
            raise Exception("wrong dimension")
        return y

    def cal_optimum(self):
        optimizer = My_PSO(func=self.cal_action, 
                           n_dim=self.env.action_space.shape[0], 
                           pop=200, 
                           max_iter=150, 
                           lb=self.env.action_space.low, 
                           ub=self.env.action_space.high, 
                           w=0.8, c1=0.5, c2=0.5,
                           func_transform=False
                           )
        optimizer.run()
        print(self.env.step(optimizer.gbest_x))
        solutions = optimizer.gbest_x * self.env.deltaCtrl + self.env.centralCtrl
        result = -optimizer.gbest_y
        # result = (1. + optimizer.gbest_y / env.co_reward) * env.max_cost
        return [solutions, result]


if __name__ == '__main__':
    import time
    t1 = time.time()
    rng = np.random.default_rng(0)
    env = sopf_optimum_Env(rng=rng)
    env.cal_optimum()
