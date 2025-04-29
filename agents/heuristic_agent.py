import numpy as np


class Heuristic():

    def __init__(self, observation_space_shape, action_space_n, host_vector_count):
        self.observation_shape = observation_space_shape[0]
        self.action_n = action_space_n
        self.host_vector_count = host_vector_count
        self.host_vec_length = int(self.observation_shape / self.host_vector_count)
        self.prev_action = None


    def get_action_id(self, action_type, host_subnet):
        """Конвертирует тип действия и подсеть в ID действия."""
        base_id = {
            1: 0,   # Машина (1,0) → базовый 0
            2: 6,   # Машина (2,0) → базовый 6
            3: 12   # Машина (3,0) → базовый 12
        }[host_subnet]
        
        action_map = {
            "service_scan": 0,
            "os_scan": 1,
            "subnet_scan": 2,
            "process_scan": 3,
            "exploit": 4,
            "pe": 5
        }
        return base_id + action_map[action_type]



    def choose_action(self, state, info):
        state_data = self.parse_state(state)        
        temp = 0
        hosts = [h for h in state_data if h["subnet"] in {1,2,3}]

        action =  self.get_action_id("subnet_scan", hosts[0]["subnet"])
        
        # 1. Проверить PE для всех машин
        for host in sorted(hosts, key=lambda h: -h["value"]):
            if host["access"] == 1 and "tomcat" in host["processes"]:
                action = self.get_action_id("pe", host["subnet"])
                if action == self.prev_action:
                    continue
                else:
                    self.prev_action = action
                    return action
        
        # 2. Exploit доступных машин
        for host in sorted(hosts, key=lambda h: -h["value"]):
            if host["reachable"] and host["discovered"] and not host["compromised"] and "ssh" in host["services"]:
                action = self.get_action_id("exploit", host["subnet"])
                if action == self.prev_action:
                    continue
                else:
                    self.prev_action = action
                    return action
        
        
        # 3. Subnet Scan из скомпрометированных
        compromised_hosts = [h for h in hosts if h["compromised"]]
        for host in compromised_hosts:
            if not all(h["discovered"] for h in hosts):
                action = self.get_action_id("subnet_scan", host["subnet"])
                if action == self.prev_action:
                    continue
                else:
                    self.prev_action = action
                    return action
        
        return action

   
    def parse_state(self, state):

        data = []

        for subnet_id in range(self.host_vector_count):
            
            start_idx = subnet_id * self.host_vec_length
            host_vec = state[start_idx : start_idx + self.host_vec_length]

            host_data = {
                # One-hot подсети
                "subnet": np.argmax(host_vec[0:4]),  
                
                # One-hot хоста
                "host": 0 if host_vec[4] == 1 else -1,  
                
                # Основные флаги
                "compromised": bool(host_vec[5]),
                "reachable": bool(host_vec[6]),
                "discovered": bool(host_vec[7]),
                
                # Ценность
                "value": int(host_vec[8]),  
                
                # Новизна обнаружения
                "discovery_value": bool(host_vec[9]),  
                
                # Уровень доступа
                "access": int(host_vec[10]),  
                
                # One-hot OS
                "os": "linux" if host_vec[11] == 1 else "unknown",  
                
                # Сервисы
                "services": ["ssh"] if host_vec[12] == 1 else [],  
                
                # Процессы
                "processes": ["tomcat"] if host_vec[13] == 1 else []  
            }

            data.append(host_data)

        return data