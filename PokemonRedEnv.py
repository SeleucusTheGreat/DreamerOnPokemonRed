import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
import multiprocessing as mp
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# ==========================================================
# LONG-TERM MEMORY / MAP DIMENSIONS (whole-game scope)
# ==========================================================
RAM_EVENTS_REGION_START = 0xD73E
RAM_EVENTS_REGION_END = 0xD860           
NUM_EVENT_BITS = (RAM_EVENTS_REGION_END - RAM_EVENTS_REGION_START) * 8  # 2320
LTM_REWARD_DIM = NUM_EVENT_BITS + 1       # +1 pokeball flag = 2321

# --- PROLOGUE REWARD SUPPRESSION ---
WEVENTFLAGS_START = 0xD747
EVENT_GOT_POKEDEX = 0x025
PROLOGUE_LTM_END = (WEVENTFLAGS_START - RAM_EVENTS_REGION_START) * 8 + EVENT_GOT_POKEDEX + 1  # 110

# Noisy event flags to exclude from reward, by LTM index
IGNORED_EVENT_INDICES = (1032,)

# Pewter Gym trainer #0
PEWTER_TRAINER_LTM_INDEX = 186
# Brock-battle-start (synthetic flag). Repurposes the slot of a very-late-game
# event a Brock-stage agent never reaches, so the real flag stays 0 and never
# collides with the synthetic write. LTM index = 72 + event_number.
# 2313 = EVENT_BEAT_MEWTWO (event 2241, Cerulean Cave) -- the latest real event
# still inside the captured RAM region 0xD73E-0xD860 (Elite Four/Champion fall
# past 0xD860 and are not captured). Was 737 = EVENT_BEAT_BLAINE.
BROCK_BATTLE_LTM_INDEX = 2313

# ==========================================================
# MAP-CATEGORY CURIOSITY (one-hot 3-bit encoding)
# ==========================================================
# Each map is classified into exactly one category and a new tile discovered on
# that map pays the category's curiosity value. The 3-bit one-hot encoding is
#   [generic, key, special]
# and is used both as the exploration state component (encoded into features) and
# as the target the world-model decoder learns to predict from the latent state.
NUM_MAPS = 256                            # map IDs are a single byte (0x00..0xFF)
MAP_CLASS_DIM = 3                         # one-hot: [generic, key, special]
CURIOSITY_GENERIC = 0.05                  # any unnamed map
CURIOSITY_KEY = 0.20                      # named / "key" map (see MAP_NAMES)
CURIOSITY_SPECIAL = 0.50                  # very special place (gyms, see GYM_MAP_IDS)
CURIOSITY_VALUES = (CURIOSITY_GENERIC, CURIOSITY_KEY, CURIOSITY_SPECIAL)

# Human-readable map names, used in exploration log messages.
MAP_NAMES = {
    12: "Route 1", 1: "Viridian City", 13: "Route 2", 51: "Viridian Forest",
    2: "Pewter City", 54: "Pewter Gym",
    14: "Route 3", 59: "Mt. Moon 1F", 60: "Mt. Moon B1F", 61: "Mt. Moon B2F",
    15: "Route 4", 3: "Cerulean City", 65: "Cerulean Gym",
    35: "Route 24", 36: "Route 25",
    16: "Route 5", 17: "Route 6", 5: "Vermilion City",
    95: "S.S. Anne 1F", 96: "S.S. Anne 2F", 101: "S.S. Anne Captain's Room",
    92: "Vermilion Gym",
    20: "Route 9", 21: "Route 10", 82: "Rock Tunnel 1F", 232: "Rock Tunnel B1F",
    4: "Lavender Town",
    19: "Route 8", 18: "Route 7", 6: "Celadon City", 134: "Celadon Gym",
    135: "Game Corner", 199: "Rocket Hideout B1F", 200: "Rocket Hideout B2F",
    201: "Rocket Hideout B3F", 202: "Rocket Hideout B4F",
    142: "Pokemon Tower 1F", 143: "Pokemon Tower 2F", 144: "Pokemon Tower 3F",
    145: "Pokemon Tower 4F", 146: "Pokemon Tower 5F", 147: "Pokemon Tower 6F",
    148: "Pokemon Tower 7F",
    10: "Saffron City",
    181: "Silph Co. 1F", 207: "Silph Co. 2F", 208: "Silph Co. 3F",
    209: "Silph Co. 4F", 210: "Silph Co. 5F", 211: "Silph Co. 6F",
    212: "Silph Co. 7F", 213: "Silph Co. 8F", 233: "Silph Co. 9F",
    234: "Silph Co. 10F", 235: "Silph Co. 11F", 178: "Saffron Gym",
    23: "Route 12", 24: "Route 13", 25: "Route 14", 26: "Route 15",
    7: "Fuchsia City", 157: "Fuchsia Gym",
    220: "Safari Zone Center", 217: "Safari Zone East", 218: "Safari Zone North",
    219: "Safari Zone West", 222: "Safari Zone Secret House",
    30: "Route 19", 31: "Route 20", 8: "Cinnabar Island",
    165: "Pokemon Mansion 1F", 214: "Pokemon Mansion 2F",
    215: "Pokemon Mansion 3F", 216: "Pokemon Mansion B1F", 166: "Cinnabar Gym",
    45: "Viridian Gym", 33: "Route 22", 34: "Route 23",
    108: "Victory Road 1F", 194: "Victory Road 2F", 198: "Victory Road 3F",
    9: "Indigo Plateau", 174: "Indigo Plateau Lobby",
    245: "Lorelei's Room", 246: "Bruno's Room", 247: "Agatha's Room",
    113: "Lance's Room", 120: "Champion's Room", 118: "Hall of Fame",
}

# Gym map IDs (subset of key maps). Gyms are the "very special" category.
GYM_MAP_IDS = frozenset(mid for mid, name in MAP_NAMES.items() if "Gym" in name)


def map_class_index(map_id):
    """Category of a map id: 2 = special (gym), 1 = key (named), 0 = generic."""
    if map_id in GYM_MAP_IDS:
        return 2
    if map_id in MAP_NAMES:
        return 1
    return 0


# Precomputed lookup tables over every possible map id (0..255), the single source
# of truth shared by the env (curiosity) and the world model (encoder/decoder).
MAP_CLASS_TABLE = np.zeros((NUM_MAPS, MAP_CLASS_DIM), dtype=np.float32)  # one-hot rows
MAP_CLASS_INDEX = np.zeros(NUM_MAPS, dtype=np.int64)                     # class index per map
MAP_CURIOSITY_TABLE = np.zeros(NUM_MAPS, dtype=np.float32)               # curiosity value per map
for _mid in range(NUM_MAPS):
    _cls = map_class_index(_mid)
    MAP_CLASS_TABLE[_mid, _cls] = 1.0
    MAP_CLASS_INDEX[_mid] = _cls
    MAP_CURIOSITY_TABLE[_mid] = CURIOSITY_VALUES[_cls]


# Local exploration grid
GRID_RADIUS = 2
GRID_SIDE = 2 * GRID_RADIUS + 1          # 5
GRID_DIM = GRID_SIDE * GRID_SIDE         # 25
GRID_CENTER_INDEX = GRID_DIM // 2        # 12


class PokemonRedEnv(gym.Env):

    # --- RAM Address Constants ---
    RAM_MAP_ID = 0xD35E
    RAM_PLAYER_Y = 0xD361
    RAM_PLAYER_X = 0xD362

    RAM_PARTY_COUNT = 0xD163
    RAM_PARTY_BASE = 0xD16B
    PKMN_DATA_LENGTH = 44

    RAM_NUM_BAG_ITEMS = 0xD31D
    RAM_BAG_ITEMS_BASE = 0xD31E
    ITEM_POKEBALL = 0x04

    def __init__(self, rom_path, state_path, image_size=96, verbose=False, window="SDL2", speed=0):
        super().__init__()

        self.state_path = state_path
        self.image_size = image_size
        self.verbose = verbose
        self.has_obtained_pokeball = False

        # to mask out flags that were already set so the LTM vector reflects *new* progress.
        self.start_event_bits = np.zeros(NUM_EVENT_BITS, dtype=np.float32)
        self.last_event_bits = np.zeros(NUM_EVENT_BITS, dtype=np.float32)

        # Reward-eligibility mask: zero out the prologue region (idx 0..109, up to GOT_POKEDEX)
        self.reward_keep_mask = np.ones(NUM_EVENT_BITS, dtype=np.float32)
        self.reward_keep_mask[:PROLOGUE_LTM_END] = 0.0
        for idx in IGNORED_EVENT_INDICES:
            self.reward_keep_mask[idx] = 0.0

        # Initialize Emulator
        self.pyboy = PyBoy(
            rom_path, 
            window=window,
            sound_volume=0, 
            cgb=True
        )
        self.pyboy.set_emulation_speed(speed) # 0 = Unlimited speed for training
        
        # Budget Constants
        self.init_steps = 10000


        # --- REWARDS CONSTANTS ---
        self.step_increase = 2500
        self.reward_event_val = 50   
        self.reward_heal_mult = 5
        self.reward_lvl_mult = 5
        
        # Action Space Definition (7 buttons)
        self.buttons =[
            (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
            (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
            (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
            (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
            (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
            (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B)
        ]
        self.action_space = spaces.Discrete(len(self.buttons))
        
        # Observation Space
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(3, self.image_size, self.image_size), 
            dtype=np.uint8
        )
        
        # Persistent Episode Trackers
        self.current_step = 0
        self.max_step_limit = self.init_steps
        
        self.max_events = 0
        self.max_level_reward = 0
        self.last_hp_fraction_sum = 0
        self.last_party_count = 0
        self.level_reward_decay_rate = 0.85
        self.episode_level_ups = 0
        self.brock_defeated = False
    # ==========================================================
    # EMULATOR & RAM HELPERS
    # ==========================================================

    def _has_pokeball(self):
        """Checks the player's Bag RAM to see if a Pokéball (ID: 0x04) is present."""
        num_items = self.pyboy.memory[self.RAM_NUM_BAG_ITEMS]
        num_items = min(max(num_items, 0), 20)
        
        for i in range(num_items):
            item_addr = self.RAM_BAG_ITEMS_BASE + (i * 2)
            item_id = self.pyboy.memory[item_addr]
            if item_id == self.ITEM_POKEBALL:
                return True
        return False

    def _get_item_counts(self):
        """Counts total number of Pokéballs (0x04) and standard Potions (0x14) in the bag."""
        num_items = self.pyboy.memory[self.RAM_NUM_BAG_ITEMS]
        num_items = min(max(num_items, 0), 20)
        pokeballs = 0
        potions = 0
        for i in range(num_items):
            item_addr = self.RAM_BAG_ITEMS_BASE + (i * 2)
            item_id = self.pyboy.memory[item_addr]
            quantity = self.pyboy.memory[item_addr + 1]
            if item_id == self.ITEM_POKEBALL:
                pokeballs += quantity
            elif item_id == 0x14:  # Potion
                potions += quantity
        return [float(pokeballs), float(potions)]

    def _get_team_levels(self):
        """Returns the raw level of each of the 6 party slot positions (0.0 if empty)."""
        party_count = min(max(self.pyboy.memory[self.RAM_PARTY_COUNT], 0), 6)
        levels = [0.0] * 6
        for i in range(party_count):
            base_addr = self.RAM_PARTY_BASE + (i * self.PKMN_DATA_LENGTH)
            levels[i] = float(self.pyboy.memory[base_addr + 0x21])
        return levels

    def _get_current_position(self):
        """Returns the player's current (map_id, x, y) coordinates."""
        map_id = self.pyboy.memory[self.RAM_MAP_ID]
        x = self.pyboy.memory[self.RAM_PLAYER_X]
        y = self.pyboy.memory[self.RAM_PLAYER_Y]
        return (map_id, x, y)

    def _get_explored_grid(self, coord):
        """5x5 agent-centered binary explored grid (row-major); read BEFORE adding the
        current tile to the visited set so the center cell is 0 exactly on a new tile."""
        map_id, x, y = coord
        grid = np.zeros(GRID_DIM, dtype=np.float32)
        idx = 0
        for dy in range(-GRID_RADIUS, GRID_RADIUS + 1):
            for dx in range(-GRID_RADIUS, GRID_RADIUS + 1):
                if (map_id, x + dx, y + dy) in self.episode_visited_tiles:
                    grid[idx] = 1.0
                idx += 1
        return grid

    def _get_party_info(self):
        party_count = min(max(self.pyboy.memory[self.RAM_PARTY_COUNT], 0), 6)
        total_level = 0
        hp_fraction_sum = 0.0
        
        for i in range(party_count):
            base_addr = self.RAM_PARTY_BASE + (i * self.PKMN_DATA_LENGTH)
            
            # Level & HP
            total_level += self.pyboy.memory[base_addr + 0x21]
            current_hp = (self.pyboy.memory[base_addr + 0x01] << 8) | self.pyboy.memory[base_addr + 0x02]
            max_hp = (self.pyboy.memory[base_addr + 0x22] << 8) | self.pyboy.memory[base_addr + 0x23]
            
            if max_hp > 0:
                hp_fraction_sum += (current_hp / max_hp)
                
        return total_level, hp_fraction_sum, party_count

    def _calculate_level_reward(self, total_level):
        """Applies Equation (2) from the paper."""
        if total_level <= 22:
            val = total_level
        else:
            val = ((total_level - 22) / 4) + 22
        return self.reward_lvl_mult * val

    def _get_obs(self):
        screen = self.pyboy.screen.ndarray 
        if screen.shape[-1] == 4:
            screen = screen[:, :, :3]
        img = cv2.resize(screen, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        
        # Transpose from (H, W, C) to PyTorch's expected (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        return np.array(img, dtype=np.uint8)

    def _apply_action(self, action):
        press, release = self.buttons[action]
        self.pyboy.send_input(press)
        self.pyboy.tick(16, True)
        self.pyboy.send_input(release)
        self.pyboy.tick(8, True)

    # ==========================================================
    # GYM METHODS
    # ==========================================================

    def step(self, action):
        brock_before = self.brock_defeated
        self._apply_action(action)
        self.current_step += 1
        sparse_reward = 0.000
        standard_reward = 0.000
        logs = []

        coord = self._get_current_position()
        map_id, x, y = coord

        # --- MAP-CATEGORY TILE CURIOSITY ---
        # A newly discovered tile pays the current map's category value
        # (generic 0.05 / key 0.20 / special 0.50). tier_event flags the higher-value
        # discoveries (key or special maps) so the replay buffer can oversample them.
        explored_grid = self._get_explored_grid(coord)
        curiosity = 0.0
        tier_event = 0.0
        if coord not in self.episode_visited_tiles:
            self.episode_visited_tiles.add(coord)
            curiosity = float(MAP_CURIOSITY_TABLE[map_id])
            if map_id in MAP_NAMES:
                tier_event = 1.0  # new tile on a key/special (named) map

        # --- EVENT REWARD ---
        current_event_bits = self._get_new_event_bits()
        current_events = int(current_event_bits.sum())

        if current_events > self.max_events:
            new_events = current_events - self.max_events
            event_reward_gain = new_events * self.reward_event_val

            newly_on = np.where((current_event_bits > 0.5) & (self.last_event_bits < 0.5))[0]
            for idx in newly_on:
                idx = int(idx)
                if idx == PEWTER_TRAINER_LTM_INDEX:
                    logs.append("\033[1;92m>>> [GYM TRAINER DEFEATED] +50.0 | first Pewter Gym trainer <<<\033[0m")
                elif idx == BROCK_BATTLE_LTM_INDEX:
                    logs.append("\033[1;95m>>> [BROCK BATTLE STARTED] +50.0  (here we go!) <<<\033[0m")
                else:
                    addr = RAM_EVENTS_REGION_START + idx // 8
                    bit = idx % 8
                    logs.append(f"\033[1;95m[EVENT REWARD] Triggered by Memory Address: {hex(addr)}, Bit: {bit} (LTM idx {idx})\033[0m")

            self.max_step_limit += (new_events * self.step_increase)
            self.max_events = current_events
            sparse_reward += event_reward_gain

            if self.verbose:
                logs.append(f"\033[1;93m[EVENT] +{event_reward_gain:.3f} | Budget: {self.max_step_limit}\033[0m")

        self.last_event_bits = current_event_bits

        # Check for Brock defeat milestone reward
        brock_after = (self.pyboy.memory[0xD356] & 1) == 1
        if not brock_before and brock_after:
            self.brock_defeated = True
            sparse_reward += 50.0
            if self.verbose:
                # Bright, hard-to-miss banner (bold bright-yellow on magenta).
                _bar = "\033[1;97;45m" + "=" * 64 + "\033[0m"
                _msg = "\033[1;93;45m" + "  ★★★  BROCK DEFEATED!  +50.0  (total 100.0)  ★★★  ".center(64) + "\033[0m"
                logs.append("")
                logs.append(_bar)
                logs.append(_msg)
                logs.append(_bar)
                logs.append("")

        # PARTY HEALING REWARD
        total_level, hp_fraction_sum, party_count = self._get_party_info()


        # --- HEALING REWARD ---
        if hp_fraction_sum > self.last_hp_fraction_sum and party_count == self.last_party_count:
            if self.last_hp_fraction_sum > 0.0:  # Avoid healing reward on respawn
                heal_gain = hp_fraction_sum - self.last_hp_fraction_sum
                heal_reward = self.reward_heal_mult * heal_gain
                standard_reward += heal_reward
                #if self.verbose:
                #    print(f"[HEAL] +{heal_reward:.3f} | HP recovered: {heal_gain:.2f}")

        self.last_hp_fraction_sum = hp_fraction_sum
        self.last_party_count = party_count

        # --- LEVEL UP REWARD ---
        current_lvl_val = self._calculate_level_reward(total_level)
        if current_lvl_val > self.max_level_reward:
            lvl_gain = current_lvl_val - self.max_level_reward
            standard_reward += lvl_gain
            self.max_level_reward = current_lvl_val
            self.episode_level_ups += 1

            if self.verbose:
                logs.append(f"\033[1;92m[LEVEL] +{lvl_gain:.3f} | Total Party Level up to: {total_level}\033[0m")

        # --- ITEM REWARD ---
        if not self.has_obtained_pokeball:
            if self._has_pokeball():
                self.has_obtained_pokeball = True
                sparse_reward += 50.0
                if self.verbose:
                    logs.append("\033[1;91m[ITEM] +50.0 | Obtained first Pokéball!\033[0m")

        # Total env reward is the sum of the two streams (unchanged externally).
        step_reward = sparse_reward + standard_reward

        # Termination & Obs
        terminated = self.current_step >= self.max_step_limit
        obs = self._get_obs()

        info = {
            "coord": coord,
            "curiosity": curiosity,                           # staleness-scaled tile curiosity
            "tier_event": tier_event,                         # 1.0 on full-rate frontier discovery
            "sparse_reward": sparse_reward,
            "standard_reward": standard_reward,
            "grid": explored_grid,                            # 5x5 explored grid
            "steps": self.current_step,
            "limit": self.max_step_limit,
            "events": current_events,
            "total_level": total_level,
            "ltm_reward": self._get_ltm_reward(),
            "map_id": int(map_id),                            # raw map number (encoded downstream)
            "team_levels": self._get_team_levels(),
            "item_counts": self._get_item_counts(),
            "logs": logs,                                     # verbose msgs for the parent to print
        }

        return obs, step_reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)
            
        self.current_step = 0
        self.max_step_limit = self.init_steps
        self.episode_level_ups = 0  # Reset the decay counter cleanly
        
        # --- Track tile curiosity for the current episode ---
        self.episode_visited_tiles = set()   # (map, x, y) tiles already paid out

        self.pyboy.tick()

        start_coord = self._get_current_position()

        # Seed the spawn tile so the agent isn't paid for the reset position.
        self.episode_visited_tiles.add(start_coord)

        # Snapshot start-state event bits so the LTM vector only reflects new progress.
        self.start_event_bits = self._get_event_bits()
        self.last_event_bits = self._get_new_event_bits()  # all zeros after masking

        # Initial Party and Event Baseline (max_events is now 0: prologue + start flags masked).
        self.max_events = int(self.last_event_bits.sum())
        total_level, hp_fraction_sum, party_count = self._get_party_info()
        self.last_hp_fraction_sum = hp_fraction_sum
        self.last_party_count = party_count
        self.max_level_reward = self._calculate_level_reward(total_level)
        self.has_obtained_pokeball = self._has_pokeball()
        self.brock_defeated = (self.pyboy.memory[0xD356] & 1) == 1

        info = {
            "coord": start_coord,
            "curiosity": 0.0,  # No curiosity on spawn
            "tier_event": 0.0,
            "grid": self._get_explored_grid(start_coord),  # center=1 (spawn already visited)
            "events": self.max_events,
            "total_level": total_level,
            "ltm_reward": self._get_ltm_reward(),
            "map_id": int(start_coord[0]),
            "team_levels": self._get_team_levels(),
            "item_counts": self._get_item_counts(),
            "logs": [],
        }
        return self._get_obs(), info

    def close(self):
        self.pyboy.stop()

    # ==========================================================
    # WHOLE-GAME LONG-TERM MEMORY HELPERS
    # ==========================================================

    def _get_event_bits(self):
        """Raw 2320-length 0/1 vector of the whole-game event-flag RAM region."""
        block = np.frombuffer(
            bytes(self.pyboy.memory[RAM_EVENTS_REGION_START:RAM_EVENTS_REGION_END]),
            dtype=np.uint8,
        )
        bits = np.unpackbits(block, bitorder="little").astype(np.float32)
        # Synthesize the "Brock battle active" bit (wIsInBattle==2, wCurOpponent==234) into the repurposed slot.
        bits[BROCK_BATTLE_LTM_INDEX] = (
            1.0 if (self.pyboy.memory[0xD057] == 2 and self.pyboy.memory[0xD059] == 234) else 0.0
        )
        return bits

    def _get_new_event_bits(self):
        """Reward-eligible event bits: start-state, prologue and ignored flags masked out."""
        return self._get_event_bits() * (1.0 - self.start_event_bits) * self.reward_keep_mask

    def _get_ltm_reward(self):
        """Whole-game long-term reward memory: new event bits + pokeball flag (LTM_REWARD_DIM)."""
        vec = np.empty(LTM_REWARD_DIM, dtype=np.float32)
        vec[:NUM_EVENT_BITS] = self._get_new_event_bits()
        vec[NUM_EVENT_BITS] = 1.0 if self.has_obtained_pokeball else 0.0
        return vec

# The one and only start state allowed. Every environment loads from this file.
START_STATE = "PokemonRed.Start.state"


# ==========================================================
# PARALLEL (SUBPROCESS) ENVIRONMENT EXECUTION
# ==========================================================
def _env_worker(conn, rom_path, state_path, image_size, window, speed, verbose):
    """Owns one PyBoy emulator and serves reset/step/close over a pipe; module-level
    so it is importable under the 'spawn' start method (required on Windows)."""
    try:
        env = PokemonRedEnv(
            rom_path=rom_path, state_path=state_path, image_size=image_size,
            window=window, speed=speed, verbose=verbose,
        )
    except Exception as e:  # surface construction failures to the parent
        try:
            conn.send(("__error__", repr(e)))
        finally:
            conn.close()
        return

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                conn.send(env.step(data))
            elif cmd == "reset":
                conn.send(env.reset())
            elif cmd == "close":
                break
            else:
                conn.send(("__error__", f"unknown command {cmd!r}"))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        conn.close()


class ParallelPokemonEnvs:
    """Runs each PyBoy emulator in its own process; step() dispatches all active
    commands before reading any reply, so the emulators advance in parallel."""

    def __init__(self, num_envs, rom_path, state_path, image_size=96,
                 window="null", speed=0, verbose=False):
        ctx = mp.get_context("spawn")  # PyBoy is not fork-safe; spawn works on Windows too
        self.num_envs = num_envs
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, image_size, image_size), dtype=np.uint8)
        self._parents = []
        self._procs = []
        for _ in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_env_worker,
                args=(child, rom_path, state_path, image_size, window, speed, verbose),
                daemon=True,
            )
            proc.start()
            child.close()  # only the worker keeps the child end
            self._parents.append(parent)
            self._procs.append(proc)

    def __len__(self):
        return self.num_envs

    @staticmethod
    def _check(result):
        if (isinstance(result, tuple) and len(result) == 2
                and isinstance(result[0], str) and result[0] == "__error__"):
            raise RuntimeError(f"Env worker error: {result[1]}")
        return result

    def reset(self):
        for p in self._parents:
            p.send(("reset", None))
        return [self._check(p.recv()) for p in self._parents]

    def reset_one(self, i):
        self._parents[i].send(("reset", None))
        return self._check(self._parents[i].recv())

    def step(self, action_idxs, active=None):
        if active is None:
            active = [True] * self.num_envs
        # Dispatch all step commands first (emulators run concurrently)...
        for i, p in enumerate(self._parents):
            if active[i]:
                p.send(("step", int(action_idxs[i])))
        # ...then collect the results.
        results = [None] * self.num_envs
        for i, p in enumerate(self._parents):
            if active[i]:
                results[i] = self._check(p.recv())
        return results

    def close(self):
        for p in self._parents:
            try:
                p.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


class SequentialPokemonEnvs:
    """Same interface as ParallelPokemonEnvs but runs every emulator in-process,
    one after another. Kept as a debugging / single-process fallback."""

    def __init__(self, envs):
        self._envs = envs
        self.num_envs = len(envs)
        self.action_space = envs[0].action_space
        self.observation_space = envs[0].observation_space

    def __len__(self):
        return self.num_envs

    def reset(self):
        return [e.reset() for e in self._envs]

    def reset_one(self, i):
        return self._envs[i].reset()

    def step(self, action_idxs, active=None):
        if active is None:
            active = [True] * self.num_envs
        results = [None] * self.num_envs
        for i, e in enumerate(self._envs):
            if active[i]:
                results[i] = e.step(int(action_idxs[i]))
        return results

    def close(self):
        for e in self._envs:
            e.close()


def create_envs(num_envs, rom_path, state_dir="StartingFiles", wind="null", speed=0, parallel=True):
    """Create `num_envs` environments, all loading from PokemonRed.Start.state (the only
    supported start state); parallel=True uses one process per emulator, False is a
    single-process debugging fallback with the same interface."""
    state_path = os.path.join(state_dir, START_STATE)
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Required start state not found: {state_path}. "
            f"Only '{START_STATE}' is supported."
        )

    print(f"Initializing {num_envs} PyBoy Environments (all using {START_STATE}, "
          f"parallel={parallel})...")

    if parallel:
        return ParallelPokemonEnvs(
            num_envs=num_envs, rom_path=rom_path, state_path=state_path,
            window=wind, speed=speed, verbose=True,
        )

    # In-process fallback.
    envs = [
        PokemonRedEnv(
            rom_path=rom_path, state_path=state_path,
            verbose=True, window=wind, speed=speed,
        )
        for _ in range(num_envs)
    ]
    return SequentialPokemonEnvs(envs)