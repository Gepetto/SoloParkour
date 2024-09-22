####################
# Terrain generator
####################
import numpy as np

from isaacgym.terrain_utils import *

class Terrain:
    """Class to handle terrain creation, usually a bunch of subterrains surrounding by a flat border area.

    Subterrains are spread on a given number of columns and rows. Each column might correspond to a given
    type of subterrains (slope, stairs, ...) and each row to a given difficulty (from easiest to hardest).
    """

    def __init__(self, cfg, num_robots) -> None:

        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return

        # Create subterrains on the fly based on Isaac subterrain primitives

        # Retrieving proportions of each kind of subterrains
        keys = list(cfg["terrainProportions"].keys())
        vals = list(cfg["terrainProportions"].values())
        self.terrain_keys = []
        self.terrain_proportions = []
        sum = 0.0
        for key, val in zip(keys, vals):
            if val != 0.0:
                sum += float(val)
                self.terrain_keys.append(key)
                self.terrain_proportions.append(np.round(sum, 2))

        self.horizontal_scale = 0.05  # Resolution of the terrain height map
        self.border_size = 8.0  # Size of the flat border area all around the terrains
        self.env_length = cfg["mapLength"]  # Length of subterrains
        self.env_width = cfg["mapWidth"]  # Width of subterrains
        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        # Number of height map cells for each subterrain in width and length
        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.float32)
        self.ceilings = np.zeros((self.tot_rows , self.tot_cols), dtype=np.float32)

        self.boxes = []
        self.vertical_scale = 0.005
        if cfg["curriculum"]:
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
            #self.randomized_terrain()
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])
        if len(self.boxes) > 0:
            for box in self.boxes:
                self.vertices, self.triangles = combine_trimeshes((self.vertices, self.triangles), box)
        self.heightsamples = self.height_field_raw

    def randomized_terrain(self):
        """Spawn random subterrain without ordering them by type or difficulty
        according to their rows and columns"""

        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            elif choice < 1.:
                discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def curiculum(self, num_robots, num_terrains, num_levels):
        """Spawn subterrain ordering them by type (one type per column)
        and by difficulty (first row is easiest, last one is hardest)
        """

        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                    length=self.length_per_env_pixels, # seems to be inverted
                                    width=self.width_per_env_pixels,
                                    vertical_scale=self.vertical_scale,
                                    horizontal_scale=self.horizontal_scale)
                difficulty = i / (num_levels-1.0)
                choice = j / num_terrains
                lava_depth=-np.random.uniform(0.7, 1.3)
                boxes=None
                ceiling = 0.4

                k = 0
                while k < len(self.terrain_proportions) and choice >= self.terrain_proportions[k]:
                    k += 1
                if k == len(self.terrain_proportions):
                    # The sum of terrain proportions is not >= 1
                    # Defaulting to flat ground
                    continue

                if self.terrain_keys[k] == "gap_parkour":
                    gap_length = 0.15 + i*0.05
                    gap_length = np.round(gap_length, 2)

                    gap_parkour(
                        terrain,
                        lava_depth=lava_depth,
                        gap_length=gap_length,
                        gap_platform_height=0.1
                    )
                    add_roughness(terrain, np.random.uniform(0.01, 0.03))
                elif self.terrain_keys[k] == "jump_parkour":
                    height = 0.05 + 0.37*difficulty
                    jump_parkour(
                        terrain,
                        lava_depth=lava_depth,
                        height=height,
                    )
                    add_roughness(terrain, np.random.uniform(0.01, 0.03))

                elif self.terrain_keys[k] == "stairs_parkour":
                    stairs_parkour(
                        terrain,
                        lava_depth=lava_depth,
                        height=0.02 + 0.18*difficulty,

                    )
                    add_roughness(terrain, np.random.uniform(0.01, 0.03))

                elif self.terrain_keys[k] == "hurdle_parkour":
                    hurdle_parkour(
                        terrain,
                        lava_depth=lava_depth,
                        height=0.05 + 0.3*difficulty,
                    )
                    add_roughness(terrain, np.random.uniform(0.01, 0.03))
                elif self.terrain_keys[k] == "crawl_parkour":
                    ceiling = 0.34 - 0.08*difficulty
                    boxes = crawl_parkour(
                        terrain,
                        lava_depth=lava_depth,
                        height=ceiling
                    )
                    add_roughness(terrain, np.random.uniform(0.01, 0.03))
                elif self.terrain_keys[k] == "random_uniform":                                                                                                                                                
                    add_roughness(terrain, np.random.uniform(0.01, 0.03))

                else:
                    # Flat ground
                    pass

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw.transpose() # Seems that width and length are inverter in SubTerrain

                if boxes is not None:
                    for box in boxes:
                        box = move_trimesh(box, np.array([[
                            self.border_size+i*self.env_length, self.border_size + (j+0.5) * self.env_width, 0.0
                        ]]))
                        self.boxes.append(box)

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map +=1

                env_origin_x = (i + 0.0) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                env_origin_z = np.random.uniform(0.0, 0.4)
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
                self.ceilings[i, j] = ceiling

def add_roughness(terrain, noise_magnitude=0.02):
    random_uniform_terrain(
        terrain,
        min_height=-noise_magnitude,
        max_height=noise_magnitude,
        step=0.005,
        downsampled_scale=0.075,
    )

def gap_parkour(terrain, platform_length=1., lava_width=0.5, lava_depth=-1.0, gap_length=0.5, gap_platform_length_min=1.25, gap_platform_length_max=1.5, gap_platform_height=0.05):
    platform_length = int(platform_length / terrain.horizontal_scale)
    lava_width = int(lava_width / terrain.horizontal_scale)
    lava_depth = int(lava_depth / terrain.vertical_scale)

    gap_platform_length_min = int(gap_platform_length_min / terrain.horizontal_scale)
    gap_platform_length_max = int(gap_platform_length_max / terrain.horizontal_scale)

    gap_length = int(gap_length / terrain.horizontal_scale)
    gap_platform_height = int(gap_platform_height / terrain.vertical_scale)

    # add gap
    start_gap = platform_length
    while start_gap + gap_length <= terrain.length - platform_length//2:
        gap_platform_length = np.random.randint(gap_platform_length_min, gap_platform_length_max)
        terrain.height_field_raw[:, start_gap:start_gap+gap_length] = lava_depth
        # randomize gap platform height
        if start_gap + gap_length + gap_platform_length <= terrain.length - platform_length //2:
            #terrain.height_field_raw[:, start_gap+gap_length:start_gap+gap_length+gap_platform_length] = np.random.randint(-gap_platform_height, gap_platform_height)
            terrain.height_field_raw[:, start_gap+gap_length:start_gap+gap_length+gap_platform_length] = -gap_platform_height

        start_gap += gap_length + gap_platform_length

    # the floor is lava
    terrain.height_field_raw[0:lava_width, 0:terrain.length] = lava_depth
    terrain.height_field_raw[-lava_width:, 0:terrain.length] = lava_depth

def jump_parkour(terrain, platform_length=1.25, lava_width=0.5, lava_depth=-1.0, height=0.5, height_platform_length=1.5):
    platform_length = int(platform_length / terrain.horizontal_scale)
    lava_width = int(lava_width / terrain.horizontal_scale)
    lava_depth = int(lava_depth / terrain.vertical_scale)

    height_platform_length = int(height_platform_length / terrain.horizontal_scale)
    height = int(height / terrain.vertical_scale)

    # Version with 2 jumps
    #terrain.height_field_raw[:, platform_length:platform_length+3*height_platform_length] = height
    #terrain.height_field_raw[:, platform_length+height_platform_length:platform_length+2*height_platform_length] = 2*height

    # Version with 3 jumps
    terrain.height_field_raw[:, 1*platform_length:6*platform_length] = 1*height
    terrain.height_field_raw[:, 2*platform_length:5*platform_length] = 2*height
    terrain.height_field_raw[:, 3*platform_length:4*platform_length] = 3*height
 
    # the floor is lava
    terrain.height_field_raw[0:lava_width, 0:terrain.length] = lava_depth
    terrain.height_field_raw[-lava_width:, 0:terrain.length] = lava_depth

def stairs_parkour(terrain, platform_length=1., lava_width=0.5, lava_depth=-1.0, height=0.18, width=0.3, stairs_platform_length=1.25):
    platform_length = int(platform_length / terrain.horizontal_scale)
    lava_width = int(lava_width / terrain.horizontal_scale)
    lava_depth = int(lava_depth / terrain.vertical_scale)

    stairs_platform_length = int(stairs_platform_length / terrain.horizontal_scale)
    height = int(height / terrain.vertical_scale)
    width = int(width / terrain.horizontal_scale)

    start = platform_length
    stop = terrain.length - platform_length//2
    curr_height = height
    while stop - start > platform_length:
        terrain.height_field_raw[:, start:stop] = curr_height
        curr_height += height
        start += width
        stop -= width
    
    # the floor is lava
    terrain.height_field_raw[0:lava_width, 0:terrain.length] = lava_depth
    terrain.height_field_raw[-lava_width:, 0:terrain.length] = lava_depth


def hurdle_parkour(terrain, platform_length=1.5, lava_width=0.5, lava_depth=-1.0, height=0.2, width_min=0.3, width_max=0.5):
    platform_length = int(platform_length / terrain.horizontal_scale)
    lava_width = int(lava_width / terrain.horizontal_scale)
    lava_depth = int(lava_depth / terrain.vertical_scale)

    height = int(height / terrain.vertical_scale)
    width_min = int(width_min / terrain.horizontal_scale)
    width_max = int(width_max / terrain.horizontal_scale)

    start = platform_length
    width = np.random.randint(width_min, width_max)
    while start + platform_length + width <= terrain.length - platform_length//2:
        terrain.height_field_raw[:, start:start+width] = height
        start += platform_length + width
        width = np.random.randint(width_min, width_max)

    
    # the floor is lava
    terrain.height_field_raw[0:lava_width, 0:terrain.length] = lava_depth
    terrain.height_field_raw[-lava_width:, 0:terrain.length] = lava_depth

def crawl_parkour(terrain, platform_length=2.0, lava_width=0.5, lava_depth=-1.0, height=0.2, depth=1.0, width=3.0, height_step=0.15):
    # First put the barriers
    boxes = []
    boxes += box_trimesh(np.array([depth, width, 0.5]), np.array([2.5, 0.0, height+0.25])),
    boxes += box_trimesh(np.array([depth, width, 0.5]), np.array([6.5, 0.0, height+0.25+height_step])),

    # Then create the heightmap
    platform_length = int(platform_length / terrain.horizontal_scale)
    lava_width = int(lava_width / terrain.horizontal_scale)
    lava_depth = int(lava_depth / terrain.vertical_scale)

    height = int(height / terrain.vertical_scale)
    height_step = int(height_step / terrain.vertical_scale)
    depth = int(depth / terrain.horizontal_scale)
    
    terrain.height_field_raw[:, int(6.0/terrain.horizontal_scale):int(7.0/terrain.horizontal_scale)] = 1*height_step

    # the floor is lava
    terrain.height_field_raw[0:lava_width, 0:terrain.length] = lava_depth
    terrain.height_field_raw[-lava_width:, 0:terrain.length] = lava_depth
    
    return boxes




def box_trimesh(
        size, # float [3] for x, y, z axis length (in meter) under box frame
        center_position, # float [3] position (in meter) in world frame
    ):

    vertices = np.empty((8, 3), dtype= np.float32)
    vertices[:] = center_position
    vertices[[0, 4, 2, 6], 0] -= size[0] / 2
    vertices[[1, 5, 3, 7], 0] += size[0] / 2
    vertices[[0, 1, 2, 3], 1] -= size[1] / 2
    vertices[[4, 5, 6, 7], 1] += size[1] / 2
    vertices[[2, 3, 6, 7], 2] -= size[2] / 2
    vertices[[0, 1, 4, 5], 2] += size[2] / 2

    triangles = -np.ones((12, 3), dtype= np.uint32)
    triangles[0] = [0, 2, 1] #
    triangles[1] = [1, 2, 3]
    triangles[2] = [0, 4, 2] #
    triangles[3] = [2, 4, 6]
    triangles[4] = [4, 5, 6] #
    triangles[5] = [5, 7, 6]
    triangles[6] = [1, 3, 5] #
    triangles[7] = [3, 7, 5]
    triangles[8] = [0, 1, 4] #
    triangles[9] = [1, 5, 4]
    triangles[10]= [2, 6, 3] #
    triangles[11]= [3, 6, 7]

    return vertices, triangles

def combine_trimeshes(*trimeshes):
    if len(trimeshes) > 2:
        return combine_trimeshes(
            trimeshes[0],
            combine_trimeshes(trimeshes[1:])
        )

    # only two trimesh to combine
    trimesh_0, trimesh_1 = trimeshes
    if trimesh_0[1].shape[0] < trimesh_1[1].shape[0]:
        trimesh_0, trimesh_1 = trimesh_1, trimesh_0
    
    trimesh_1 = (trimesh_1[0], trimesh_1[1] + trimesh_0[0].shape[0])
    vertices = np.concatenate((trimesh_0[0], trimesh_1[0]), axis= 0)
    triangles = np.concatenate((trimesh_0[1], trimesh_1[1]), axis= 0)

    return vertices, triangles

def move_trimesh(trimesh, move: np.ndarray):
    trimesh = list(trimesh)
    trimesh[0] += move
    return tuple(trimesh)
