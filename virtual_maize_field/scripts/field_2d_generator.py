import sys
import jinja2
import numpy as np
import cv2
import os
import rospkg
from matplotlib import pyplot as plt

from world_description import WorldDescription
from row_segments import StraightSegment, CurvedSegment, IslandSegment


class Field2DGenerator:
    def __init__(self, world_description=WorldDescription()):
        self.wd = world_description
        np.random.seed(self.wd.structure["params"]["seed"])

    def render_matplotlib(self):
        # Segments
        for segment in self.segments:
            segment.render()

        # Plants
        plt.scatter(self.placements[:, 0], self.placements[:, 1], color="c", marker=".")

    def generate(self):
        self.chain_segments()
        self.center_plants()
        self.seed_weeds()
        self.generate_ground()
        self.fix_gazebo()
        self.render_to_template()
        return [self.sdf, self.heightmap]

    def chain_segments(self):
        # Generate start points
        rows = self.wd.rows_left + self.wd.rows_right
        x_start = (
            -self.wd.row_width / 2 - self.wd.row_width * (self.wd.rows_left - 1)
            if self.wd.rows_left > 0
            else self.wd.row_width / 2
        )
        x_end = (
            self.wd.row_width / 2 + self.wd.row_width * (self.wd.rows_right - 1)
            if self.wd.rows_right > 0
            else -self.wd.row_width / 2
        )
        current_p = np.array([np.linspace(x_start, x_end, rows), np.repeat(0, rows)]).T
        current_dir = [0, 1]

        # Placement parameters
        offset = None
        self.placements = [[] for _ in range(rows)]

        # Chain all segments from the world description
        self.segments = []
        for segment in self.wd.structure["segments"]:
            if segment["type"] == "straight":
                seg = StraightSegment(
                    current_p,
                    current_dir,
                    self.wd.structure["params"],
                    segment["length"],
                )
            elif segment["type"] == "curved":
                seg = CurvedSegment(
                    current_p,
                    current_dir,
                    self.wd.structure["params"],
                    segment["radius"],
                    segment["curve_dir"],
                    segment["arc_measure"],
                )
            elif segment["type"] == "island":
                seg = IslandSegment(
                    current_p,
                    current_dir,
                    self.wd.structure["params"],
                    segment["radius"],
                    segment["island_model"],
                    segment["island_model_radius"],
                    segment["island_row"],
                )
            else:
                raise ValueError("Unknown segment type. [" + segment["type"] + "]")

            # Collect all plant placements
            seg_placements, offset = seg.placements(offset)
            for row, seg_row in zip(self.placements, seg_placements):
                row.extend(seg_row)

            # Update current end points, direction and row length
            current_p, current_dir = seg.end()
            self.segments.append(seg)

        self.placements = np.vstack(self.placements)

        # TODO This is a an unbounded normal distribution, which causes problems with
        # the plant placements. There will be some outliers each time, because the
        # number of plants is so heigh
        self.placements += np.random.normal(
            scale=self.wd.structure["params"]["plant_placement_error_max"],
            size=self.placements.shape,
        )

    # Because the heightmap must be square and has to have a side length of 2^n + 1
    # this means that we could have smaller maps, by centering the placements around 0,0
    def center_plants(self):
        x_min = self.placements[:, 0].min()
        y_min = self.placements[:, 1].min()

        self.placements -= np.array([x_min, y_min])

        x_max = self.placements[:, 0].max()
        y_max = self.placements[:, 1].max()

        self.placements -= np.array([x_max, y_max]) / 2

    # The function calculates the placements of the weed plants and
    # stores them under self.weeds : np.array([[x,y],[x,y],...])
    def seed_weeds(self):
        self.weeds = np.array([])
        pass

    def generate_ground(self):
        # Calculate image resolution
        metric_x_min = self.placements[:, 0].min()
        metric_x_max = self.placements[:, 0].max()
        metric_y_min = self.placements[:, 1].min()
        metric_y_max = self.placements[:, 1].max()

        metric_width = metric_x_max - metric_x_min + 4
        metric_height = metric_y_max - metric_y_min + 4

        resolution = 0.02 #FIXME dynamic resolution 
        min_image_size = int(
            np.ceil(max(metric_width / resolution, metric_height / resolution))
        )
        # gazebo expects heightmap in format 2**n -1
        image_size = int(2 ** np.ceil(np.log2(min_image_size))) + 1

        # Generate noise
        heightmap = np.zeros((image_size, image_size))

        n = 0
        while 2 ** n < image_size:
            heightmap += (
                cv2.resize(
                    np.random.random((image_size // 2 ** n, image_size // 2 ** n)),
                    (image_size, image_size),
                )
                * (n + 1) ** 2
            )
            n += 1

        # Normalize heightmap
        heightmap -= heightmap.min()
        heightmap /= heightmap.max()

        DITCH_DEPTH = 0.3  #m
        DITCH_DISTANCE = 2 #m
        DITCH_WIDTH = 0.4  #m
        max_elevation = self.wd.structure["params"]["ground_max_elevation"]

        self.heightmap_elevation = DITCH_DEPTH + (max_elevation / 2)

        heightmap *= ((max_elevation) / self.heightmap_elevation)

        field_mask = np.ones((image_size, image_size))

        offset = image_size // 2
        def metric_to_pixel(pos):
            return int(pos // resolution) + offset

        # Make plant placements flat and save the heights for the sdf renderer
        self.placements_ground_height = []
        for mx, my in self.placements:
            px = metric_to_pixel(mx)
            py = metric_to_pixel(my)

            field_mask = cv2.circle(field_mask, (px, py), int((DITCH_DISTANCE + DITCH_WIDTH) / resolution), 0, -1)

            height = heightmap[py, px]
            #height = 1
            heightmap = cv2.circle(heightmap, (px, py), 3, height, -1)
            self.placements_ground_height.append(
                self.heightmap_elevation * height
            )

        # raise field to create a ditch
        for mx, my in self.placements:
            px = metric_to_pixel(mx)
            py = metric_to_pixel(my)
            field_mask = cv2.circle(field_mask, (px, py), int(DITCH_DISTANCE / resolution), 1, -1)

        blur_size = (int(0.2 / resolution) // 2) * 2 + 1
        field_mask = cv2.GaussianBlur(field_mask, (blur_size, blur_size), 0)

        heightmap += ((DITCH_DEPTH - (max_elevation / 2)) / self.heightmap_elevation) * field_mask

        assert(heightmap.max() <= 1)
        assert(heightmap.min() >= 0)

        # Convert to grayscale
        self.heightmap = (255 * heightmap[::-1, :]).astype(np.uint8)

        self.metric_size = image_size * resolution
        # Calc heightmap position. Currently unused, overwritten in @ref fix_gazebo
        self.heightmap_position = [
            metric_x_min - 2 + 0.5 * self.metric_size,
            metric_y_min - 2 + 0.5 * self.metric_size,
        ]

    def fix_gazebo(self):
        # move the plants to the center of the flat circles
        self.placements -= 0.01

        # set heightmap position to origin, see gazebo issue 2996:
        # https://github.com/osrf/gazebo/issues/2996
        self.heightmap_position = [0, 0]

    def render_to_template(self):
        def into_dict(xy, ground_height, radius, height, mass, index):
            coordinate = dict()
            coordinate["type"] = np.random.choice(
                self.wd.structure["params"]["plant_types"].split(",")
            )
            inertia = dict()
            inertia["ixx"] = (mass * (3 * radius ** 2 + height ** 2)) / 12.0
            inertia["iyy"] = (mass * (3 * radius ** 2 + height ** 2)) / 12.0
            inertia["izz"] = (mass * radius ** 2) / 2.0
            coordinate["inertia"] = inertia
            coordinate["mass"] = mass
            coordinate["x"] = xy[0]
            coordinate["y"] = xy[1]
            coordinate["z"] = ground_height
            coordinate["radius"] = (
                radius
                + (2 * np.random.rand() - 1)
                * self.wd.structure["params"]["plant_radius_noise"]
            )
            if coordinate["type"] == "cylinder":
                coordinate["height"] = height
            coordinate["name"] = "{}_{:04d}".format(coordinate["type"], index)
            coordinate["yaw"] = np.random.rand() * 2.0 * np.pi
            return coordinate

        coordinates = [
            into_dict(
                plant,
                self.placements_ground_height[i],
                self.wd.structure["params"]["plant_radius"],
                self.wd.structure["params"]["plant_height_min"],
                self.wd.structure["params"]["plant_mass"],
                i,
            )
            for i, plant in enumerate(self.placements)
        ]

        pkg_path = rospkg.RosPack().get_path("virtual_maize_field")
        template_path = os.path.join(pkg_path, "scripts/field.world.template")
        template = open(template_path).read()
        template = jinja2.Template(template)
        self.sdf = template.render(
            coordinates=coordinates,
            seed=self.wd.structure["params"]["seed"],
            heightmap={
                "size": self.metric_size,
                "pos": {
                    "x": self.heightmap_position[0],
                    "y": self.heightmap_position[1],
                },
                "max_elevation": self.heightmap_elevation,
            },
        )
