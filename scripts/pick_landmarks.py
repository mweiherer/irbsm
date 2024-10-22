import argparse
import pyvista as pv
import numpy as np
import os


class PointPicker:
    '''
    Interactive application that allows picking points on a triangle mesh 
    or point cloud. Resulting point positions can be exported to a csv file.
    Important mouse and key controlls: 
        - Right click: Picks a point
        - 's': Saves clicked points to csv file
        - 'd': Deletes last clicked point
        - 'e': Closes application
    :param pc_file: The path to the mesh/point cloud points should be picked on 
    :param output_dir: Optional output directory where to save picked points
    '''
    def __init__(self, pc_file, output_dir = './'):
        self.pc_file = pc_file
        self.output_dir = output_dir
     
        self.point_cloud = pv.read(pc_file)
        self.is_mesh = True if len(self.point_cloud.faces) > 0 else False
       
        self.plotter = pv.Plotter(notebook = False)
        self.plotter.track_click_position(self.pick_points, side = 'right')
        self.plotter.add_key_event('s', self.save)
        self.plotter.add_key_event('d', self.delete_last)
        self.plotter.set_background(color = 'black')

    def pick_points(self, *args):
        picked_pt = np.array(self.plotter.pick_mouse_position())
         
        if self.is_mesh:
            direction = picked_pt - self.plotter.camera_position[0]
            direction = direction / np.linalg.norm(direction)

            start = picked_pt - 1000 * direction
            end = picked_pt + 10000 * direction
            point, _ = self.point_cloud.ray_trace(start, end, first_point = True)
        else:
            idx = self.point_cloud.find_closest_point(picked_pt)
            point = self.point_cloud.points[idx]

        if len(point) > 0:
            self.clicked_points.append(point)
            self.sphere_actors.append(self.plotter.add_mesh(
                pv.Sphere(radius = 0.005 * self.point_cloud.length, center = point), color = 'red'))
        
        return

    def save(self):
        '''
        Saves picked points, if any, to disk.
        '''
        fname = os.path.join(self.output_dir, self.pc_file.split('/')[-1][:-4])

        if len(self.clicked_points) > 0:
            np.savetxt(f'{fname}.csv', self.clicked_points, delimiter = ',')
            print(f'Saved {len(self.clicked_points)} points under {fname}.csv.')
       
    def delete_last(self):
        '''
        Deletes the last picked point.
        '''
        if len(self.clicked_points) > 0:
            self.clicked_points = self.clicked_points[:-1]
            self.plotter.remove_actor(self.sphere_actors[-1])
            self.sphere_actors = self.sphere_actors[:-1]

    def start(self):
        self.clicked_points = []
        self.sphere_actors = []

        self.mesh_actor = self.plotter.add_mesh(self.point_cloud, color = 'w')
        self.plotter.view_xy()
        self.plotter.show()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('point_cloud', type = str, 
                           help = 'Path to point cloud or mesh where landmarks should be clicked on.')
    argparser.add_argument('--output_dir', type = str, 
                           help = 'The directory to save the landmarks file in.',
                           default = './')
    
    args, _ = argparser.parse_known_args()

    picker = PointPicker(pc_file = args.point_cloud, output_dir = args.output_dir)
    picker.start()