import bpy
import math
import numpy as np

import os
os.chdir('/home/mjohnsrud/drive/barsoe/blender')


# for obj in bpy.data.objects:
#     print(obj.name)
    

data_name = ['x', 'y', 'z']
xyz = [np.loadtxt(name + '.csv', delimiter=',') for name in data_name]
vertices = np.loadtxt('data.csv', delimiter=',')
# # make mesh
# vertices = [
#     (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
#     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
#     ]

edges = []
faces = []




name = 'from_data'


# if bpy.context.object.mode == 'EDIT':
#     bpy.ops.object.mode_set(mode='OBJECT')


bpy.ops.object.select_all(action='DESELECT')



objs = [obj for obj in bpy.data.objects if obj.name[:len(name)]==name]
bpy.ops.object.delete({'selected_objects': objs})



# emptyMesh = bpy.data.meshes.new('emptyMesh'):                 
new_mesh = bpy.data.meshes.new(name)
new_mesh.from_pydata(vertices, edges, faces)
new_mesh.update()


theObj = bpy.data.objects.new(name, new_mesh)
bpy.context.collection.objects.link(theObj)
